import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from einops import rearrange, repeat
from torch.nn.utils import remove_weight_norm

from wespeaker.models.ssl.modules import GradMultiply
from wespeaker.models.ssl_backend import *
from wespeaker.models.ssl.MCWavLM import *
from wespeaker.models.fusion_modules import Identity, MicWeightedAvg, AttAcrossChans, CoAttention, TACFusion

class WavLM_Base_MC_MHFA(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim,
                 group,cnn_scale=0.0,
                 layer_drop=0.05,
                 n_chans=4,
                 fusion_modules='TACFusion,MicWeightedAvg:ref_mic_w=(float)0.25',
                 cnn_out_fusion='TACFusion',
                 rep_fusion=None,
                 feature_grad_mult=0.08):
        super(WavLM_Base_MC_MHFA, self).__init__()
        checkpoint = torch.load(model_path)
        print(pooling)
        checkpoint['cfg']['encoder_layerdrop']=layer_drop
        checkpoint['cfg']['feature_grad_mult']=cnn_scale

        self.feature_grad_mult = feature_grad_mult
        self.n_chans = n_chans
        self.fusion_modules = nn.ModuleList(self._parse_fusion_module_string(fusion_modules))
        self.cnn_out_fusion = self._parse_fusion_module_string(cnn_out_fusion)[0]

        if rep_fusion is None or rep_fusion == 'first':
            self.rep_fusion_inst = list()
            self.rep_fusion_fn = self._extract_layer_rep_first
        elif rep_fusion.startswith('wavg'):
            # format: "wavg:0.9"
            name_weight = rep_fusion.split(':')
            comb_params = dict()
            if len(name_weight) == 1:
                weight = 0.9
            else:
                weight_params = name_weight[1].split(';')
                weight = float(weight_params[0])
                for ps in weight_params[1:]:
                    k,v = ps.split('=')
                    m = re.match(r'\(([^\)]+)\)(.+)', v)
                    if not m:
                        raise ValueError('Regex fail on "{}", try checking the fusion string modules format: "{}"'.format(v, string))
                    dtype = m.group(1)
                    v = m.group(2)
                    v = eval(dtype)(v)
                    comb_params[k] = v

            self.rep_fusion_inst = nn.ModuleList([
                MicWeightedAvg(n_chans, ref_mic_w=weight, **comb_params) for _ in range(sum([not fm.miso for fm in self.fusion_modules])+(not self.cnn_out_fusion.miso)) # TODO: from orig impl, modify
            ])
            self.rep_fusion_fn = self._extract_layer_rep_wavg
        else:
            raise ValueError('Unsupported representation fusion: "{}"'.format(rep_fusion))

        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(
                cfg,
                n_chans=n_chans, # NEW
                trans_out_fusion=self.fusion_modules,
                cnn_out_fusion=self.cnn_out_fusion,
                grad_mult_global=self.feature_grad_mult,
            )
        # We apply scaling of fusion modules weights to promote close-to-identity behaviour
        if hasattr(self.cnn_out_fusion, 'scale_weights'):
            self.cnn_out_fusion.scale_weights()
        for m in self.fusion_modules:
            if hasattr(m, 'scale_weights'):
                m.scale_weights()

        self.loadParameters(checkpoint['model'])
        if pooling == 'MHFA':
            self.back_end = MHFA(head_nb=head_nb,outputs_dim=embed_dim)
        elif pooling == 'G_MHFA':
            self.back_end = MHFA_Group(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'QKV':
            self.back_end = MHFA_Dotproduct(compression_dim=256, outputs_dim=embed_dim)
        elif pooling == 'G_MHFA_MQSKMV':
            self.back_end = MHFA_Group_MQ_SK_MV(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'G_MHFA_MQMKSV':
            self.back_end = MHFA_Group_MQ_MK_SV(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'G_MHFA_Conv2D':
            self.back_end = MHFA_Group_Conv2D(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'MHFA_Context':
            self.back_end = MHFA_context(head_nb=head_nb,outputs_dim=embed_dim)
        elif pooling == 'G_MHFA_Conv2D_MeanStd':
            self.back_end = MHFA_Group_Conv2D_MeanStd(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'TSTP':
            self.back_end = TSTP(outputs_dim=embed_dim)
        elif pooling == 'ASTP':
            self.back_end = ASTP(outputs_dim=embed_dim)
        elif pooling == 'Last_ASTP':
            self.back_end = Last_ASTP(outputs_dim=embed_dim)
        elif pooling == 'CorrelationPoolingDrop':
            self.back_end = CorrelationPoolingDrop(outputs_dim=embed_dim)
        elif pooling == 'CorrelationPooling':
            self.back_end = CorrelationPooling(outputs_dim=embed_dim)        

    def forward(self,wav_and_flag):
        
        x = wav_and_flag # (B, chan, samples)
        # layer_results: (T,B,ch,C)
        rep, layer_results = self.model.extract_features(x[:,:,:16000*20], output_layer=12)
        layer_reps = [self.rep_fusion_fn(x, idx) for idx, (x, _) in enumerate(layer_results)]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1) # (B, C, T, n_layers)
        
        x = GradMultiply.apply(x, self.feature_grad_mult)
        
        spk_embedding = self.back_end(x)
        
        return spk_embedding


    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            

            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

    def _parse_fusion_module_string(self, string):
        module_insts = list()
        for fm in string.split(','):
            name_params = fm.split(':')
            module_name = name_params[0].strip()

            args = list() if module_name != 'MicWeightedAvg' else [self.n_chans,]
            kwargs = dict()
            module_cls = eval(module_name)
            if len(name_params) != 1:
                # parameters provided
                params_string = ','.join(name_params[1:])
                for ps in params_string.split(';'):
                    k,v = ps.split('=')
                    m = re.match(r'\(([^\)]+)\)(.+)', v)
                    if not m:
                        raise ValueError('Regex fail on "{}", try checking the fusion string modules format: "{}"'.format(v, string))
                    dtype = m.group(1)
                    v = m.group(2)
                    v = eval(dtype)(v)
                    kwargs[k] = v
            module_insts.append( module_cls(*args, **kwargs) )
        return module_insts

    def _extract_layer_rep_first(self, rep, idx):
        # input: (T,B,ch,C) or (T,B,C)
        # output: (B,T,C)
        if rep.dim() == 4:
            # take just the first channel
            rep = rep[:,:,0,:]
            rep = GradMultiply.apply(rep, 1./self.feature_grad_mult)
        elif rep.dim() == 3 and idx == len(self.rep_fusion_inst)-1:
            rep = GradMultiply.apply(rep, 1./self.feature_grad_mult)
        elif rep.dim() != 3:
            raise ValueError('Unexpected dimensionality of a layer representation: {}'.format(rep.dim()))
        return rep.transpose(0, 1) # (B,T,C)

    def _extract_layer_rep_wavg(self, rep, idx):
        # input: (T,B,ch,C) or (T,B,C)
        # output: (B,T,C)
        if rep.dim() == 4:
            # take just the first channel
            rep = self.rep_fusion_inst[idx](rep)
            rep = GradMultiply.apply(rep, 1./self.feature_grad_mult)
        elif rep.dim() == 3 and idx == len(self.rep_fusion_inst)-1:
            rep = GradMultiply.apply(rep, 1./self.feature_grad_mult)
        elif rep.dim() != 3:
            raise ValueError('Unexpected dimensionality of a layer representation: {}'.format(rep.dim()))
        return rep.transpose(0, 1) # (B,T,C)


if __name__ == "__main__":
    from thop import profile
    # from ptflops import get_model_complexity_info
    model_path = '/home/jpeng/ntt/work/Data/pretrained_model/WavLM-Large.pt'
    pooling = 'MHFA'
    embed_dim = 256
    head_nb = 64
    group = 1
    model = WavLM_Base_MHFA(model_path, pooling, head_nb, embed_dim, group,cnn_scale=0.0,layer_drop=0.00)
    flops, params = profile(model.eval(), inputs=(torch.randn(1, 16000*2),))

    print("FLOPS: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
