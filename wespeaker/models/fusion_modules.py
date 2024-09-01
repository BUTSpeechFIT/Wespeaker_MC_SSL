import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

class Identity(nn.Module):
    def __init__(self):
        self.miso = False
        super().__init__()

    def forward(self, x):
        return x


class MicWeightedAvg(nn.Module):
    def __init__(self, n_chans, ref_mic_w=0.1, trainable=True):
        super().__init__()
        x = -1.
        y = np.log( (ref_mic_w/(1-ref_mic_w+1e-8)) * (n_chans-1) * np.exp(x) )
        init_w = torch.Tensor([y,] + [x]*(n_chans-1))
        
        self.mic_weights = nn.Parameter(data=init_w, requires_grad=bool(trainable))
        self.miso = True
    
    def forward(self, x):
        # input:  (T,B,ch,C)
        # output: (T,B,C)
        w = nn.functional.softmax(self.mic_weights, dim=0)
        y = torch.mul(x, w[None,None,:,None]) # (T,B,ch,C)
        y = torch.sum(y, dim=-2)
        return y


class AttAcrossChansWoFF(nn.Module):
    def __init__(self, embed_dim=768, n_heads=8, attn_dropout=0., init_mult=1e-2):
        super().__init__()
        self.init_mult = init_mult

        # default input: (seq, batch, feature)
        self.att_layer_norm = nn.LayerNorm(embed_dim)
        self.mh_att = nn.MultiheadAttention(embed_dim, n_heads, bias=True, kdim=None, vdim=None, dropout=attn_dropout)

        self.miso = False

    def scale_weights(self):
        self.mh_att.out_proj.bias.data *= 0.
        self.mh_att.out_proj.weight.data *= self.init_mult

    def forward(self, x):
        # x: (T,B,ch,C)
        frames, B, chans, feat_dim = x.shape
        x = torch.permute(x, (2,1,0,3)) # (ch, B, T, C)
        x = torch.reshape(x, (chans, -1, feat_dim)) # (ch, B*T, C)

        residual = x

        x = self.att_layer_norm(x)
        x, _ = self.mh_att(x, x, x, need_weights=False)
        # we don't apply dropout here
        x = x + residual

        x = torch.reshape(x, (chans,B,-1,feat_dim)) # (ch,B,T,C)
        x = torch.permute(x, (2,1,0,3)) # (T,B,ch,C)

        return x


class AttAcrossChans(nn.Module):
    def __init__(self, embed_dim=768, ffn_embedding_dim=3072, n_heads=8, act_fn='PReLU', attn_dropout=0., init_mult=1e-2, pre_norm=True):
        super().__init__()
        self.init_mult = init_mult

        # default input: (seq, batch, feature)
        self.att_layer_norm = nn.LayerNorm(embed_dim)
        self.mh_att = nn.MultiheadAttention(embed_dim, n_heads, bias=True, kdim=None, vdim=None, dropout=attn_dropout)
        # with torch.no_grad():
        #     self.mh_att.out_proj.bias.data *= 0.
        #     #self.mh_att.out_proj.weight.data *= init_mult
        #     self.mh_att.out_proj.weight.copy_(self.mh_att.out_proj.weight.data*0.)
        #     print(id(self.mh_att.out_proj.weight), id(self.mh_att.out_proj.weight.data))

        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.act_fn = self._get_act_fn_by_name(act_fn)
        self.fc1 = nn.Linear(embed_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embed_dim)
        # self.fc2.bias.data *= 0.
        # print('setting fc2')
        # self.fc2.weight.data *= init_mult
        self.pre_norm = bool(pre_norm)
        self.miso = False

    def _get_act_fn_by_name(self, name):
        return eval('torch.nn.'+name)()

    def scale_weights(self):
        self.mh_att.out_proj.bias.data *= 0.
        self.mh_att.out_proj.weight.data *= self.init_mult

        self.fc2.bias.data *= 0.
        self.fc2.weight.data *= self.init_mult
        if not self.pre_norm:
            self.att_layer_norm.weight.data *= self.init_mult
            self.final_layer_norm.weight.data *= self.init_mult

    def forward(self, x):
        # x: (T,B,ch,C)
        frames, B, chans, feat_dim = x.shape
        x = torch.permute(x, (2,1,0,3)) # (ch, B, T, C)
        x = torch.reshape(x, (chans, -1, feat_dim)) # (ch, B*T, C)

        if self.pre_norm:
            residual = x

            x = self.att_layer_norm(x)
            x, _ = self.mh_att(x, x, x, need_weights=False)
            # we don't apply dropout here
            x = x + residual

            residual = x

            x = self.final_layer_norm(x)
            x = self.act_fn(self.fc1(x)) # (ch, B*T, ffn_embedding_dim)
            # no dropout here
            x = self.fc2(x) # (ch, B*T, C)
            # no dropout here
            x = x + residual
        else:
            residual = x

            x, _ = self.mh_att(x, x, x, need_weights=False)
            x = x + residual
            x = self.att_layer_norm(x)

            residual = x
            x = self.act_fn(self.fc1(x)) # (ch, B*T, ffn_embedding_dim)
            x = self.fc2(x) # (ch, B*T, C)
            x = x + residual

            x = self.final_layer_norm(x)

        x = torch.reshape(x, (chans,B,-1,feat_dim)) # (ch,B,T,C)
        x = torch.permute(x, (2,1,0,3)) # (T,B,ch,C)

        return x


class MultiHeadCoAttention(nn.Module):
    def __init__(self, multi_dim, single_dim, num_heads):
        assert multi_dim % num_heads == 0, 'multi_dim must be divisible by num_heads'
        assert single_dim % num_heads == 0, 'single_dim must be divisible by num_heads'
        super().__init__()
        self.q_proj = nn.Linear(multi_dim, multi_dim)
        self.k_proj = nn.Linear(multi_dim, multi_dim)
        self.multi_v_proj = nn.Linear(multi_dim, multi_dim) # D'
        self.single_v_proj = nn.Linear(single_dim, single_dim) # D

        self.multi_out_proj  = nn.Linear(multi_dim, multi_dim) # D'
        self.single_out_proj = nn.Linear(single_dim, single_dim) # D

        self.multi_dim = multi_dim
        self.single_dim = single_dim
        self.num_heads = num_heads

    def forward(self, query, key, multi_value, single_value):
        # q, k, multi_v: (T,B,ch,D')
        # single_v: (T,B,1,D)
        query = torch.transpose(query, 0, 1) # (B,T,ch,D')...[32, 150, 4, 64]
        key = torch.transpose(key, 0, 1) # (B,T,ch,D')...[32, 150, 4, 64]
        multi_value = torch.permute(multi_value, (1,2,0,3)) # (B,ch,T,D')...[32, 4, 150, 64]
        single_value = torch.permute(single_value, (1,2,0,3)) # (B,1,T,D)...[32, 1, 150, 256]
        ###########

        q = torch.split( self.q_proj(query), self.multi_dim//self.num_heads, dim=-1 ) # seq: (B,T,ch,D'/h)
        q = torch.stack(q, dim=1) # (B,h,T,ch,D'/h)...[32, 8, 150, 4, 8]

        k = torch.split( self.k_proj(key),   self.multi_dim//self.num_heads, dim=-1 ) # seq: (B,T,ch,D'/h)
        k = torch.stack(k, dim=1) # (B,h,T,ch,D'/h)...[32, 8, 150, 4, 8]

        multi_v  = torch.split( self.multi_v_proj(multi_value), self.multi_dim//self.num_heads, dim=-1 ) # seq: (B,ch,T,D'/h)
        multi_v  = torch.stack(multi_v, dim=1) # (B, h, ch, T, D'/h)...[32, 8, 4, 150, 8]

        single_v = torch.split( self.single_v_proj(single_value), self.single_dim//self.num_heads, dim=-1 ) # seq: (B,1,T,D/h)
        single_v = torch.stack(single_v, dim=1) # seq: (B,h,1,T,D/h)...[32, 32, 1, 150, 8]

        q = q.view(*q.shape[:-2], -1) # (B, h, T, ch*D/h)
        k = k.view(*k.shape[:-2], -1) # (B, h, T, ch*D/h)
        normalizer = torch.sqrt( torch.Tensor([float(q.shape[-1])]).to(q.device) )

        sim_mat = torch.matmul(q, torch.transpose(k,-2,-1)) / normalizer # (B, h, T, T)
        att_mat = torch.unsqueeze( F.softmax(sim_mat, dim=-1), 2 ) # (B, h, 1, T, T)

        # co-attention
        multi_result  = torch.matmul(att_mat, multi_v) # (B, h, ch, T, D'/h)
        single_result = torch.matmul(att_mat, single_v) # (B, h, 1, T, D/h)

        multi_result  = torch.permute(multi_result, (3,0,2,1,4)) # (T, B, ch, h, D'/h)
        single_result = torch.permute(single_result, (3,0,2,1,4)) # (T, B, 1, h, D/h)
        multi_result  = torch.reshape(multi_result,  multi_result.shape[:-2]+(-1,)) # (T, B, ch, D')
        single_result = torch.reshape(single_result, single_result.shape[:-2]+(-1,)) # (T, B, 1, D)

        multi_result = self.multi_out_proj(multi_result)
        single_result = self.single_out_proj(single_result)
        return multi_result, single_result


class CoAttention(nn.Module):
    def __init__(self, embed_dim=768, single_dim=256, multi_dim=64, n_heads=8, attn_dropout=0., init_mult=1e-2): #, pre_norm=True):
        super().__init__()
        self.init_mult = init_mult

        self.in_single_proj = nn.Linear(embed_dim, single_dim) # single_dim == D
        self.in_single_ln   = nn.LayerNorm(single_dim)

        self.in_multi_proj = nn.Linear(embed_dim, multi_dim) # multi_dim == D'
        self.in_multi_ln   = nn.LayerNorm(multi_dim)

        self.mca = MultiHeadCoAttention(multi_dim, single_dim, n_heads)
        self.mca_multi_out_ln = nn.LayerNorm(multi_dim)
        self.mca_single_out_ln = nn.LayerNorm(single_dim)

        # default MHA input: (seq, batch, feature)
        self.cross_frame_mha = nn.MultiheadAttention(single_dim, n_heads, dropout=attn_dropout, bias=True, kdim=None, vdim=None)
        self.mha_ln = nn.LayerNorm(single_dim)

        self.cat_proj = nn.Linear(single_dim+multi_dim, embed_dim)

        self.miso = False

    def scale_weights(self):
        self.cat_proj.bias.data *= 0.
        self.cat_proj.weight.data *= self.init_mult

    def forward(self, x):
        # x: (T,B,ch,F); (150, 32, 4, 768)
        frames, B, chans, feat_dim = x.shape

        single_x = torch.mean(x, dim=-2, keepdim=True) # (T,B,1,F)
        single_x = self.in_single_ln(self.in_single_proj(single_x)) # (T,B,1,D)

        multi_x = self.in_multi_ln(self.in_multi_proj(x)) # (T,B,ch,D')
        
        # MCA
        multi_mca, single_mca = self.mca(multi_x, multi_x, multi_x, single_x) # (T,B,ch,D'), (T,B,ch,D)
        single_x = single_x + single_mca
        multi_x  = multi_x  + multi_mca
        multi_x = self.mca_multi_out_ln(multi_x) # (T,B,ch,D')
        single_x = torch.squeeze( self.mca_single_out_ln(single_x), -2 ) # (T,B,D)

        # MHA
        single_mha, _ = self.cross_frame_mha(single_x, single_x, single_x, need_weights=False) # (T, B, D)
        single_x = self.mha_ln(single_mha + single_x)

        # join representations
        single_x = single_x.unsqueeze(-2) # (T,B,1,D)
        single_x_tile = torch.tile(single_x, (1,1,chans,1)) # (T,B,ch,D)
        cat_x = torch.cat([single_x_tile, multi_x], dim=-1) # (T,B,ch,D+D')
        out = self.cat_proj(cat_x) # (T,B,ch,F)

        return out + x


class TACFusion(nn.Module):
    """Transform-Average-Concatenate inter-microphone-channel permutation invariant communication block [1].

    Args:
        input_dim (int): Number of features of input representation.
        hidden_dim (int, optional): size of hidden layers in TAC operations.
        activation (str, optional): type of activation used. See asteroid.masknn.activations.
        norm_type (str, optional): type of normalization layer used. See asteroid.masknn.norms.

    .. note:: Supports inputs of shape :math:`(batch, mic\_channels, features, chunk\_size, n\_chunks)`
        as in FasNet-TAC. The operations are applied for each element in ``chunk_size`` and ``n_chunks``.
        Output is of same shape as input.

    References
        [1] : Luo, Yi, et al. "End-to-end microphone permutation and number invariant multi-channel
        speech separation." ICASSP 2020.
    """

    def __init__(self, input_dim=768, hidden_dim=960, activation=nn.PReLU, norm_type=nn.LayerNorm, gammma_init_mult=1e-2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_tf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activation()
        )
        self.avg_tf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), activation()
        )
        last_proj = nn.Linear(2 * hidden_dim, input_dim)
        self.concat_tf = nn.Sequential(
            last_proj, activation() #(init=1.)
        )
        self.norm = norm_type(input_dim) # bias initialized to zeros
        self.norm.weight.data *= gammma_init_mult
        self.miso = False

    def forward(self, x):
        """
        Args:
            x: (:class:`torch.Tensor`): Input multi-channel features.
                Shape: :math:`(frames, batch, mic\_channels, features)`.

        Returns:
            output (:class:`torch.Tensor`): features for each mic_channel after TAC inter-channel processing.
                Shape :math:`(frames, batch, mic\_channels, features)`.
        """
        # x: (T,B,ch,C)

        # First operation: transform the input for each frame and independently on each mic channel.
        output = self.input_tf(x) # (T,B,ch,hidden_dim)

        # Mean pooling across channels
        mics_mean = output.mean(2) # (T,B,hidden_dim)

        # The average is processed by a non-linear transform
        mics_mean = self.avg_tf(mics_mean) # (T,B,hidden_dim)
        mics_mean = (
            mics_mean.unsqueeze(2).expand_as(output)
        ) # (T,B,ch,hidden_dim)

        # Concatenate the transformed average in each channel with the original feats and
        # project back to same number of features
        output = torch.cat([output, mics_mean], -1) # (T,B,ch,2*hidden_dim)
        output = self.concat_tf(output) # (T,B,ch,C)
        output = self.norm(output) # (T,B,ch,C)

        output += x
        return output
