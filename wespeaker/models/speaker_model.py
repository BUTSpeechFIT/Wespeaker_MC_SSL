# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wespeaker.models.tdnn as tdnn
import wespeaker.models.ecapa_tdnn as ecapa_tdnn
import wespeaker.models.resnet as resnet
import wespeaker.models.repvgg as repvgg
import wespeaker.models.campplus as campplus
import wespeaker.models.Transformer_WavLM as Transformer_WavLM
import wespeaker.models.Transformer_WavLM2 as Transformer_WavLM2
import wespeaker.models.Transformer_WavLM_Large as Transformer_WavLM_Large

import wespeaker.models.Transformer_WavLM_DINO as Conformer

def get_speaker_model(model_name: str):
    if model_name.startswith("XVEC"):
        return getattr(tdnn, model_name)
    elif model_name.startswith("ECAPA_TDNN"):
        return getattr(ecapa_tdnn, model_name)
    elif model_name.startswith("ResNet"):
        return getattr(resnet, model_name)
    elif model_name.startswith("REPVGG"):
        return getattr(repvgg, model_name)
    elif model_name.startswith("CAMPPlus"):
        return getattr(campplus, model_name)
    elif model_name.startswith("WavLM_Base_MHFA"):
        return getattr(Transformer_WavLM, model_name)
    elif model_name.startswith("WavLM_Base_LR_MHFA"):
        return getattr(Transformer_WavLM2, model_name)
    elif model_name.startswith("WavLM_Large_MHFA"):
        return getattr(Transformer_WavLM_Large, model_name)
    elif model_name.startswith("Conformer"):
        return getattr(Conformer, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
