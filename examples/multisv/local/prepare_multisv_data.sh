#!/bin/bash

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

stage=-1
stop_stage=-1
data=data
data_dir=
voices_dir=
merged_dir=

. tools/parse_options.sh || exit 1

data=`realpath ${data}`
download_dir=${data}/download_data
mkdir -p $download_dir

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # create wav.scp and other lists for the training data
    echo "Download training data description and create training lists"
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/training/metadata/MultiSV2_train.zip -P $download_dir
    unzip $download_dir/MultiSV2_train.zip -d $data
    python local/descr2lists.py --descr=${data}/MultiSV2_train.csv --out_dir=${data}/multisv --data_dir=$data_dir

    echo "Training lists preparation succeeded !!!"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # lists for embedding extraction
    echo "Download evaluation data lists and merge evaluation data."
    list_dir=${download_dir}/lists
    mkdir -p ${list_dir}/MRE
    mkdir -p ${list_dir}/MRE_hard
    # MRE
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_dev.enroll.chmapMRE.scp \
        -O ${list_dir}/MRE/MutliSV_dev_MRE.enroll.chmap.scp
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_dev.test.chmap.scp \
        -O ${list_dir}/MRE/MutliSV_dev_MRE.test.chmap.scp
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_eval.enroll.chmapMRE.scp \
        -O ${list_dir}/MRE/MutliSV_eval_v1_MRE.enroll.chmap.scp
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_eval.test.chmapv1.scp \
        -O ${list_dir}/MRE/MutliSV_eval_v1_MRE.test.chmap.scp
    
    # MRE_hard
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_dev.enroll.chmapMRE_hard.scp \
        -O ${list_dir}/MRE_hard/MutliSV_dev_MRE_hard.enroll.chmap.scp
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_dev.test.chmap.scp \
        -O ${list_dir}/MRE_hard/MutliSV_dev_MRE_hard.test.chmap.scp
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_eval.enroll.chmapMRE_hard.scp \
        -O ${list_dir}/MRE_hard/MutliSV_eval_v1_MRE_hard.enroll.chmap.scp
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_eval.test.chmapv1.scp \
        -O ${list_dir}/MRE_hard/MutliSV_eval_v1_MRE_hard.test.chmap.scp
    
    echo "Merging single-channel data. It may tak a few tens of minutes."
    python local/merge_data.py --list_dir=${list_dir} \
        --data="$data" \
        --voices_dir="$voices_dir" \
        --out_dir="$merged_dir"

    echo "Evaluation data preparation succeeded !!!"
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Download trials definition and prepare trial lists"
    down_trial_dir=${download_dir}/trials
    trial_dir=${data}/trials
    mkdir -p $down_trial_dir
    mkdir -p $trial_dir
    # trials description
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_dev.txt.gz -P $down_trial_dir
    test -e ${down_trial_dir}/MultiSV_dev.txt || gunzip ${down_trial_dir}/MultiSV_dev.txt.gz
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_eval.txt.gz -P $down_trial_dir
    test -e ${down_trial_dir}/MultiSV_eval.txt || gunzip ${down_trial_dir}/MultiSV_eval.txt.gz
    
    # enrollment, test lists
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_dev.enroll.scp -P $down_trial_dir
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_dev.test.scp -P $down_trial_dir
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_eval.enroll.scp -P $down_trial_dir
    wget https://github.com/BUTSpeechFIT/MultiSV/raw/main/evaluation/VOiCES_multichan/core/MultiSV_eval.test.scp -P $down_trial_dir

    # create trial lists for WeSpeaker
    python local/create_trial_lists.py --orig_trials_path=${down_trial_dir}/MultiSV_dev.txt \
        --orig_enroll_path=${down_trial_dir}/MultiSV_dev.enroll.scp \
        --orig_test_path=${down_trial_dir}/MultiSV_dev.test.scp \
        --new_trials_path=${trial_dir}/MutliSV_dev_MRE.txt
    python local/create_trial_lists.py --orig_trials_path=${down_trial_dir}/MultiSV_eval.txt \
        --orig_enroll_path=${down_trial_dir}/MultiSV_eval.enroll.scp \
        --orig_test_path=${down_trial_dir}/MultiSV_eval.test.scp \
        --new_trials_path=${trial_dir}/MutliSV_eval_v1_MRE.txt
    
    python local/create_trial_lists.py --orig_trials_path=${down_trial_dir}/MultiSV_dev.txt \
        --orig_enroll_path=${down_trial_dir}/MultiSV_dev.enroll.scp \
        --orig_test_path=${down_trial_dir}/MultiSV_dev.test.scp \
        --new_trials_path=${trial_dir}/MutliSV_dev_MRE_hard.txt
    python local/create_trial_lists.py --orig_trials_path=${down_trial_dir}/MultiSV_eval.txt \
        --orig_enroll_path=${down_trial_dir}/MultiSV_eval.enroll.scp \
        --orig_test_path=${down_trial_dir}/MultiSV_eval.test.scp \
        --new_trials_path=${trial_dir}/MutliSV_eval_v1_MRE_hard.txt

    echo "Trial lists preparation succeeded !!!"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Download pre-trained single-channel model"
    wget https://nextcloud.fit.vutbr.cz/s/zjFJdk3fnbKkrGz/download/spkid2label.txt -P $data

    mkdir -p ${data}/sc_vox_wavlm_mhfa
    wget https://nextcloud.fit.vutbr.cz/s/F9neAa8rDYFR6rD/download/sc_vox_trained.pt -P ${data}/sc_vox_wavlm_mhfa
fi
