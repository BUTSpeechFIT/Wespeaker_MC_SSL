#!/bin/bash

. ./path.sh || exit 1

stage=1
stop_stage=5

data=data
data_type="raw"  # shard/raw
data_dir= # input
voices_dir= # input
merged_dir= # output

config=conf/wavlm_base_ep15_voxinit_tac.yaml
exp_dir=exp/wavlm_base_ep15_voxinit_tac
gpus="[0,1]"
num_avg=4
checkpoint=

trials="MutliSV_dev_MRE.txt MutliSV_eval_v1_MRE.txt MutliSV_dev_MRE_hard.txt MutliSV_eval_v1_MRE_hard.txt"

base_port=1024
max_port=40000
current_time=$(date +%s)
port=$((current_time % (max_port - base_port) + base_port))

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  if [[ -z $data_dir ]]; then
    echo "Directory with multichannel data is required at this point, please make sure to get the data first."
    exit 1
  fi
  if [[ -z $voices_dir ]]; then
    echo "Directory with the VOiCES data not provided. If you did not download the data before, "\
         "plese follow instructions at https://iqtlabs.github.io/voices/downloads/"
    exit 1
  fi
  ./local/prepare_multisv_data.sh --stage 1 --stop_stage 4 \
    --data ${data} --data_dir ${data_dir} --voices_dir ${voices_dir} --merged_dir ${merged_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Convert data to ${data_type}..."
  for dset in multisv; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 32 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train_V2_pgroups.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/multisv/${data_type}.list \
      --train_label ${data}/multisv/utt2spk \
      --PORT ${port} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model
  echo "Extract embeddings ..."
  local/extract_msv.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 2 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score_msv.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
