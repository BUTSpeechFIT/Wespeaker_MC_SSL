import argparse
import os
import csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--descr', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--data_dir', required=True)

args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
wav_scp_path = os.path.join(args.out_dir, 'wav.scp')
utt2spk_path = os.path.join(args.out_dir, 'utt2spk')
spk2utt_path = os.path.join(args.out_dir, 'spk2utt')

if not os.path.exists(args.descr):
    raise ValueError(f'Data description file does not exist: {args.descr}')

with open(args.descr, 'r', newline='') as csvfile, \
     open(wav_scp_path, 'w') as wav_scp_fid, \
     open(utt2spk_path, 'w') as utt2spk_fid:

    descr_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    header = next(descr_reader)
    path_idx = header.index('speech')
    spk_idx = header.index('spk_id')

    spk2utt_dict = defaultdict(list)
    for row in descr_reader:
        sub_path = row[path_idx] + '.wav'
        spk = row[spk_idx].replace('VOXCELEB2', '')
        utt_name = sub_path.replace('dev/aac', '').lstrip('/')
        path = os.path.join(args.data_dir, sub_path)

        wav_scp_fid.write('{} {}\n'.format(utt_name, path))
        utt2spk_fid.write('{} {}\n'.format(utt_name, spk))

        spk2utt_dict[spk].append(utt_name)

with open(spk2utt_path, 'w') as spk2utt_fid:
    for spk in spk2utt_dict:
        spk2utt_fid.write('{} {}\n'.format(spk, ' '.join(spk2utt_dict[spk])))
