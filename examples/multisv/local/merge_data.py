import os
import json
import argparse
import zipfile
import tempfile
import numpy as np
import scipy.io.wavfile as wavio

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--list_dir', required=True)
parser.add_argument('--voices_dir', required=True)
parser.add_argument('--out_dir', required=True)

args = parser.parse_args()

chmaps = {
    'MRE': {
        'MRE/MutliSV_dev': [
            os.path.join(args.list_dir, 'MRE/MutliSV_dev_MRE.enroll.chmap.scp'),
            os.path.join(args.list_dir, 'MRE/MutliSV_dev_MRE.test.chmap.scp'),
        ],
        'MRE/MutliSV_eval_v1': [
            os.path.join(args.list_dir, 'MRE/MutliSV_eval_v1_MRE.enroll.chmap.scp'),
            os.path.join(args.list_dir, 'MRE/MutliSV_eval_v1_MRE.test.chmap.scp'),
        ],
    },
    'MRE_hard': {
        'MRE_hard/MutliSV_dev': [
            os.path.join(args.list_dir, 'MRE_hard/MutliSV_dev_MRE_hard.enroll.chmap.scp'),
            os.path.join(args.list_dir, 'MRE_hard/MutliSV_dev_MRE_hard.test.chmap.scp'),
        ],
        'MRE_hard/MutliSV_eval_v1': [
            os.path.join(args.list_dir, 'MRE_hard/MutliSV_eval_v1_MRE_hard.enroll.chmap.scp'),
            os.path.join(args.list_dir, 'MRE_hard/MutliSV_eval_v1_MRE_hard.test.chmap.scp'),
        ],
    },
}


for proto in chmaps:
    print(f'Protocol: {proto}')

    base_dir = os.path.join(args.out_dir, proto)
    
    for set_key in chmaps[proto]:
        print(f'Set key: {set_key}')
        out_list_dir_cur = os.path.join(args.data, set_key)
        os.makedirs(out_list_dir_cur, exist_ok=True)
        raw_list_path = os.path.join(out_list_dir_cur, 'raw.list')
        wav_scp_path  = os.path.join(out_list_dir_cur, 'wav.scp')
    
        with open(raw_list_path, 'w') as raw_list, open(wav_scp_path, 'w') as wav_scp:
            for chmap in chmaps[proto][set_key]:
                print(f'Channel map: {chmap}')
                with open(chmap, 'r') as in_fid:
                    for line in in_fid:
                        mc_name, sc_names = line.strip().split('=')
                        sc_names = sc_names.split()
                        out_path = os.path.join(base_dir, mc_name + '.wav')
    
                        paths = [os.path.join(args.voices_dir, n + '.wav') for n in sc_names]
                        if not all([os.path.exists(p) for p in paths]):
                            print('WARNING: some paths do not exist: {}'.format(paths))
                            continue
                        
                        fs_audio = [wavio.read(p) for p in paths]
                        fs = fs_audio[0][0] # we assume fs is the same for all recordings
                        audio = np.array([a[1] for a in fs_audio]).T # (samples, chans)
    
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        wavio.write(out_path, fs, audio)
                        
                        spk = mc_name.replace('Lab41-SRI-VOiCES-', '').split('-')[0]
                        descr = {'key': mc_name, 'spk': spk, 'wav': out_path}
                        scp_line = '{} {}'.format(mc_name, out_path)
                        raw_list.write(json.dumps(descr) + '\n')
                        wav_scp.write(scp_line + '\n')

