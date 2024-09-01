import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--orig_trials_path', required=True)
parser.add_argument('--orig_enroll_path', required=True)
parser.add_argument('--orig_test_path', required=True)
parser.add_argument('--new_trials_path', required=True)

args = parser.parse_args()


with open(args.orig_enroll_path, 'r') as fid:
    enr_key2logic = {line.split('=')[0]: line.split('=')[1].strip() for line in fid}
with open(args.orig_test_path, 'r') as fid:
    tst_key2logic = {line.split('=')[0]: line.split('=')[1].strip() for line in fid}

os.makedirs(os.path.dirname(args.new_trials_path), exist_ok=True)
print('Creating trial list: {}'.format(args.new_trials_path))

with open(args.orig_trials_path, 'r') as in_fid, open(args.new_trials_path, 'w') as out_fid:
    for line in in_fid:
        enr_key, tst_key, result = line.split()

        enr_logic = enr_key2logic[enr_key]
        tst_logic = tst_key2logic[tst_key]

        if result == 'imp':
            result = 'nontarget'
        elif result == 'tgt':
            result = 'target'
        else:
            raise ValueError('Unknown type of a trial: "{}". Expected options: imp, tgt'.format(result))

        out_fid.write('{} {} {}\n'.format(enr_logic, tst_logic, result))
