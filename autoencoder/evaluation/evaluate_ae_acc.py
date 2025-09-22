from collections import defaultdict
import h5py
from tqdm import tqdm
import os
import argparse
import sys
import pickle
sys.path.append("..")
from cadlib.macro import *

# Then run show_ae_acc_by_sl for plot

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default=None, required=True)
args = parser.parse_args()
args.accuracy_by_sequence_length = True
TOLERANCE = 3

result_dir = args.src
filenames = sorted(os.listdir(result_dir))

# overall accuracy
avg_cmd_acc = [] # ACC_cmd
avg_param_acc = [] # ACC_param

# accuracy w.r.t. each command type
each_cmd_cnt = np.zeros((len(ALL_COMMANDS),))
each_cmd_acc = np.zeros((len(ALL_COMMANDS),))

# accuracy w.r.t each parameter
args_mask = CMD_ARGS_MASK.astype(float)
N_ARGS = args_mask.shape[1]
each_param_cnt = np.zeros([*args_mask.shape])
each_param_acc = np.zeros([*args_mask.shape])

# accuracy w.r.t. sequence length
if args.accuracy_by_sequence_length:
    avg_cmd_acc_sl = defaultdict(list)
    avg_param_acc_sl = defaultdict(list)


for name in tqdm(filenames):
    path = os.path.join(result_dir, name)
    with h5py.File(path, "r") as fp:
        out_vec = fp["out_vec"][:].astype(int)
        gt_vec = fp["gt_vec"][:].astype(int)

    out_cmd = out_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

    out_param = out_vec[:, 1:]
    gt_param = gt_vec[:, 1:]

    cmd_acc = (out_cmd == gt_cmd).astype(int)
    param_acc = []
    for j in range(len(gt_cmd)):
        cmd = gt_cmd[j]
        each_cmd_cnt[cmd] += 1
        each_cmd_acc[cmd] += cmd_acc[j]
        if cmd in [SOL_IDX, EOS_IDX]:
            continue

        if out_cmd[j] == gt_cmd[j]: # NOTE: only account param acc for correct cmd
            tole_acc = (np.abs(out_param[j] - gt_param[j]) < TOLERANCE).astype(int)
            # filter param that do not need tolerance (i.e. requires strictly equal)
            if cmd == EXT_IDX:
                tole_acc[-2:] = (out_param[j] == gt_param[j]).astype(int)[-2:]
            elif cmd == ARC_IDX:
                tole_acc[3] = (out_param[j] == gt_param[j]).astype(int)[3]

            valid_param_acc = tole_acc[args_mask[cmd].astype(bool)].tolist()
            param_acc.extend(valid_param_acc)

            each_param_cnt[cmd, np.arange(N_ARGS)] += 1
            each_param_acc[cmd, np.arange(N_ARGS)] += tole_acc
        if param_acc == []:
            param_acc = [0]

    # cmd_acc, param_acc = get_accs_single(pred_vec=out_vec, gt_vec=gt_vec)

    param_acc = np.mean(param_acc)
    avg_param_acc.append(param_acc)
    cmd_acc = np.mean(cmd_acc)
    avg_cmd_acc.append(cmd_acc)
    
    if args.accuracy_by_sequence_length:
        seq_len = out_cmd.shape[0]
        avg_cmd_acc_sl[seq_len].append(cmd_acc)
        avg_param_acc_sl[seq_len].append(param_acc)


save_path = result_dir + "_acc_stat.txt"
fp = open(save_path, "w")
# overall accuracy (averaged over all data)
avg_cmd_acc = np.mean(avg_cmd_acc)
print("avg command acc (ACC_cmd):", avg_cmd_acc, file=fp)
avg_param_acc = np.mean(avg_param_acc)
print("avg param acc (ACC_param):", avg_param_acc, file=fp)

# acc of each command type
each_cmd_acc = each_cmd_acc / (each_cmd_cnt + 1e-6)
print("each command count:", each_cmd_cnt, file=fp)
print("each command acc:", each_cmd_acc, file=fp)

# acc of each parameter type
each_param_acc = each_param_acc * args_mask
each_param_cnt = each_param_cnt * args_mask
each_param_acc = each_param_acc / (each_param_cnt + 1e-6)
for i in range(each_param_acc.shape[0]):
    log_str = f'{ALL_COMMANDS[i]+" param acc:"} {each_param_acc[i][args_mask[i].astype(bool)]}'.replace('\n', '')
    # print(log_str)
    fp.write(log_str + '\n')
    # print(ALL_COMMANDS[i] + " param acc:", each_param_acc[i][args_mask[i].astype(bool)], file=fp)
fp.close()

with open(save_path, "r") as fp:
    res = fp.readlines()
    for l in res:
        print(l, end='')

if args.accuracy_by_sequence_length:
    mean = lambda x: x
    avg_cmd_acc_sl = {k: mean(v) for k, v in avg_cmd_acc_sl.items()}
    avg_param_acc_sl = {k: mean(v) for k, v in avg_param_acc_sl.items()}


    dic_save_path = os.path.splitext(save_path)[0] + '_by_seq_len.pkl'
    print(dic_save_path)
    dic_to_save = {
        'avg_cmd_acc': avg_cmd_acc_sl,
        'avg_param_acc': avg_param_acc_sl,
    }
    with open(dic_save_path, 'wb') as f:
        pickle.dump(dic_to_save, f)

    # Print evaluations by SL

    command_accs_avg = []
    param_accs_avg = []
    missing_keys = []
    for key in range(3, 60):
        if key in avg_cmd_acc_sl:
            command_accs_avg.append(np.mean(avg_cmd_acc_sl[key]))
            param_accs_avg.append(np.mean(avg_param_acc_sl[key]))
        else:
            missing_keys.append(key)
    print("missing keys", missing_keys)
    command_accs_avg = np.array(command_accs_avg)
    param_accs_avg = np.array(param_accs_avg)
    avg_cmd = np.mean(command_accs_avg)
    avg_param = np.mean(param_accs_avg)

    print("="*30)
    print("="*30)
    print("command _SL", np.round(avg_cmd * 100, 2), "param_SL", np.round(avg_param * 100, 2))
print("avg command acc (ACC_cmd):", np.round(avg_cmd_acc * 100, 2))
print("avg param acc (ACC_param):", np.round(avg_param_acc * 100, 2))
print("="*30)
print("="*30)

    # avg_cmd_acc_sl_gt = {k: mean(v) for k, v in avg_cmd_acc_sl_gt.items()}
    # avg_param_acc_sl_gt = {k: mean(v) for k, v in avg_param_acc_sl_gt.items()}
    # dic_save_path = os.path.splitext(save_path)[0] + '_by_seq_len_gt.pkl'
    # print(dic_save_path)
    # dic_to_save_gt = {
    #     'avg_cmd_acc': avg_cmd_acc_sl_gt,
    #     'avg_param_acc': avg_param_acc_sl_gt,
    # }
    # with open(dic_save_path, 'wb') as f:
    #     pickle.dump(dic_to_save_gt, f)