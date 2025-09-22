from cadlib.macro import *

TOLERANCE = 3

def get_accs_single(pred_vec, gt_vec):
    args_mask = CMD_ARGS_MASK.astype(float)

    out_cmd = pred_vec[:, 0]
    gt_cmd = gt_vec[:, 0]

    out_param = pred_vec[:, 1:]
    gt_param = gt_vec[:, 1:]

    try:
        # The original deepcad does this. Question: is this actually desired?
        seq_len = gt_cmd.tolist().index(EOS_IDX)
        gt_cmd = gt_cmd[:seq_len]
        out_cmd = out_cmd[:seq_len]
    except:
        pass


    cmd_acc = (out_cmd == gt_cmd).astype(int)
    param_acc = []
    for j in range(len(gt_cmd)):
        cmd = gt_cmd[j]

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

        if param_acc == []:
            param_acc = [0]

    cmd_acc = np.mean(cmd_acc)
    param_acc = np.mean(param_acc)
    return cmd_acc, param_acc

def get_cmd_arg_accs(pred_vec_batch, gt_vec_batch):
    # input vec as B, 17

    B = len(pred_vec_batch)
    avg_param_acc = []
    avg_cmd_acc = []

    for i in range(B):
        pred_vec = pred_vec_batch[i]
        gt_vec = gt_vec_batch[i]

        cmd_acc, param_acc = get_accs_single(pred_vec=pred_vec, gt_vec=gt_vec)
        
        avg_param_acc.append(param_acc)
        avg_cmd_acc.append(cmd_acc)

    return avg_cmd_acc, avg_param_acc