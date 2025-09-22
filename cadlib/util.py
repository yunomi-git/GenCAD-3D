from IPython.core.pylabtools import figsize
from tqdm import tqdm
import torch
import os
import h5py

import paths
from cadlib.macro import (EOS_VEC, EXT_IDX, EOS_IDX, ARGS_DIM, ALL_COMMANDS, 
                          CMD_ARGS_MASK, LINE_IDX, CIRCLE_IDX, ARC_IDX, SOL_IDX, COMMAND_CMAP, COMMAND_COLORS, MAX_TOTAL_LEN, MAX_N_EXT)
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_cad_vec(data_path, data_id, max_total_len=60, pad=True, as_tensor=True, as_single_vec=False):
    h5_path = os.path.join(data_path, data_id + ".h5")
    with h5py.File(h5_path, "r") as fp:
        cad_vec = fp["vec"][:]  # (len, 1 + N_ARGS)

    if pad:
        pad_len = max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

    if as_single_vec:
        if as_tensor:
            cad_vec = torch.tensor(cad_vec, dtype=torch.long)
        return cad_vec
    else:
        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        if as_tensor:
            command = torch.tensor(command, dtype=torch.long)
            args = torch.tensor(args, dtype=torch.long)
        return command, args   

def load_split_cad_vec(data_path, data_id, as_tensor=True, as_single_vec=False, pad_extrudes=True):
    # loads a data_id originally as S, 17 split into extrudes:
    # output: e, S, 17

    # command sequence is always padded
    # extrude sequence may be padded

    cad_vec = load_cad_vec(data_path=data_path, data_id=data_id, as_tensor=False, pad=False, as_single_vec=True)
    cad_vec_split = split_at_extrudes(cad_vec)
    cad_vec_split = [pad_to_length(cad_subvec)[np.newaxis] for cad_subvec in cad_vec_split]
    cad_vec_split = np.concatenate(cad_vec_split, axis=0)
    # e, S, 17

    # Now do tensors and splits
    if pad_extrudes:
        out = np.repeat(EOS_VEC[np.newaxis], MAX_TOTAL_LEN, axis=0)
        out = np.repeat(out[np.newaxis], MAX_N_EXT, axis=0)
        # print(out.shape)
        out[:len(cad_vec_split)] = cad_vec_split
        cad_vec_split = out

    if as_single_vec:
        if as_tensor:
            cad_vec_split = torch.tensor(cad_vec_split, dtype=torch.long)
        return cad_vec_split
    else:
        command, args = separate_cad_vec(cad_vec_split)
        if as_tensor:
            command = torch.tensor(command, dtype=torch.long)
            args = torch.tensor(args, dtype=torch.long)
        return command, args   
    
def separate_cad_vec(cad_vec):
    command = cad_vec[..., 0]
    args = cad_vec[..., 1:]

    return command, args   

def collate_cad_vec(command, args):
    if isinstance(command, torch.Tensor):
        cad_vec = torch.concat([command.unsqueeze(-1), args], dim=-1)
    else:
        cad_vec = np.concatenate([command[..., np.newaxis], args], axis=-1)
    return cad_vec

def get_extrude_idx(command=None, args=None, cad_vec = None):
    if cad_vec is not None:
        command, _ = separate_cad_vec(cad_vec)
    ext_indices = np.where(command == EXT_IDX)[0]
    return ext_indices

def split_at_extrudes(cad_vec):
    command, args = separate_cad_vec(cad_vec)
    ext_indices = get_extrude_idx(command)
    # cad_vec = collate_cad_vec(command, args)
    ext_vecs = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]
    return ext_vecs

def pad_to_length(cad_vec: np.ndarray, pad_length=MAX_TOTAL_LEN):
    out = np.repeat(EOS_VEC[np.newaxis, ...], 60, axis=0)
    seq_len = cad_vec.shape[0]
    out[:seq_len] = cad_vec
    return out # S, 17

def calc_cad_sequence_length(command=None, args=None, cad_vec=None):
    # command, args = self.load_cad(data_id)
    if cad_vec is not None:
        command, args = separate_cad_vec(cad_vec)
    seq_len = command.tolist().index(EOS_IDX)
    return seq_len

def shape_to_stl(shape, out_file):
    from OCC.Extend.DataExchange import write_stl_file
    write_stl_file(shape, out_file,
                    mode="binary",
                    linear_deflection=0.001,
                    angular_deflection=0.1)
    
def shape_to_step(shape, out_file):
    from OCC.Extend.DataExchange import write_step_file
    write_step_file(shape, out_file)

def apply_command_mask(cad_vec=None, command=None, args=None):
    # numpy
    if command is None:
        command, args = separate_cad_vec(cad_vec)
    
    if isinstance(command, np.ndarray):
        mask = CMD_ARGS_MASK[command.astype(np.int32)]
        args[~mask.astype(bool)] = -1
    else: # tensor
        mask = CMD_ARGS_MASK[command.to(torch.int32).detach().cpu().numpy()]
        args[~mask.astype(bool)] = -1
    
    if cad_vec is None:
        return command, args
    cad_vec = collate_cad_vec(command, args)
    return cad_vec

def visualize_program(cad_vec, out_file=None, ax=None, cbar=True, error_plot=False, legend=False):
    # visualize the padded version for consistency
    # cad_vec should be numpy
    close_ax = False
    if ax is None:
        close_ax = True
        fig, ax = plt.subplots(figsize=(7,10))

    cad_vec = apply_command_mask(cad_vec)
    cell_color_t = cad_vec.astype(np.float32)
    
    # normalize colors
    cell_color_t[:, 1:] = cell_color_t[:, 1:] / (1.0 * ARGS_DIM)
    cell_color_t[:, 1:] = cell_color_t[:, 1:]
    cell_color_t[:, 0] = cell_color_t[:, 0] / (1.0 * len(ALL_COMMANDS))

    if not error_plot:
        mask = (cad_vec == -1)
        cell_color_masked = np.ma.masked_array(cell_color_t, mask=mask)

        # Create heatmap for parameters
        im = ax.imshow(cell_color_masked, cmap='twilight', aspect='auto', vmin=0, vmax=1)
    else:
        mask = (cell_color_t <= 0)
        cell_color_masked = np.ma.masked_array(cell_color_t, mask=mask)
        # Create heatmap for parameters
        im = ax.imshow(cell_color_masked, cmap='twilight', aspect='auto', vmin=0, vmax=1)
    
    #
    if not error_plot:
        masked_data = np.ma.masked_array(cad_vec, mask=True)
        masked_data.mask[:, 0] = False
        im2 = ax.imshow(masked_data, cmap=COMMAND_CMAP, aspect='auto', vmin=0, vmax=len(ALL_COMMANDS) - 1)

    # Set ticks and labels
    ax.set_xticks(range(0, 17, 5))
    ax.set_xticks(np.arange(1, 17 + 1, step=4) - 0.5)
    ax.set_yticks(np.arange(60 + 1, step=10) - 0.7)
    ax.set_xticklabels(range(1, 17 + 1, 4))
    ax.set_yticklabels(range(0, 60 + 1, 10))
    ax.grid(True, linestyle='--', color='black', alpha=0.7)

    
    # Add colorbar
    if cbar:
        plt.colorbar(im, ax=ax)

    if legend:
        legend_elements = [Patch(facecolor=color, label=ALL_COMMANDS[idx]) for idx, color in COMMAND_COLORS.items()]
        ax.legend(handles=legend_elements, loc="lower right")

    if out_file is not None:
        paths.mkdir(out_file)
        plt.savefig(out_file, bbox_inches='tight')

    if close_ax:
        plt.close(fig)
        return
    
    return ax

from matplotlib.patches import Patch

def compare_cad_vecs(cad_vec1, cad_vec2, out_file=None, names=None):
    """
    Compare two CAD vectors for equality.
    :param cad_vec1: First CAD vector.
    :param cad_vec2: Second CAD vector.
    :return: True if they are equal, False otherwise.
    """
    fig, axs = plt.subplots(1, 3)
    if names is not None:
        axs[0].set_title(names[0])
        axs[1].set_title(names[1])
        axs[2].set_title("Error Map")
    visualize_program(cad_vec1, ax=axs[0], cbar=False, error_plot=False)
    visualize_program(cad_vec2, ax=axs[1], cbar=False, error_plot=False)

    error = np.abs(cad_vec1 - cad_vec2)
    error[error[:, 0] > 0, 0] = 1  # Convert to binary error map
    fig.set_figheight(6) # Set height to 8 inches
    fig.set_figwidth(10)  # Set width to 12 inches

    visualize_program(error, ax=axs[2], cbar=True, error_plot=True, legend=True)

    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    plt.close()


    return axs


def visualize_code(codes, ends, max_num_codes, out_file=None, ax=None, cbar=True, error_plot=False):
    # visualize the padded version for consistency
    # codes: E, Q
    # ends: E
    cell_color_t = codes.astype(np.float32)
    cell_color_t = cell_color_t / max_num_codes
    cell_color_t = np.concatenate([ends[:, np.newaxis], cell_color_t], axis=-1)
    E, Q = codes.shape
    
    # normalize colors
    if not error_plot:
        mask = np.zeros_like(cell_color_t).astype(np.bool)
        mask[ends == 1, 1:] = True
        # mask = (cad_vec == -1)
        cell_color_masked = np.ma.masked_array(cell_color_t, mask=mask)

        # Create heatmap for parameters
        im = ax.imshow(cell_color_masked, cmap='rainbow', aspect='auto', vmin=0, vmax=1)
    else:
        mask = (cell_color_t <= 0)
        cell_color_masked = np.ma.masked_array(cell_color_t, mask=mask)
        # Create heatmap for parameters
        im = ax.imshow(cell_color_masked, cmap='rainbow', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(0, Q+1, 5))
    ax.set_xticks(np.arange(1, Q+1 + 1, step=4) - 0.5)
    ax.set_yticks(np.arange(10, step=3) - 0.5)
    ax.set_xticklabels(range(1, Q+1 + 1, 4))
    ax.set_yticklabels(range(0, 10 + 1, 3))
    ax.grid(True, linestyle='--', color='black', alpha=0.7)

    
    # Add colorbar
    if cbar:
        plt.colorbar(im, ax=ax)

    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight')
    
    return ax

def compare_codes(pred_codes, gt_codes, pred_ends, gt_ends, max_num_codes, out_file=None, names=None):
    # codes is 1, E, Q
    # ends is E
    """
    Compare two CAD vectors for equality.
    :param cad_vec1: First CAD vector.
    :param cad_vec2: Second CAD vector.
    :return: True if they are equal, False otherwise.
    """
    fig, axs = plt.subplots(1, 3)
    if names is not None:
        axs[0].set_title(names[0])
        axs[1].set_title(names[1])
        axs[2].set_title("Error Map")

    pred_codes[pred_ends == 1, :] = 0
    gt_codes[gt_ends == 1, :] = 0
    visualize_code(pred_codes, pred_ends, max_num_codes, ax=axs[0], cbar=False, error_plot=False)
    visualize_code(gt_codes, gt_ends, max_num_codes, ax=axs[1], cbar=False, error_plot=False)

    error = np.abs(pred_codes - gt_codes)
    error_end = np.abs(pred_ends - gt_ends)
    fig.set_figheight(3) # Set height to 8 inches
    fig.set_figwidth(10)  # Set width to 12 inches

    visualize_code(error, error_end, max_num_codes, ax=axs[2], cbar=True, error_plot=True)

    if out_file is not None:
        plt.savefig(out_file)
    else:
        plt.show()
    plt.close()


    return axs
