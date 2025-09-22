# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder.model_utils import _get_padding_mask, _get_visibility_mask
from cadlib.macro import CMD_ARGS_MASK


class CADLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_commands = cfg.n_commands
        self.args_dim = cfg.args_dim + 1
        self.weights = cfg.loss_weights

        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK).to(cfg.device))

    def forward(self, output):
        # Target & predictions
        tgt_commands, tgt_args = output["tgt_commands"], output["tgt_args"]

        # exclude firt token to make autoregressive
        tgt_commands, tgt_args = tgt_commands[:, 1:], tgt_args[:, 1:, :]
        if torch.min(tgt_commands) < 0 or torch.max(tgt_commands) > 5:
            print(torch.min(tgt_commands), torch.max(tgt_commands))
            tgt_commands = torch.clip(tgt_commands, 0, 5)

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)
        padding_mask = _get_padding_mask(tgt_commands, seq_dim=-1, extended=True) * visibility_mask.unsqueeze(-1)

        command_logits, args_logits = output["command_logits"], output["args_logits"]

        # exclude last token to make autoregressive
        command_logits, args_logits = command_logits[:, :-1], args_logits[:, :-1, :]
        
        mask = self.cmd_args_mask[tgt_commands.long()]

        c1 = command_logits[padding_mask.bool()].reshape(-1, self.n_commands)
        c2 = tgt_commands[padding_mask.bool()].reshape(-1).long()

        a1 = args_logits[mask.bool()].reshape(-1, self.args_dim)
        a2 = tgt_args[mask.bool()].reshape(-1).long() + 1

        loss_cmd = F.cross_entropy(c1, c2)
        if torch.min(a2) < 0 or torch.max(a2) > 256:
            print(torch.min(a2), torch.max(a2))
            a2 = torch.clip(a2, 0, 256)
        loss_args = F.cross_entropy(a1, a2)  # shift due to -1 PAD_VAL
        loss_cmd = self.weights["loss_cmd_weight"] * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        res = {"loss_cmd": loss_cmd, "loss_args": loss_args}
        return res