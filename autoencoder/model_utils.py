import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from cadlib.macro import EOS_IDX, SOL_IDX, EXT_IDX, CMD_ARGS_MASK



def _make_seq_first(*args):
    # N, S, ... -> S, N, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)


def _make_batch_first(*args):
    # S, N, ... -> N, S, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)


def _get_key_padding_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [s, ...]
    """
    with torch.no_grad():
        key_padding_mask = (commands == EOS_IDX).cumsum(dim=seq_dim) > 0

        if seq_dim == 0:
            return key_padding_mask.transpose(0, 1)
        return key_padding_mask


def _get_padding_mask(commands, seq_dim=0, extended=False):
    with torch.no_grad():
        padding_mask = (commands == EOS_IDX).cumsum(dim=seq_dim) == 0
        padding_mask = padding_mask.float()

        if extended:
            # padding_mask doesn't include the final EOS, extend by 1 position to include it in the loss
            S = commands.size(seq_dim)
            torch.narrow(padding_mask, seq_dim, 3, S-3).add_(torch.narrow(padding_mask.clone(), seq_dim, 0, S-3)).clamp_(max=1)

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask


def _get_group_mask(commands, seq_dim=0):
    """
    Args:
        commands: Shape [S, ...]
    """
    with torch.no_grad():
        # group_mask = (commands == SOS_IDX).cumsum(dim=seq_dim)
        group_mask = (commands == EXT_IDX).cumsum(dim=seq_dim)
        return group_mask


def _get_visibility_mask(commands, seq_dim=0): # Checks if sequence is full of EOS tokens
    """
    Args:
        commands: Shape [S, ...]
    """
    S = commands.size(seq_dim)
    with torch.no_grad():
        visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) < S - 1

        if seq_dim == 0:
            return visibility_mask.unsqueeze(-1)
        return visibility_mask


def _get_key_visibility_mask(commands, seq_dim=0):
    S = commands.size(seq_dim)
    with torch.no_grad():
        key_visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) >= S - 1

        if seq_dim == 0:
            return key_visibility_mask.transpose(0, 1)
        return key_visibility_mask


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _sample_categorical(temperature=0.0001, *args_logits):
    if len(args_logits) == 1:
        arg_logits, = args_logits
        return Categorical(logits=arg_logits / temperature).sample()
    return (*(Categorical(logits=arg_logits / temperature).sample() for arg_logits in args_logits),)


def _threshold_sample(arg_logits, threshold=0.5, temperature=1.0):
    scores = F.softmax(arg_logits / temperature, dim=-1)[..., 1]
    return scores > threshold


def logits2vec(outputs, refill_pad=True, to_numpy=True, device=None):
    """network outputs (logits) to final CAD vector"""
    # This should work for both batch-first and seq-first, and regardless of non-last dimensions
    out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)  # (B, S)
    out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1  # (B, S, N_ARGS)
    if refill_pad: # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda(out_args.device)[out_command.long()]
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec

def sample_logits2vec(outputs, refill_pad=True, to_numpy=True, device=None, temperature=1.0):
    """network outputs (logits) to final CAD vector, via sampling"""
    # command logits are  (B, S, 6)
    # arg logits are      (B, S, 16, 256)
    command_logits = torch.softmax(outputs['command_logits'] / temperature, dim=-1)
    args_logits    = torch.softmax(outputs['args_logits'] / temperature, dim=-1)

    B, S = command_logits.shape[:2]
    n_args, args_vocab_size = args_logits.shape[2:]
    cmd_vocab_size = command_logits.shape[2]
    flat_cmd = command_logits.view(-1, cmd_vocab_size)
    flat_args = args_logits.view(-1, args_vocab_size)

    # Top k
    top_k_cmd = 3
    # Get top k values and indices
    top_k_logits, top_k_indices = torch.topk(flat_cmd, top_k_cmd, dim=-1)
    # Set all other values to -inf
    flat_cmd = torch.full_like(flat_cmd, 0.)
    flat_cmd.scatter_(-1, top_k_indices, top_k_logits)

    top_k_arg = 50
    # Get top k values and indices
    top_k_logits, top_k_indices = torch.topk(flat_args, top_k_arg, dim=-1)
    # Set all other values to -inf
    flat_args = torch.full_like(flat_args, 0.)
    flat_args.scatter_(-1, top_k_indices, top_k_logits)

    out_command = torch.multinomial(flat_cmd, num_samples=1)
    out_args    = torch.multinomial(flat_args, num_samples=1)

    out_command = out_command.view(B, S)
    out_args = out_args.view(B, S, n_args)

    if refill_pad: # fill all unused element to -1
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda(out_args.device)[out_command.long()]
        out_args[mask] = -1

    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
