# Adapted from Alam et al, https://github.com/ferdous-alam/GenCAD

import copy
from contextlib import contextmanager
from functools import partial, wraps

import torch
import torch.nn.functional as F
# import torch.distributed as distributed
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint

from einops import rearrange, repeat, reduce
from x_clip.distributed import all_gather

from collections import namedtuple

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def pad_dim_to(t, length, dim = 0):
    pad_length = length - t.shape[dim]
    zero_pairs = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    return F.pad(t, (*((0, 0) * zero_pairs), 0, pad_length))

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

# contrastive learning functions

def model_forward_with_context(
    *,
    fn,
    args,
    kwargs,
    freeze,
    model=None,
):
    encoding_context = null_context if not freeze else torch.no_grad

    if freeze:
        if model is None:
            freeze_model_and_make_eval_(fn)
        else:
            freeze_model_and_make_eval_(model)
        # fn.eval()
    with encoding_context():
        if kwargs is not None:
            enc = fn(*args, **kwargs)
        else:
            enc = fn(*args)

        if freeze:
            enc = enc.clone()
            enc.detach_()

    return enc

# main clip class

class CLIP(nn.Module):
    def __init__(
        self,
        *,
        image_encoder = None,
        cad_encoder = None,
        dim_cad = 256,
        dim_image = 512 * 8 * 8,
        dim_latent = 256,
        **kwargs
    ):
        super().__init__()

        # store some parameters for access
        self.dim_cad = dim_cad
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        # instantiate cad transformer
        freeze_model_and_make_eval_(cad_encoder)
        self.cad_encoder = cad_encoder

        # instantiate image transformer
        self.image_encoder = image_encoder

        self.to_cad_latent = nn.Sequential(nn.Linear(dim_cad, dim_latent), 
                                        nn.Linear(dim_latent, dim_latent), 
                                        nn.Tanh())
 
        # image latent projection
        self.to_visual_latent = nn.Sequential(nn.Linear(dim_image, dim_latent), nn.Tanh())

        # temperature
        self.temperature = nn.Parameter(torch.tensor(1.))

        self.to_cad_latent_extra = copy.deepcopy(self.to_cad_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

        # is distributed or not
        self.requires_all_gather = False

    def forward(
        self,
        cad_list, # This is a tuple (cmd, args) or a list of tuples [(), (), ()]. Each tuple is fed one at a time to the encoder
        geom_list, # This is a tuple, passed directly to the image encoder, or a list of tuples [(), (), ()]. Each tuple is feed in one at a time to the encoder
        return_loss = False,
        return_encodings = False,
        return_latents = False,
        freeze_geometry_encoder = False,   # image encoder is not trained if this is set to True, proposed by LiT paper
        freeze_cad_encoder = False,    # cad encoder is not trained if this is set to True
    ):
        # First convert everything to common form as a list
        if not isinstance(cad_list, list):
            cad_list = [cad_list]
        if not isinstance(geom_list, list):
            geom_list = [geom_list]

        # temp_cad = cad_list[0]
        # batch, device = temp_cad[0].shape[0], temp_cad[0].device

        num_batch_cads = num_batch_images = 1

        # get encoded cad
        # Do multiple separately then concatenate
        enc_cad = []
        enc_geometry = []
        for (cad, image) in zip(cad_list, geom_list):
            batch_cmd, batch_args = cad[0], cad[1]
            cad_args = (batch_cmd, batch_args)
            cad_kwargs = {}
            # cad_kwargs = {'encode_mode': True}  # we only need the encoder
            enc_cad_instance = model_forward_with_context(
                fn = self.cad_encoder.encode_inference,
                args = cad_args,
                kwargs = cad_kwargs,
                freeze = freeze_cad_encoder,
                model = self.cad_encoder
            )

            enc_image_instance = model_forward_with_context(
                fn = self.image_encoder,
                args=image,
                kwargs=None,
                freeze = freeze_geometry_encoder
            )
            enc_cad.append(enc_cad_instance)
            enc_geometry.append(enc_image_instance)

        for parameter in self.image_encoder.parameters():
            if torch.isnan(parameter).any():
                print("encoder_paramers is nan")

        enc_cad = torch.concatenate(enc_cad, dim=0)
        enc_geometry = torch.concatenate(enc_geometry, dim=0)

        if torch.isnan(enc_geometry).any():
            print("encoded geometry is nan")

        enc_geometry = torch.nan_to_num(enc_geometry)


        # early return of encodings, if needed (for DALL-E2)
        if return_encodings:
            return enc_cad, enc_geometry

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only

        cad_embeds = enc_cad[:, 0] if enc_cad.ndim == 3 else enc_cad
        geometry_embeds = enc_geometry[:, 0] if enc_geometry.ndim == 3 else enc_geometry

        # project to latents
        cad_latents = self.to_cad_latent(cad_embeds)
        geometry_latents = self.to_visual_latent(geometry_embeds)
        cad_latents, geometry_latents = map(l2norm, (cad_latents, geometry_latents))

        # whether to early return latents

        if return_latents:
            return cad_latents, geometry_latents

        # get temperature
        temp = self.temperature.exp()
        if torch.isnan(self.temperature):
            self.temperature.data.fill_(1.)

        # early return, if needed
        if not return_loss:
            einsum_args = (cad_latents, geometry_latents)
            return einsum('b d, b d -> b', *einsum_args) * temp

        # split out multiview dimension for cad and images
        cad_latents = rearrange(cad_latents, '(m b) ... -> m b ...', m = num_batch_cads)
        geometry_latents = rearrange(geometry_latents, '(m b) ... -> m b ...', m = num_batch_images)

        # maybe distributed all gather
        if self.requires_all_gather:
            latents = torch.stack((cad_latents, geometry_latents))
            latents, sizes = all_gather(latents, 2, None)
            cad_latents, geometry_latents = latents

        # contrastive loss
        # 
        # m - num batches of text
        # n - num batches of geometries
        # x - batches of text
        # y - batches of geometries
        # t - sequence dimension along text tokens
        # i - sequence dimension along geometry tokens
        cad_to_image = einsum('m t d, n i d -> m n t i', cad_latents, geometry_latents) * temp
        image_to_cad = rearrange(cad_to_image, '... t i -> ... i t')

        # calculate loss
        cad_to_image = rearrange(cad_to_image, 'm n ... -> (m n) ...')
        image_to_cad = rearrange(image_to_cad, 'm n ... -> (m n) ...')

        # exponentiate
        cad_to_image_exp, image_to_cad_exp = map(torch.exp, (cad_to_image, image_to_cad))

        # numerators
        cad_to_geo_pos, geo_to_cad_pos = map(matrix_diag, (cad_to_image_exp, image_to_cad_exp))

        # denominator
        cad_to_geo_denom, geo_to_cad_denom = map(lambda t: t.sum(dim = -1), (cad_to_image_exp, image_to_cad_exp))

        if torch.isnan(cad_to_geo_pos).any() or torch.isnan(geo_to_cad_pos).any():
            print("c2i pos is nan")
        if torch.isnan(cad_to_geo_denom).any() or torch.isnan(geo_to_cad_denom).any():
            print("c2i denom is nan")

        if (cad_to_geo_pos <= 0).any() or (geo_to_cad_pos <= 0).any():
            cad_to_geo_pos[cad_to_geo_pos <= 0] = 1.0
            geo_to_cad_pos[geo_to_cad_pos <= 0] = 1.0
            print("c2i pos is < 0")
        if (cad_to_geo_denom <= 0).any() or (geo_to_cad_denom <= 0).any():
            cad_to_geo_denom[cad_to_geo_denom <= 0] = 1.0
            geo_to_cad_denom[geo_to_cad_denom <= 0] = 1.0
            print("c2i denom is < 0")
        # loss

        cad_to_geo_loss = (-log(cad_to_geo_pos) + log(cad_to_geo_denom)).mean(dim = -1)
        geo_to_cad_loss = (-log(geo_to_cad_pos) + log(geo_to_cad_denom)).mean(dim = -1)

        # calculate CL loss
        cl_losses = (cad_to_geo_loss + geo_to_cad_loss) / 2

        # get main CL loss vs multiview CL losses
        cl_loss, multiview_cl_loss = cl_losses[0], cl_losses[1:]

        loss = cl_loss

        # add similarity regularization loss with weight if needed
        if torch.isnan(loss).any():
            print("loss is nan")
        return loss
    



EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class BaseClipAdapter(nn.Module):
    def __init__(self, clip: CLIP, **kwargs):
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_image(self, image):
        raise NotImplementedError



class XClipAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim_latent

    @property
    def image_size(self):
        return self.clip.image_size

    @property
    def image_channels(self):
        return self.clip.image_channels

    @property
    def max_text_len(self):
        return self.clip.text_seq_len

    @torch.no_grad()
    def embed_cad(self, cad, normalization=True):
        """
        cad --> (B, d_model)
        returns: 
            cad_latents, cad_encodings
        """
        commands, args = cad
        encoder_output = self.clip.cad_encoder(commands, args, encode_mode=True)  # (B, 1, d_model)

        cad_cls, cad_encodings = encoder_output.squeeze(), encoder_output.squeeze()  # (B, d_model)
        cad_embed = self.clip.to_cad_latent(cad_cls)

        if normalization: 
            return EmbeddedText(l2norm(cad_embed), cad_encodings)
        else: 
            return EmbeddedText(cad_embed, cad_encodings)

    @torch.no_grad()
    def embed_image(self, *image, normalization=True):
        """returns: 
            image_latents, image_encodings
        """
        encoder_output = self.clip.image_encoder(*image)
        image_cls, image_encodings = encoder_output, encoder_output
        image_embed = self.clip.to_visual_latent(image_cls)

        if normalization: 
            return EmbeddedImage(l2norm(image_embed), image_encodings)
        else: 
            return EmbeddedImage(image_embed, image_encodings)



class DavinciClipAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim_latent

    @property
    def image_size(self):
        return self.clip.image_size

    @property
    def image_channels(self):
        return self.clip.image_channels

    @property
    def max_text_len(self):
        return self.clip.text_seq_len

    @torch.no_grad()
    def embed_cad(self, cad, normalization=True):
        """
        cad --> (B, d_model)0
        returns: 
            cad_latents, cad_encodings
        """
        commands, args = cad
        encoder_output = self.clip.cad_encoder(commands, args, encode_mode=True)  # (B, 1, d_model)

        cad_embed = encoder_output.squeeze()

        if normalization: 
            return l2norm(cad_embed)
        else: 
            return cad_embed


    @torch.no_grad()
    def embed_image(self, *image, normalization=True):
        """returns: 
            image_latents, image_encodings
        """
        encoder_output = self.clip.image_encoder(*image)
        image_encodings = encoder_output
        image_embed = self.clip.to_visual_latent(image_encodings)

        if normalization: 
            return l2norm(image_embed)
        else: 
            return image_embed
