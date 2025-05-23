from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
from timm.models.layers import to_2tuple

from torch_scatter import scatter
from .build import MODELS
from stack_chamfer import chamfer_loss
# import numpy as np
# import open3d as o3d


def mse_loss(pred, sel_features, indicator):
    loss = (pred - sel_features) ** 2
    loss = scatter(loss, indicator, dim=0, reduce="mean") # mean loss per sample
    return loss


def chamfer_plus_mse_loss(pred, sel_features, indicator):
    loss_xyz = chamfer_loss(pred[:, :3], sel_features[:, :3], indicator, indicator, reduce='mean')
    loss_other = mse_loss(pred[:, 3:], sel_features[:, 3:], indicator)
    return loss_xyz+loss_other


class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = SparseAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        y = self.norm1(x)
        y = self.attn(y, attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed3D(nn.Module):

    def __init__(self, patch_size=16, patch_num=196, in_chans=6, embed_dim=768,  decoder_embed_dim=512, finetune=False):
        super().__init__()
        self.in_chans = in_chans
        self.patch_num = patch_num
        self.pos_embed_enc = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.finetune = finetune
        if not self.finetune:
            self.pos_embed_dec = nn.Sequential(
                nn.Linear(3, decoder_embed_dim),
                nn.GELU(),
                nn.Linear(decoder_embed_dim, decoder_embed_dim),
            )
        self.proj_weights = nn.Parameter(torch.ones((patch_size**3, in_chans, embed_dim), dtype=torch.float32))

    def forward(self, info):
        features_rela = info[1]['sel_features']
        coords_rela_query = info[1]['coords_rela_query']
        features = torch.bmm(features_rela.reshape(features_rela.shape[0], 1, -1), self.proj_weights[coords_rela_query]).squeeze()
        features = scatter(features, info[1]['patch_unq_inv'], dim=0, reduce="mean")
        features_patch = torch.zeros((info['batch_size'], self.patch_num, features.shape[1]), dtype=features.dtype, device=features.device)
        features_patch = features_patch.reshape(-1, features.shape[1])
        features_patch[info[1]['pad_mask'], :] = features
        N, C = features_patch.shape
        features_patch = features_patch.reshape(info['batch_size'], N//info['batch_size'], C) # B, L, C
        coords_abs_cut = info[1]['coords_abs_cut']
        coords_abs_cut = coords_abs_cut.reshape(info['batch_size'], N//info['batch_size'], coords_abs_cut.shape[1]) # B, L, C
        pos_emb_enc = self.pos_embed_enc(coords_abs_cut) # B, L, C
        if not self.finetune:
            pos_emb_dec = self.pos_embed_dec(coords_abs_cut) # B, L, 512
            return features_patch + pos_emb_enc, info[1]['sel_features'].detach().clone(), info[1]['sel_coors'], pos_emb_dec
        else:
            return features_patch + pos_emb_enc


class MaskedAutoencoder(nn.Module):
    def __init__(self, space_size=224, patch_size=16, patch_num=196, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=480, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, loss_func='mae', mask_ratio=0.75):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed3D(patch_size, patch_num, in_chans, embed_dim, decoder_embed_dim)
        self.patch_num = patch_num
        self.space_size = space_size
        self.patch_size = patch_size
        assert space_size%patch_size==0, f"Space cannot be divided with the patch size {patch_size}"
        self.grid_size = space_size // patch_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed_enc = nn.Parameter(torch.zeros(1, self.grid_size**3 + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.pos_embed_dec = nn.Parameter(torch.zeros(1, self.grid_size**3 + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**3 * in_chans, bias=True) # decoder to patch
        self.occupancy_pred = nn.Linear(decoder_embed_dim, patch_size**3 * 2, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.in_chans = in_chans
        self.loss_func = loss_func
        self.mask_ratio = mask_ratio
        
        if loss_func=='mse':
            self.calculate_loss = mse_loss
        elif loss_func=='chamfer':
            self.calculate_loss = chamfer_plus_mse_loss

        self.ce_loss = nn.CrossEntropyLoss()
        # self.count = 0

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj_weights.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, attn_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        attn_mask_masked = torch.gather(attn_mask, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, L))
        attn_mask_masked = torch.gather(attn_mask_masked, dim=2, index=ids_keep.unsqueeze(1).repeat(1, len_keep, 1))
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, attn_mask_masked

    def forward_encoder(self, samples, inference=False):
        attn_mask = samples[1]['attn_mask']
        # patch_unq_hash = samples[1]['patch_unq_hash']

        # embed patches
        x, sel_features, sel_coors, pos_emb_dec = self.patch_embed(samples)
        # pos_embed = self.pos_embed_enc[:, 1:, :].reshape(-1, self.embed_dim)[patch_unq_hash].reshape(x.shape)
        # x = x + pos_embed

        # masking: length -> length * mask_ratio
        if not inference:
            x, mask, ids_restore, attn_mask_masked = self.random_masking(x, attn_mask)
        else:
            attn_mask_masked = attn_mask

        # append cls token
        B, L, C = x.shape
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        attn_mask_masked = torch.cat((torch.zeros((B, 1, L), dtype=attn_mask_masked.dtype, device=attn_mask_masked.device), attn_mask_masked), dim=1)
        attn_mask_masked = torch.cat((torch.zeros((B, L+1, 1), dtype=attn_mask_masked.dtype, device=attn_mask_masked.device), attn_mask_masked), dim=2)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask_masked)
        x = self.norm(x)

        if not inference:
            return x, mask, ids_restore, sel_features, sel_coors, pos_emb_dec
        else:
            return x[:, 0, :]

    def forward_decoder(self, x, ids_restore, pos_emb_dec, attn_mask):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        B, L, C = x_.shape
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        attn_mask = torch.cat((torch.zeros((B, 1, L), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask), dim=1)
        attn_mask = torch.cat((torch.zeros((B, L+1, 1), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask), dim=2)
        # add pos embed
        # pos_embed = self.pos_embed_dec[:, 1:, :].reshape(-1, self.decoder_embed_dim)[patch_unq_hash].reshape(B, L, C)
        # pos_embed = torch.cat((self.pos_embed_dec[:, :1, :].expand(B, -1, -1), pos_embed), dim=1)
        x[:, 1:, :] = x[:, 1:, :] + pos_emb_dec

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)
        x = self.decoder_norm(x)

        # predictor projection
        occu = self.occupancy_pred(x)
        pred = self.decoder_pred(x)

        # remove cls token
        occu = occu[:, 1:, :]
        pred = pred[:, 1:, :]

        return pred, occu
    
    def forward_loss(self, sel_features, sel_coors, pred, mask, occu):
        N, L, C = pred.shape
        pred = pred.reshape(N, L, C//self.in_chans, self.in_chans)
        occu = occu.reshape(N, L, self.patch_size, self.patch_size, self.patch_size, 2).permute(0, 5, 1, 2, 3, 4)

        with torch.no_grad():
            patch_indicator = sel_coors[:, 0] * L + sel_coors[:, 1]
            patch_indicator = patch_indicator.long()
            mask = mask.reshape(-1)
            mask = mask[patch_indicator]
            sel_coors = sel_coors[mask==1]
            sel_features = sel_features[mask==1]
            patch_indicator = patch_indicator[mask==1]

        target_query = sel_coors[:, -1]
        mul = 1
        for i in range(2, 4):
            mul = mul * pred.shape[-i]
            target_query = target_query + (sel_coors[:, -i] * mul)
        target_query = target_query.long()

        pred = pred.reshape(-1, self.in_chans)
        pred = pred[target_query]
        occu_gt = torch.zeros((N*L*self.patch_size**3), dtype=torch.int64, device=occu.device)
        occu_gt[target_query] = 1
        occu_gt = occu_gt.reshape(N, L, self.patch_size, self.patch_size, self.patch_size)

        loss = self.calculate_loss(pred.float(), sel_features.float(), sel_coors[:, 0].long())
        occu_loss = self.ce_loss(occu, occu_gt.long())

        loss = loss.sum() / mask.sum()  # mean loss on removed patches
        return loss + occu_loss

    def forward(self, samples, inference=False):
        if not inference:
            latent, mask, ids_restore, sel_features, sel_coors, pos_emb_dec = self.forward_encoder(samples, inference)
            pred, occu = self.forward_decoder(latent, ids_restore, pos_emb_dec, samples[1]['attn_mask'])
            loss = self.forward_loss(sel_features, sel_coors, pred, mask, occu)
            return loss
        else:
            x = self.forward_encoder(samples, inference)
            return x
        

@MODELS.register_module()
class MaskedAutoencoderSparseSmall(MaskedAutoencoder):
    def __init__(self, config):
        super().__init__(in_chans=config.in_chans, patch_size=16, patch_num=config.patch_num, 
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), loss_func=config.loss_func, 
        mask_ratio=config.mask_ratio, decoder_depth=4, decoder_num_heads=6, decoder_embed_dim=384)


@MODELS.register_module()
class MaskedAutoencoderSparseBase(MaskedAutoencoder):
    def __init__(self, config):
        super().__init__(in_chans=config.in_chans, patch_size=16, patch_num=config.patch_num, 
        embed_dim=768, depth=12, num_heads=12, norm_layer=partial(nn.LayerNorm, eps=1e-6), loss_func=config.loss_func, 
        mask_ratio=config.mask_ratio)