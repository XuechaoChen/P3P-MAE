from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .MAE3Dsparse import PatchEmbed3D, Block
# import open3d as o3d
from .build import MODELS
from .pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


class SWITransformer(nn.Module):
    def __init__(self, space_size=224, patch_size=16, patch_num=196, in_chans=12, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, finetune=True, linear_probe=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, global_pool=False, smoothing=True):
        super().__init__()
        assert space_size%patch_size==0, f"Space cannot be divided with the patch size {patch_size}"
        self.grid_size = space_size // patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.smoothing = smoothing
        self.global_pool = global_pool
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        if self.global_pool:
            self.head = nn.Sequential(
                    nn.Linear(embed_dim*2, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                ) 
        else:
            self.head = nn.Linear(embed_dim, num_classes)
            # manually initialize fc layer
            trunc_normal_(self.head.weight, std=2e-5)

        self.patch_embed = PatchEmbed3D(patch_size, patch_num, in_chans, embed_dim, finetune=finetune)
        self.build_loss_func()
        self.linear_probe = linear_probe
        if self.linear_probe:
            self.freeze_for_linear_probe()

    def freeze_for_linear_probe(self):
        for name, param in self.named_parameters():
            if 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed_cls', 'cls_token'}
    
    def load_from_ckpts(self, ckpts):
        checkpoint = torch.load(ckpts, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % ckpts)
        checkpoint_model = checkpoint['base_model']
        state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, pred, gt):
        gt = gt.contiguous().view(-1).long()

        if self.smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def forward_features(self, samples):
        attn_mask = samples[1]['attn_mask']
        # patch_unq_hash = samples[1]['patch_unq_hash']
        x = self.patch_embed(samples)
        B, L, C = x.shape

        attn_mask = torch.cat((torch.zeros((B, 1, L), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask), dim=1)
        attn_mask = torch.cat((torch.zeros((B, L+1, 1), dtype=attn_mask.dtype, device=attn_mask.device), attn_mask), dim=2)

        cls_token = (self.cls_token + self.pos_embed_cls).expand(B, -1, -1)  # B, 1, C
        x = torch.cat((cls_token, x), dim=1) # B, 1+L, C
        # pos_embed = self.pos_embed_enc[:, 1:, :].reshape(-1, self.embed_dim)[patch_unq_hash].reshape(B, L, C)
        # pos_embed = torch.cat((self.pos_embed_enc[:, :1, :].expand(B, -1, -1), pos_embed), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, attn_mask)
        
        x = self.norm(x)

        if self.global_pool:
            outcome = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        else:
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@MODELS.register_module()
class SWITransformerSmall(SWITransformer):
    def __init__(self, config):
        if hasattr(config, "in_chans"):
            in_chans = config.in_chans
        else:
            in_chans = 12
        super().__init__(in_chans=in_chans, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                patch_num=config.patch_num, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=config.num_classes,
                drop_path_rate=config.drop_path_rate, global_pool=config.global_pool, linear_probe=config.linear_probe, 
                smoothing=config.smoothing)


@MODELS.register_module()
class SWITransformerBase(SWITransformer):
    def __init__(self, config):
        if hasattr(config, "in_chans"):
            in_chans = config.in_chans
        else:
            in_chans = 12
        super().__init__(in_chans=in_chans, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                patch_num=config.patch_num, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=config.num_classes,
                drop_path_rate=config.drop_path_rate, global_pool=config.global_pool, linear_probe=config.linear_probe, 
                smoothing=config.smoothing)
