# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192,
                 signal_length=2048, signal_patch_size=32):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.signal_length = signal_length
        self.signal_patch_size = signal_patch_size
        self.num_patches = num_patches

        # Keep the original image projection for CIFAR compatibility, while
        # raw vibration signals use a genuine 1-D convolutional projection.
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.signal_proj = nn.Conv1d(
            1, embed_dim, kernel_size=signal_patch_size, stride=signal_patch_size
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(
                    f"1-D signal input must have shape [batch, 1, length], got {tuple(x.shape)}"
                )
            if x.shape[-1] < self.signal_patch_size:
                raise ValueError(
                    f"Signal length {x.shape[-1]} is smaller than patch size "
                    f"{self.signal_patch_size}"
                )
            return self.signal_proj(x).transpose(1, 2)
        if x.ndim == 4:
            return self.proj(x).flatten(2).transpose(1, 2)
        raise ValueError(
            f"Expected a 1-D signal [B, 1, L] or image [B, C, H, W], got {tuple(x.shape)}"
        )

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MetaExpert specific components
        self.channels = [embed_dim, embed_dim, embed_dim, embed_dim]
        # Indices to extract features. 
        # WRN uses outputs after 1st block, 2nd block, 3rd block, and final output.
        # We simulate this by tapping at 1/4, 1/2, 3/4, and full depth.
        self.stage_indices = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        
        # Expert BN (using BN1d since we pool to 1D vector)
        self.BNH = nn.BatchNorm1d(embed_dim)
        self.BNT = nn.BatchNorm1d(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes) 
        self.aux_classifier1 = nn.Linear(embed_dim, num_classes)
        self.aux_classifier2 = nn.Linear(embed_dim, num_classes)

        self.aux_classifier1.requires_grad_(False)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.pos_embed
        if pos_embed.shape[1] != x.shape[1]:
            # Permit other signal lengths as long as at least one complete
            # signal patch is available.
            cls_pos = pos_embed[:, :1]
            patch_pos = pos_embed[:, 1:].transpose(1, 2)
            patch_pos = F.interpolate(
                patch_pos, size=x.shape[1] - 1, mode="linear", align_corners=False
            ).transpose(1, 2)
            pos_embed = torch.cat((cls_pos, patch_pos), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        intermediate_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.stage_indices:
                # Use CLS token as feature
                intermediate_features.append(self.norm(x)[:, 0])
        
        # Final norm
        x = self.norm(x)
        return x[:, 0], intermediate_features

    def forward(self, x, only_feat=False, **kwargs):
        x, intermediates = self.forward_features(x)
        if only_feat:
            return x
        head_fs, tail_fs = self.BNH(x), self.BNT(x)
        logits = self.classifier(x)
        tail_logits = self.aux_classifier2(x)
        return {
            'feat': x,
            'feat_for_fuse': {f'feat{i + 1}': f for i, f in enumerate(intermediates)},
            'logits': logits,
            'aux_logits2': tail_logits,
            'logitsH': self.classifier(head_fs),
            'logitsT': self.classifier(tail_fs),
            'aux_logitsH2': self.aux_classifier2(head_fs),
            'aux_logitsT2': self.aux_classifier2(tail_fs),
        }


def vit_tiny_metaexpert(pretrained=False, pretrained_path=None, **kwargs):
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def vit_small_metaexpert(pretrained=False, pretrained_path=None, **kwargs):
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
