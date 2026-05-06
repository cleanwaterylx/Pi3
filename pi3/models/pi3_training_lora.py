import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, ContextTransformerDecoder, FeatureHead
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file

try:
    from peft import LoraConfig, LoraModel
except ImportError:
    LoraConfig = None
    LoraModel = None


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


class Pi3(nn.Module):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            load_vggt=True,
            freeze_encoder=True,
            freeze_decoder=False,
            use_global_points=False,
            train_conf=False,
            train_feature=False,
            num_dec_blk_not_to_checkpoint=4,
            ckpt=None,
            use_decoder_lora=False,
            decoder_lora_r=8,
            decoder_lora_alpha=16,
            decoder_lora_dropout=0.05,
            decoder_lora_targets=('attn.qkv', 'attn.proj'),
            decoder_lora_adapter_name='decoder_lora',
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):  # eg rope100
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError

        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # ----------------------
        #  Global Points Decoder
        # ----------------------
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2 * self.dec_embed_dim,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.use_decoder_lora = use_decoder_lora
        self.decoder_lora_r = decoder_lora_r
        self.decoder_lora_alpha = decoder_lora_alpha
        self.decoder_lora_dropout = decoder_lora_dropout
        self.decoder_lora_targets = tuple(decoder_lora_targets)
        self.decoder_lora_adapter_name = decoder_lora_adapter_name

        if load_vggt:
            vggt_weight = load_file('ckpts/VGGT-1B/model.safetensors')
            vggt_enc_weight = {
                k.replace('aggregator.patch_embed.', ''): vggt_weight[k]
                for k in list(vggt_weight.keys())
                if k.startswith('aggregator.patch_embed.')
            }
            print("Loading vggt encoder", self.encoder.load_state_dict(vggt_enc_weight, strict=False))

            vggt_dec_weight = {
                k.replace('aggregator.global_blocks.', ''): vggt_weight[k]
                for k in list(vggt_weight.keys())
                if k.startswith('aggregator.global_blocks.')
            }
            vggt_dec_weight1 = {}
            for k in list(vggt_dec_weight.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight1[f'{int(idx) * 2 + 1}{other}'] = vggt_dec_weight[k]
            vggt_dec_weight = vggt_dec_weight1

            vggt_dec_weight_frame = {
                k.replace('aggregator.frame_blocks.', ''): vggt_weight[k]
                for k in list(vggt_weight.keys())
                if k.startswith('aggregator.frame_blocks.')
            }
            for k in list(vggt_dec_weight_frame.keys()):
                idx = k.split('.')[0]
                other = k[len(idx):]
                vggt_dec_weight[f'{int(idx) * 2}{other}'] = vggt_dec_weight_frame[k]

            print("Loading vggt decoder", self.decoder.load_state_dict(vggt_dec_weight, strict=False))

        self.train_conf = train_conf
        if train_conf:
            assert ckpt is not None

            # ----------------------
            #     Conf Decoder
            # ----------------------
            self.conf_decoder = deepcopy(self.point_decoder)
            self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

            freeze_all_params([
                self.encoder,
                self.decoder,
                self.point_decoder,
                self.point_head,
                self.camera_decoder,
                self.camera_head,
                self.register_token,
            ])
            if use_global_points:
                freeze_all_params([self.global_points_decoder, self.global_point_head])

        self.train_feature = train_feature
        if train_feature:
            print('Training feature decoder and head.')
            self.feature_decoder = TransformerDecoder(
                in_dim=4 * self.dec_embed_dim,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.feature_head = FeatureHead(dec_embed_dim=1024, output_dim=512)
            freeze_all_params([
                self.encoder,
                self.point_decoder,
                self.point_head,
                self.camera_decoder,
                self.camera_head,
            ])

        if freeze_encoder:
            print('Freezing the encoder.')
            freeze_all_params([self.encoder])

        if freeze_decoder:
            print('Freezing the decoder point_decoder point_head register_token.')
            freeze_all_params([self.decoder, self.point_decoder, self.point_head, self.register_token])

        self.num_dec_blk_not_to_checkpoint = num_dec_blk_not_to_checkpoint

        if ckpt is not None:
            checkpoint_data = load_file(ckpt)

            res = self.load_state_dict(checkpoint_data, strict=False)
            print(f'[Pi3] Load checkpoints from {ckpt}: {res}')

            del checkpoint_data
            torch.cuda.empty_cache()

        if self.use_decoder_lora:
            self._apply_decoder_lora()
            self._set_decoder_lora_trainable()
            print(
                f'Applied LoRA to decoder: r={self.decoder_lora_r}, '
                f'alpha={self.decoder_lora_alpha}, dropout={self.decoder_lora_dropout}, '
                f'targets={self.decoder_lora_targets}'
            )

    def _apply_decoder_lora(self):
        if LoraConfig is None or LoraModel is None:
            raise ImportError("PEFT is required for decoder LoRA. Please install `peft` first.")

        lora_config = LoraConfig(
            r=self.decoder_lora_r,
            lora_alpha=self.decoder_lora_alpha,
            lora_dropout=self.decoder_lora_dropout,
            target_modules=list(self.decoder_lora_targets),
            bias="none",
        )

        for i, block in enumerate(self.decoder):
            self.decoder[i] = LoraModel(
                block,
                lora_config,
                adapter_name=self.decoder_lora_adapter_name,
            )

    def _set_decoder_lora_trainable(self):
        for name, param in self.decoder.named_parameters():
            param.requires_grad = 'lora_' in name

    def merge_decoder_lora(self, safe_merge=True, progressbar=False):
        merged_blocks = 0
        for i, block in enumerate(self.decoder):
            if hasattr(block, "merge_and_unload"):
                self.decoder[i] = block.merge_and_unload(
                    progressbar=progressbar,
                    safe_merge=safe_merge,
                    adapter_names=[self.decoder_lora_adapter_name],
                )
                merged_blocks += 1

        if merged_blocks > 0:
            self.use_decoder_lora = False

        return merged_blocks

    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B * N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H // self.patch_size, W // self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        intermediate_outputs = []

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))
            intermediate_outputs.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1), intermediate_outputs

    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        # encode by dinov2
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]  # (B*N, num_patches, dim)

        hidden, pos, intermediate_outputs = self.decode(hidden, N, H, W)
        # hidden: (B*N, num_patches+num_register_tokens, 2*dim)
        # pos: (B*N, num_patches+num_register_tokens, 2)

        point_hidden = self.point_decoder(hidden, xpos=pos)  # (B*N, num_patches+num_register_tokens, dim)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.train_feature:
            hidden_for_feature = torch.cat(
                [intermediate_outputs[2], intermediate_outputs[3], intermediate_outputs[-2], intermediate_outputs[-1]],
                dim=-1,
            )
            feature_hidden = self.feature_decoder(hidden_for_feature, xpos=pos)
        if self.use_global_points:
            context = hidden.reshape(B, N, patch_h * patch_w + self.patch_start_idx, -1)[:, 0:1].repeat(1, N, 1, 1).reshape(B * N, patch_h * patch_w + self.patch_start_idx, -1)
            global_point_hidden = self.global_points_decoder(hidden, context, xpos=pos, ypos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                conf = None

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            if self.train_feature:
                feature_hidden = feature_hidden.float()
                feat = self.feature_head(feature_hidden, N, self.patch_start_idx)
            else:
                feat = None

            # Global points
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head([global_point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            else:
                global_points = None

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            feat=feat,
            global_points=global_points
        )
