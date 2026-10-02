"""
Unified Multi-Modal Medical Segmentation Model (OneModel)

Combines innovations from the MedCLIPSeg project into a single model that handles
all modalities (ultrasound, MRI, CT, endoscopy, dermoscopy) with:
1. Modality-aware encoding via learnable modality embeddings
2. Cross-scale probabilistic fusion with uncertainty propagation
3. Frequency-spatial decomposition decoder with dual-prompt attention
4. Boundary-aware contrastive learning
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.nn import functional as F
from open_clip_lib import create_model_and_transforms, HFTokenizer, get_mean_std
from typing import Optional
from trainers.layers import PVL_Adapter
from trainers.scale_block import ScaleBlock
from huggingface_hub import hf_hub_download


def download_checkpoint(filename: str):
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    local_path = os.path.join(ckpt_dir, filename)
    if os.path.isfile(local_path):
        return local_path
    hf_hub_download(
        repo_id="TahaKoleilat/MedCLIPSeg",
        repo_type="model",
        filename=f"checkpoints/{filename}",
        local_dir=os.path.join(os.path.dirname(__file__), '..'),
        local_dir_use_symlinks=False,
    )
    return local_path


def load_unimedclip_to_device(cfg):
    if cfg.MODEL.BACKBONE == "ViT-B/16":
        model_name = 'ViT-B-16-quickgelu'
        pretrained_weights = download_checkpoint("unimed_clip_vit_b16.pt")
    else:
        raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not implemented.")
    text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    mean, std = get_mean_std()
    device = cfg.MODEL.DEVICE
    model, _, _ = create_model_and_transforms(
        model_name, pretrained_weights,
        precision='amp', device=device, force_quick_gelu=True,
        mean=mean, std=std, inmem=True,
        text_encoder_name=text_encoder_name,
    )
    return model.to(device).eval()


# ============================================================
# Modality-Aware Encoding
# ============================================================

class ModalityEmbedding(nn.Module):
    """Learnable modality embeddings added to vision tokens to condition
    the model on the imaging modality."""

    def __init__(self, num_modalities=5, embed_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x_img, modality_ids):
        # x_img: [S, B, D] (LND format)
        # modality_ids: [B] integer tensor
        mod_emb = self.embedding(modality_ids)  # [B, D]
        mod_emb = mod_emb.unsqueeze(0)  # [1, B, D]
        return x_img + mod_emb


# ============================================================
# Uncertainty Propagation Modules
# ============================================================

class UncertaintyExtractor(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, vis_pvl_residual):
        return self.net(vis_pvl_residual)


class UncertaintyMemoryBank(nn.Module):
    def __init__(self, num_layers=10):
        super().__init__()
        self.alpha_logits = nn.Parameter(torch.zeros(num_layers))

    def get_alpha(self, layer_idx):
        return torch.sigmoid(self.alpha_logits[layer_idx])

    def update(self, memory, new_unc, layer_idx):
        if memory is None:
            return new_unc
        alpha = self.get_alpha(layer_idx)
        return alpha * memory + (1 - alpha) * new_unc


class UncertaintyGuidedModulation(nn.Module):
    def __init__(self, embed_dim=768, num_layers=10, init_gamma=0.01):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((num_layers,), init_gamma))
        self.channel_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Tanh(),
        )

    def forward(self, x_img, memory, layer_idx):
        if memory is None:
            return x_img
        mod = self.channel_proj(memory)  # [B, S, D]
        mod = mod.permute(1, 0, 2)  # [S, B, D]
        gamma = self.gamma[layer_idx]
        return x_img * (1.0 + gamma * mod)


# ============================================================
# Frequency-Spatial Decomposition
# ============================================================

class FrequencyBranch(nn.Module):
    def __init__(self, dim, freq_ratio=0.25):
        super().__init__()
        self.freq_ratio = freq_ratio
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

    def _freq_split(self, x):
        B, C, H, W = x.shape
        freq = torch.fft.rfft2(x.float(), norm='ortho')
        mask = torch.zeros(B, C, H, W // 2 + 1, device=x.device)
        r_h = int(H * self.freq_ratio)
        r_w = int((W // 2 + 1) * self.freq_ratio)
        mask[:, :, :r_h, :r_w] = 1.0
        mask[:, :, -r_h:, :r_w] = 1.0
        low_freq = torch.fft.irfft2(freq * mask, s=(H, W), norm='ortho').to(x.dtype)
        high_freq = torch.fft.irfft2(freq * (1 - mask), s=(H, W), norm='ortho').to(x.dtype)
        return low_freq, high_freq

    def forward(self, x):
        low_freq, high_freq = self._freq_split(x)
        return self.low_freq_conv(low_freq), self.high_freq_conv(high_freq)


# ============================================================
# Dual-Prompt Cross-Modal Attention
# ============================================================

class DualPromptCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj_spatial = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.q_proj_freq = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj_spatial = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.out_proj_freq = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.gate = nn.Parameter(torch.tensor(0.5))

    def _cross_attn(self, q_proj, out_proj, visual_feats, text_seq):
        B, C, H, W = visual_feats.shape
        L = text_seq.shape[1]

        Q = q_proj(visual_feats).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        K = self.k_proj(text_seq).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(text_seq).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return out_proj(out) + visual_feats

    def forward(self, spatial_feats, freq_feats, spatial_text, freq_text):
        spatial_out = self._cross_attn(self.q_proj_spatial, self.out_proj_spatial, spatial_feats, spatial_text)
        freq_out = self._cross_attn(self.q_proj_freq, self.out_proj_freq, freq_feats, freq_text)
        gate = torch.sigmoid(self.gate)
        return gate * spatial_out + (1 - gate) * freq_out


# ============================================================
# Unified Multi-Scale Decoder with Bayesian Fusion + Freq-Spatial
# ============================================================

class UnifiedMultiScaleDecoder(nn.Module):
    """Combines Bayesian precision-weighted multi-scale fusion with
    frequency-spatial decomposition and dual-prompt attention."""

    def __init__(self, embed_dim, text_proj_dim, num_upscale=2,
                 ms_layer_indices=[2, 5, 8, 11]):
        super().__init__()
        self.ms_layer_indices = ms_layer_indices
        self.text_proj_dim = text_proj_dim
        num_scales = len(ms_layer_indices)

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, text_proj_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, text_proj_dim),
                nn.GELU()
            ) for _ in range(num_scales)
        ])

        self.var_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(text_proj_dim, text_proj_dim // 4, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(text_proj_dim // 4, 1, kernel_size=1),
                nn.Softplus(),
            ) for _ in range(num_scales)
        ])

        self.enc_unc_proj = nn.ModuleList([
            nn.Conv2d(1, text_proj_dim, kernel_size=1, bias=False)
            for _ in range(num_scales)
        ])

        self.freq_branch = FrequencyBranch(text_proj_dim, freq_ratio=0.25)
        self.dual_prompt_attn = DualPromptCrossAttention(text_proj_dim, num_heads=8)

        self.upscale = nn.Sequential(
            *[ScaleBlock(text_proj_dim) for _ in range(num_upscale)]
        )
        self.pred_head = nn.Conv2d(text_proj_dim, 1, kernel_size=1)

    def forward(self, hidden_states, spatial_text_feats, freq_text_feats,
                encoder_uncertainties=None, target_size=(224, 224)):
        extracted = []
        for idx_idx, l_idx in enumerate(self.ms_layer_indices):
            feat = hidden_states[l_idx]
            feat = feat.permute(1, 0, 2)  # [B, S, D]
            feat = feat[:, 1:, :]  # drop CLS
            B, num_patch, D = feat.shape
            grid = int(num_patch ** 0.5)
            feat = feat.reshape(B, grid, grid, D).permute(0, 3, 1, 2)
            extracted.append(self.lateral_convs[idx_idx](feat))

        if encoder_uncertainties is not None:
            for i in range(len(extracted)):
                if i < len(encoder_uncertainties) and encoder_uncertainties[i] is not None:
                    unc_map = encoder_uncertainties[i]
                    extracted[i] = extracted[i] + self.enc_unc_proj[i](unc_map)

        # Bayesian precision-weighted fusion
        scale_vars = [self.var_heads[i](extracted[i]) for i in range(len(extracted))]
        precisions = [1.0 / (v + 1e-6) for v in scale_vars]
        target_h, target_w = extracted[0].shape[-2:]

        weighted_sum = torch.zeros_like(extracted[0])
        precision_sum = torch.zeros(B, 1, target_h, target_w, device=extracted[0].device)

        for i, (feat, prec) in enumerate(zip(extracted, precisions)):
            feat_resized = F.interpolate(feat, size=(target_h, target_w),
                                         mode="bilinear", align_corners=False)
            prec_resized = F.interpolate(prec, size=(target_h, target_w),
                                         mode="bilinear", align_corners=False)
            weighted_sum = weighted_sum + prec_resized * feat_resized
            precision_sum = precision_sum + prec_resized

        fused = weighted_sum / (precision_sum + 1e-8)
        fused_uncertainty = 1.0 / (precision_sum + 1e-8)

        # Frequency-spatial decomposition
        low_freq_feats, high_freq_feats = self.freq_branch(fused)
        out = self.dual_prompt_attn(low_freq_feats, high_freq_feats,
                                     spatial_text_feats, freq_text_feats)

        out = self.upscale(out)
        out = self.pred_head(out)
        out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False).squeeze(1)

        fused_unc_out = F.interpolate(fused_uncertainty, size=target_size,
                                       mode="bilinear", align_corners=False).squeeze(1)
        return out, fused_unc_out


# ============================================================
# Boundary Contrastive Components
# ============================================================

class BoundaryProjectionHead(nn.Module):
    def __init__(self, in_dim=512, proj_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)


def get_boundary_patch_masks(gt_mask, grid_size=14):
    B, H, W = gt_mask.shape
    kernel_size = H // grid_size
    mask_4d = gt_mask.unsqueeze(1).float()
    patch_mask = F.avg_pool2d(mask_4d, kernel_size=kernel_size)
    fg_mask = (patch_mask > 0.5).float()
    dilated = F.max_pool2d(fg_mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-fg_mask, kernel_size=3, stride=1, padding=1)
    boundary_band = ((dilated - eroded) > 0.5).squeeze(1)
    fg_bool = (fg_mask.squeeze(1) > 0.5)
    boundary_fg = (boundary_band & fg_bool).view(B, -1)
    boundary_bg = (boundary_band & ~fg_bool).view(B, -1)
    interior_fg = (~boundary_band & fg_bool).view(B, -1)
    return boundary_fg, boundary_bg, interior_fg


class BoundaryContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, patch_embeds, text_embed, boundary_fg, boundary_bg, interior_fg):
        B = patch_embeds.shape[0]
        total_loss = 0.0
        count = 0
        for b in range(B):
            anchors = patch_embeds[b][boundary_fg[b]]
            positives = patch_embeds[b][interior_fg[b]]
            negatives = patch_embeds[b][boundary_bg[b]]
            if anchors.shape[0] == 0 or negatives.shape[0] == 0:
                continue
            text_pos = text_embed[b].unsqueeze(0)
            all_pos = torch.cat([positives, text_pos], dim=0) if positives.shape[0] > 0 else text_pos
            pos_sim = anchors @ all_pos.T / self.temperature
            neg_sim = anchors @ negatives.T / self.temperature
            pos_logsumexp = torch.logsumexp(pos_sim, dim=1)
            all_logits = torch.cat([pos_sim, neg_sim], dim=1)
            all_logsumexp = torch.logsumexp(all_logits, dim=1)
            total_loss += -(pos_logsumexp - all_logsumexp).mean()
            count += 1
        if count == 0:
            return torch.tensor(0.0, device=patch_embeds.device, requires_grad=True)
        return total_loss / count


class UncertaintyCalibrationLoss(nn.Module):
    def __init__(self, sparsity_weight=0.01):
        super().__init__()
        self.sparsity_weight = sparsity_weight

    def forward(self, uncertainty_map, seg_logits, gt_mask):
        pred_error = torch.abs(torch.sigmoid(seg_logits) - gt_mask.float())
        unc_flat = uncertainty_map.reshape(uncertainty_map.shape[0], -1)
        err_flat = pred_error.reshape(pred_error.shape[0], -1)
        unc_centered = unc_flat - unc_flat.mean(dim=1, keepdim=True)
        err_centered = err_flat - err_flat.mean(dim=1, keepdim=True)
        corr = (unc_centered * err_centered).sum(dim=1) / (
            unc_centered.norm(dim=1) * err_centered.norm(dim=1) + 1e-8
        )
        return -corr.mean() + self.sparsity_weight * unc_flat.mean()


# ============================================================
# Main Unified Model
# ============================================================

class UnifiedMedSeg(nn.Module):
    """
    Unified Multi-Modal Medical Segmentation Model.

    Key innovations:
    1. Modality embeddings condition the vision encoder on imaging modality
    2. Cross-scale uncertainty propagation through PVL adapter layers
    3. Bayesian precision-weighted multi-scale fusion decoder
    4. Frequency-spatial decomposition with dual-prompt attention
    5. Boundary-aware contrastive learning for sharp segmentation edges
    """

    SPATIAL_PROMPT = "Global anatomical shape and structure"
    FREQ_PROMPT = "Fine boundary details and local texture patterns"

    def __init__(self, cfg, clip_model):
        super().__init__()

        self.cfg = cfg
        self.vision_model = clip_model.visual
        self.text_model = clip_model.text_encoder
        self.logit_scale = clip_model.logit_scale
        self.temperature = getattr(cfg.MODEL, "TEMPERATURE", 0.2)
        self.fusion_stages = cfg.MODEL.LAYERS

        if cfg.MODEL.BACKBONE == "ViT-B/16":
            self.embed_dim = 768
            self.patch_size = 16
            self.text_proj_dim = 512
        else:
            raise NotImplementedError(f"Backbone {cfg.MODEL.BACKBONE} not supported.")

        self.dtype = self.text_model.transformer.dtype
        self.im_size = cfg.DATASET.SIZE
        self.device = cfg.MODEL.DEVICE

        self.tokenizer = HFTokenizer(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            context_length=256, **{},
        )

        adapter_channels = cfg.MODEL.ADAPTER_DIM
        self.num_upscale = cfg.MODEL.NUM_UPSCALE
        self.beta = cfg.MODEL.BETA
        self.gate_init = cfg.MODEL.GATE_INIT

        # PVL Adapters (frozen from base checkpoint, fine-tuned optionally)
        self.pvl_adapters = nn.ModuleList([
            PVL_Adapter(
                in_channels_vis=self.embed_dim, in_channels_txt=self.embed_dim,
                adapter_channels=adapter_channels, beta=self.beta, gate_init=self.gate_init
            ) for _ in range(len(self.fusion_stages))
        ])

        # === Innovation 1: Modality-Aware Encoding ===
        num_modalities = getattr(cfg.MODEL, "NUM_MODALITIES", 5)
        self.modality_embedding = ModalityEmbedding(num_modalities, self.embed_dim)

        # === Innovation 2: Cross-Scale Uncertainty Propagation ===
        num_adapters = len(self.fusion_stages)
        unc_hidden = getattr(cfg.MODEL, "UNC_EXTRACTOR_HIDDEN", 128)
        unc_gamma_init = getattr(cfg.MODEL, "UNC_MODULATION_INIT", 0.01)

        self.unc_extractors = nn.ModuleList([
            UncertaintyExtractor(self.embed_dim, unc_hidden)
            for _ in range(num_adapters)
        ])
        self.unc_memory_bank = UncertaintyMemoryBank(num_adapters)
        self.unc_modulation = UncertaintyGuidedModulation(
            self.embed_dim, num_adapters, unc_gamma_init
        )

        # === Innovation 3: Unified Multi-Scale Decoder ===
        self.decoder_tap_layers = getattr(cfg.MODEL, "DECODER_TAP_LAYERS", [2, 5, 8, 11])
        self.decoder = UnifiedMultiScaleDecoder(
            self.embed_dim, self.text_proj_dim,
            num_upscale=self.num_upscale,
            ms_layer_indices=self.decoder_tap_layers
        )

        # === Innovation 4: Boundary Contrastive Head ===
        boundary_proj_dim = getattr(cfg.MODEL, "BOUNDARY_PROJ_DIM", 128)
        self.boundary_proj = BoundaryProjectionHead(
            in_dim=self.text_proj_dim, proj_dim=boundary_proj_dim
        )

    def _encode_auxiliary_prompt(self, prompt_str, batch_size):
        tokens = self.tokenizer([prompt_str] * batch_size).to(self.device)
        with torch.no_grad():
            embeds = self.text_model.transformer.embeddings.word_embeddings(tokens).type(self.dtype)
            attention_mask = (tokens != self.text_model.config.pad_token_id).long()
            extended_mask = attention_mask[:, None, None, :]
            extended_mask = extended_mask.to(dtype=self.dtype)
            extended_mask = (1.0 - extended_mask) * torch.finfo(self.dtype).min

            x = self.text_model.transformer.embeddings(inputs_embeds=embeds)
            for layer in self.text_model.transformer.encoder.layer:
                x = layer(x, attention_mask=extended_mask)
                if isinstance(x, tuple):
                    x = x[0]
        return self.text_model.proj(x)

    def encode_text_image(self, tokenized_prompts, text_prompts, image, modality_ids,
                          attention_mask: Optional[torch.LongTensor] = None):
        if attention_mask is None:
            attention_mask = (tokenized_prompts != self.text_model.config.pad_token_id).long()

        x_txt = self.text_model.transformer.embeddings(inputs_embeds=text_prompts)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        x_img = self.vision_model.conv1(image)
        x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], -1)
        x_img = x_img.permute(0, 2, 1)
        x_img = torch.cat([
            self.vision_model.class_embedding.to(x_img.dtype) +
            torch.zeros(x_img.shape[0], 1, x_img.shape[-1], dtype=x_img.dtype, device=x_img.device),
            x_img
        ], dim=1)

        pos_embed = self.vision_model.positional_embedding.to(x_img.dtype)
        if pos_embed.shape[0] != x_img.shape[1]:
            cls_pos = pos_embed[:1, :]
            patch_pos = pos_embed[1:, :]
            orig_grid = int(patch_pos.shape[0] ** 0.5)
            new_grid = int((x_img.shape[1] - 1) ** 0.5)
            patch_pos = patch_pos.reshape(1, orig_grid, orig_grid, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(new_grid * new_grid, -1)
            pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        x_img = x_img + pos_embed
        x_img = self.vision_model.ln_pre(x_img)
        x_img = x_img.permute(1, 0, 2)  # [S, B, D]

        # Add modality embedding
        x_img = self.modality_embedding(x_img, modality_ids)

        hidden_states = []
        unc_memory = None
        layer_uncertainties = {}

        for i, (block, layer) in enumerate(zip(
            self.vision_model.transformer.resblocks,
            self.text_model.transformer.encoder.layer
        )):
            if i in self.fusion_stages:
                adapter_idx = self.fusion_stages.index(i)

                if unc_memory is not None:
                    x_img = self.unc_modulation(x_img, unc_memory, adapter_idx)

                vis_pvl, txt_pvl = self.pvl_adapters[adapter_idx](
                    x_img.transpose(1, 0), x_txt
                )
                x_txt = x_txt + txt_pvl
                x_img = x_img + vis_pvl.transpose(1, 0)

                vis_pvl_batch = vis_pvl.transpose(1, 0).permute(1, 0, 2)
                unc_i = self.unc_extractors[adapter_idx](vis_pvl_batch)
                unc_memory = self.unc_memory_bank.update(unc_memory, unc_i, adapter_idx)

                if i in self.decoder_tap_layers:
                    patch_unc = unc_memory[:, 1:, :]
                    B_unc = patch_unc.shape[0]
                    grid = int(patch_unc.shape[1] ** 0.5)
                    spatial_unc = patch_unc.reshape(B_unc, grid, grid, 1).permute(0, 3, 1, 2)
                    layer_uncertainties[i] = spatial_unc

            x_img = block(x_img)
            x_txt = layer(x_txt, attention_mask=extended_attention_mask)
            hidden_states.append(x_img)

            if isinstance(x_txt, tuple):
                x_txt = x_txt[0]

        x_img = x_img.permute(1, 0, 2)
        x_img = self.vision_model.ln_post(x_img)

        if self.vision_model.proj is not None:
            x_img = x_img @ self.vision_model.proj

        x_txt_seq = self.text_model.proj(x_txt)
        pooled_out = x_txt[:, 0, :]
        projected = self.text_model.proj(pooled_out)

        enc_uncs = [layer_uncertainties.get(l, None) for l in self.decoder_tap_layers]
        return x_img, hidden_states, projected, x_txt_seq, enc_uncs

    def forward(self, image, text, modality_ids=None):
        B, C, H, W = image.shape

        if modality_ids is None:
            modality_ids = torch.zeros(B, dtype=torch.long, device=image.device)

        tokenized_prompts = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            prompts = self.text_model.transformer.embeddings.word_embeddings(
                tokenized_prompts
            ).type(self.dtype)

        image_features, hidden_states, text_pooled, text_seq_feats, enc_uncs = \
            self.encode_text_image(tokenized_prompts, prompts, image, modality_ids)

        spatial_text_feats = self._encode_auxiliary_prompt(self.SPATIAL_PROMPT, B)
        freq_text_feats = self._encode_auxiliary_prompt(self.FREQ_PROMPT, B)

        seg_logits, fused_uncertainty = self.decoder(
            hidden_states, spatial_text_feats, freq_text_feats,
            encoder_uncertainties=enc_uncs, target_size=(H, W)
        )

        if self.training:
            patch_logits = image_features[:, 1:, :]
            patch_logits = patch_logits / patch_logits.norm(dim=-1, keepdim=True)
            patch_mean = patch_logits.mean(dim=1)

            text_pooled_norm = text_pooled / text_pooled.norm(dim=-1, keepdim=True)

            logits_per_image = (patch_mean @ text_pooled_norm.T) / self.temperature
            logits_per_text = (text_pooled_norm @ patch_mean.T) / self.temperature

            with torch.no_grad():
                text_sim = (text_pooled_norm @ text_pooled_norm.T) / self.temperature
                text_sim = text_sim / text_sim.norm(dim=-1, keepdim=True)
                soft_targets = F.softmax(text_sim, dim=-1)

            log_probs_i = F.log_softmax(logits_per_image, dim=-1)
            log_probs_t = F.log_softmax(logits_per_text, dim=-1)
            loss_i2t = -(soft_targets * log_probs_i).sum(dim=-1).mean()
            loss_t2i = -(soft_targets.T * log_probs_t).sum(dim=-1).mean()
            contrastive_loss = (loss_i2t + loss_t2i) / 2

            boundary_patch_embeds = self.boundary_proj(patch_logits)
            boundary_text_embed = self.boundary_proj(
                text_pooled_norm.unsqueeze(1)
            ).squeeze(1)

            return (seg_logits, fused_uncertainty, contrastive_loss,
                    boundary_patch_embeds, boundary_text_embed)

        return seg_logits, fused_uncertainty


def build_unified_model(cfg):
    print(f"Loading UniMedCLIP (backbone: {cfg.MODEL.BACKBONE})")
    clip_model = load_unimedclip_to_device(cfg)
    clip_model.float()

    print("Building Unified Multi-Modal Segmentation Model")
    model = UnifiedMedSeg(cfg, clip_model)

    # Freeze backbone encoders
    for name, param in model.named_parameters():
        if any(k in name for k in [
            "decoder", "boundary_proj", "unc_extractors",
            "unc_memory_bank", "unc_modulation", "modality_embedding",
            "pvl_adapters"
        ]):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,} parameters")

    return model
