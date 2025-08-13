import torch
import torch.nn as nn
from flamingo_core.helpers import PerceiverResampler, PerceiverAttention, FeedForward
from endo_fm_backbone.timesformer import get_vit_base_patch16_224
from pmc_clip_backbone import ModifiedResNet, AttentionPool2d
from einops import rearrange, repeat

class CrossAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, queries, keys_values):
        b, T, nq, d = queries.shape
        _, _, nkv, _ = keys_values.shape

        queries = self.norm_q(queries)
        keys_values = self.norm_kv(keys_values)

        q = self.to_q(queries)
        k, v = self.to_kv(keys_values).chunk(2, dim=-1)

        q = rearrange(q, 'b T nq (h d) -> b h T nq d', h=self.heads)
        k = rearrange(k, 'b T nkv (h d) -> b h T nkv d', h=self.heads)
        v = rearrange(v, 'b T nkv (h d) -> b h T nkv d', h=self.heads)

        q = q * self.scale

        attn = torch.einsum('b h T i d, b h T j d -> b h T i j', q, k)
        attn = attn.softmax(dim=-1)

        out = torch.einsum('b h T i j, b h T j d -> b h T i d', attn, v)
        out = rearrange(out, 'b h T nq d -> b T nq (h d)', h=self.heads)

        return self.to_out(out)


class VisionQFormer(nn.Module):
    def __init__(self, dim, num_queries=64, depth=2, heads=8, dim_head=64):
        super(VisionQFormer, self).__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttention(dim=dim, heads=heads, dim_head=dim_head),
                FeedForward(dim=dim)
            ])for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, visual_tokens):
        b, T, n, d = visual_tokens.shape
        queries = self.queries.expand(b, T, -1, -1)  # (b, T, num_queries, d)
        for attn, ff in self.layers:
            queries = queries + attn(queries, visual_tokens)
            queries = queries + ff(queries)
        return self.norm(queries)

class EndoFMAdapter(nn.Module):
    """
    将Endo-FM模型提取的视频特征，映射为Flamingo所需的固定数量的视觉token，以便与语言模型（BenTsao）进行跨模态融合
    EndoFMAdapter 模块封装了 Endo-FM Transformer，并使用线性映射和 Perceiver Resampler
    将可变长度的视频帧特征压缩为固定数量的视觉 token。
    """

    def __init__(self, cfg, endo_checkpoint_path: str,
                 target_hidden_dim: int = 4096, num_latents: int = 64,
                 perceiver_depth: int = 2, perceiver_heads: int = 8, perceiver_dim_head: int = 64,
                 freeze_endo: bool = True):

        super(EndoFMAdapter, self).__init__()

        # 初始化 Endo-FM backbone 模型
        self.cfg = cfg
        self.endo_model = get_vit_base_patch16_224(cfg=self.cfg, no_head=True)  # 不带分类头

        # 加载预训练权重
        ckpt = torch.load(endo_checkpoint_path, map_location='cpu')
        state_dict = {k[len("backbone."):]: v for k, v in ckpt.items() if k.startswith("backbone.")}
        if len(state_dict) == 0:
            state_dict = ckpt
        missing = self.endo_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded Endo-FM checkpoint. Missing: {missing.missing_keys}, Unexpected: {missing.unexpected_keys}")

        # 冻结模型参数（如果需要）
        if freeze_endo:
            for param in self.endo_model.parameters():
                param.requires_grad = False

        # 获取特征维度
        # self.endo_output_dim = self.endo_model.embed_dim
        self.endo_output_dim = int(self.endo_model.embed_dim) if not isinstance(self.endo_model.embed_dim,
                                                                                int) else self.endo_model.embed_dim  # 确定Endo-FM模型输出特征的维度
        # 特征维度映射
        self.linear_proj = nn.Linear(self.endo_output_dim, target_hidden_dim)  # 定义线性层，将Endo-FM输出特征维度转换到目标维度

        # Perceiver Resampler
        self.perceiver_resampler = PerceiverResampler(dim=target_hidden_dim, depth=perceiver_depth,
                                                      heads=perceiver_heads, dim_head=perceiver_dim_head,
                                                      num_latents=num_latents)  # 用于压缩特征序列长度

        self.output_dim = target_hidden_dim
        # self.visual = self

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        :param video_frames: (B, T, C, H, W)
        :return: (B, T, num_latents, target_hidden_dim)
        """
        B, T, C, H, W = video_frames.shape

        # 1) TimeSformer 期望 (B, C, T, H, W) —— 一律先换轴
        x = video_frames.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        # 2) 抽特征（TimeSformer 的 get_vit_base_patch16_224）
        features = self.endo_model.forward_features(x, get_all=True)

        D = self.endo_output_dim  # (= self.endo_model.embed_dim)

        # 3) 统一整理成 (B, T, N_patch, D)
        if features.dim() == 4:
            # 常见返回：(B, T, L, D) 或 (B, T, 1+L, D)；安全地去掉 CLS（若存在）
            patch_tokens = features[:, :, 1:, :] if features.size(2) > 0 else features
        elif features.dim() == 3:
            # 常见返回：(B*T, L, D) 或 (B*T, 1+L, D)
            tokens_no_cls = features[:, 1:, :] if features.size(1) > 0 else features  # 通常第0个是 CLS
            L = tokens_no_cls.size(1)  # 每帧 token 数
            patch_tokens = tokens_no_cls.view(B, T, L, D)
        else:
            raise RuntimeError(f"Unexpected Endo-FM feature shape: {features.shape}")

        # 4) 线性映射 + Perceiver 压缩到固定 num_latents
        mapped_tokens = self.linear_proj(patch_tokens)  # (B, T, N_patch, hidden)
        image_embeds = self.perceiver_resampler(mapped_tokens)  # (B, T, num_latents, hidden)
        return image_embeds


class PMCClipAdapter(nn.Module):
    """
    PMCClipAdapter encapsulates the PMC-CLIP pre-trained ResNet50 image encoder:
    - Loads a PMC-CLIP image model's local weights (state_dict), removing the attention pooling and projection head, keeping only the ResNet50 backbone.
    - Uses a linear layer to map the extracted visual features to a target hidden dim (e.g. 4096 to match the language model).
    - Uses a PerceiverResampler to compress the variable-length patch sequence into a fixed number of visual tokens (e.g. 64 tokens).
    - Outputs a tensor of shape (B, T, num_latents, target_hidden_dim), aligned with EndoFMAdapter output for fusion.
    """
    def __init__(self, pmc_checkpoint_path: str,
                 target_hidden_dim: int = 4096, num_latents: int = 64,
                 perceiver_depth: int = 2, perceiver_heads: int = 8, perceiver_dim_head: int = 64,
                 freeze_pmc: bool = True):
        super(PMCClipAdapter, self).__init__()
        # Initialize PMC-CLIP ResNet50 backbone (excluding classification/projection head)
        # Use official ModifiedResNet from PMC-CLIP (ResNet50: layers [3,4,6,3], width=64)
        embed_dim = 64 * 32  # 2048, ResNet50 conv feature dimension
        heads = embed_dim // 64  # e.g. 32 heads for 2048-d embed_dim
        self.pmc_model = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=embed_dim,
                                        heads=heads, width=64)
        # Load PMC-CLIP pre-trained weights
        ckpt = torch.load(pmc_checkpoint_path, map_location='cpu')
        # Extract vision branch weights (keys starting with "visual.") and skip attnpool/projection layers
        filtered_state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("attnpool"):
                continue  # 跳过注意力汇聚层（attnpool）和投影层（c_proj）
            filtered_state_dict[k] = v

        missing = self.pmc_model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded PMC-CLIP checkpoint. Missing keys: {missing.missing_keys}, "
              f"Unexpected keys: {missing.unexpected_keys}")
        # Freeze PMC-CLIP backbone parameters if specified
        if freeze_pmc:
            for param in self.pmc_model.parameters():
                param.requires_grad = False
        # ResNet output channel dimension (ResNet50 last conv layer outputs 2048 channels)
        self.pmc_output_dim = 2048
        # Linear projection to target hidden dimension (e.g. 4096)
        self.linear_proj = nn.Linear(self.pmc_output_dim, target_hidden_dim)
        # Perceiver Resampler: compress variable-length patch sequence to fixed num_latents tokens
        self.perceiver_resampler = PerceiverResampler(dim=target_hidden_dim,
                                                     depth=perceiver_depth,
                                                     heads=perceiver_heads,
                                                     dim_head=perceiver_dim_head,
                                                     num_latents=num_latents)
        # Record output dimension
        self.output_dim = target_hidden_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: Image tensor of shape (B, T, C, H, W) where B = batch size,
                       T = number of images per sample (set T=1 if no temporal dimension),
                       C = 3 (RGB channels), H, W = image height and width.
        :return: Image feature token sequence of shape (B, T, num_latents, target_hidden_dim)
        """
        B, T, C, H, W = images.shape
        # Combine batch and T dims for backbone forward: (B*T, C, H, W)
        images_flat = images.reshape(B * T, C, H, W)
        # Forward through ResNet50 backbone (conv layers only, exclude attnpool)
        x = images_flat.type(self.pmc_model.conv1.weight.dtype)
        # Stem: 3 conv layers with ReLU and AvgPool (anti-aliased downsampling)
        x = self.pmc_model.relu1(self.pmc_model.bn1(self.pmc_model.conv1(x)))
        x = self.pmc_model.relu2(self.pmc_model.bn2(self.pmc_model.conv2(x)))
        x = self.pmc_model.relu3(self.pmc_model.bn3(self.pmc_model.conv3(x)))
        x = self.pmc_model.avgpool(x)
        # ResNet layers 1-4
        x = self.pmc_model.layer1(x)
        x = self.pmc_model.layer2(x)
        x = self.pmc_model.layer3(x)
        x = self.pmc_model.layer4(x)
        conv_features = x  # shape: (B*T, 2048, H_out, W_out)
        # Flatten spatial dimensions: (B*T, 2048, H_out*W_out) -> (B*T, patch_count, 2048)
        patch_tokens = conv_features.flatten(2).transpose(1, 2)
        # Restore batch and T dimensions: (B, T, patch_count, 2048)
        patch_tokens = patch_tokens.view(B, T, -1, self.pmc_output_dim)
        # Linear projection of features to target_hidden_dim: (B, T, patch_count, target_hidden_dim)
        mapped_tokens = self.linear_proj(patch_tokens)
        # Perceiver Resampler compresses each image's patch sequence to fixed num_latents: (B, T, num_latents, target_hidden_dim)
        image_embeds = self.perceiver_resampler(mapped_tokens)
        return image_embeds

# class DualVisualAdapter(nn.Module):
#     """
#     DualVisualAdapter 模块融合两个视觉分支 (内窥镜视频Endo-FM和PMC-CLIP图像)：
#     - 包含 EndoFMAdapter 分支用于处理视频帧序列;
#     - 包含 PMCClipAdapter 分支用于处理静态图像;
#     - 支持通过 enable_endo 和 enable_pmc 参数选择启用哪个分支（可单独或同时启用）;
#     - 支持 add_branch_tokens 参数，在每个分支输出的视觉token序列前添加一个可学习的源标识token;
#     - 当两路分支都启用时，在 token 维度将它们的输出序列进行拼接。
#     输出张量维度为 (B, T, N, target_hidden_dim)，其中 N 的取值取决于启用的分支数量以及是否添加源标识token：
#     若仅单一路径且不添加标识token，则 N=64；单一路径且添加token则 N=65；
#     若双路径且未添加token，则 N=128；双路径且添加token则 N=130。
#     """
#
#     def __init__(self, cfg, endo_checkpoint_path, pmc_checkpoint_path,
#                  target_hidden_dim=4096, num_latents=64,
#                  perceiver_depth=2, perceiver_heads=8, perceiver_dim_head=64,
#                  freeze_endo=True, freeze_pmc=True,
#                  enable_endo=True, enable_pmc=True,
#                  num_queries=64, qformer_depth=2):
#
#         super(DualVisualAdapter, self).__init__()
#
#         # 保存启用状态
#         self.enable_endo = enable_endo
#         self.enable_pmc = enable_pmc
#
#         # 初始化 Endo-FM 适配器分支
#         if enable_endo:
#             self.endo_adapter = EndoFMAdapter(cfg, endo_checkpoint_path,
#                                               target_hidden_dim, num_latents,
#                                               perceiver_depth, perceiver_heads,
#                                               perceiver_dim_head, freeze_endo)
#             self.endo_qformer = VisionQFormer(target_hidden_dim, num_queries, qformer_depth)
#             self.endo_token = nn.Parameter(torch.randn(1, 1, target_hidden_dim))  # learnable Endo source token
#
#         # 初始化 PMC-CLIP 适配器分支
#         if enable_pmc:
#             self.pmc_adapter = PMCClipAdapter(pmc_checkpoint_path,
#                                               target_hidden_dim, num_latents,
#                                               perceiver_depth, perceiver_heads,
#                                               perceiver_dim_head, freeze_pmc)
#             self.pmc_qformer = VisionQFormer(target_hidden_dim, num_queries, qformer_depth)
#             self.pmc_token = nn.Parameter(torch.randn(1, 1, target_hidden_dim))  # learnable PMC source token
#
#         # 使用Perceiver resampler得到固定长度的token
#         self.resampler = PerceiverResampler(dim=target_hidden_dim,
#                                             num_latents=num_latents,
#                                             depth=perceiver_depth,
#                                             heads=perceiver_heads,
#                                             dim_head=perceiver_dim_head)
#
#         self.output_dim = target_hidden_dim
#
#     def forward(self, vision_x):
#         B, T, C, H, W = vision_x.shape
#         endo_tokens, pmc_tokens = None, None
#
#         # Processing PMC-CLIP visual branch
#         if self.enable_pmc:
#             pmc_feats = self.pmc_adapter(vision_x)
#             pmc_tokens = self.pmc_qformer(pmc_feats)
#             pmc_tokens = self.resampler(pmc_tokens)
#             pmc_tokens = torch.cat([self.pmc_token.expand(B, T, -1, -1), pmc_tokens], dim=2)  # add PMC source token
#
#         # Processing Endo-FM visual branch
#         if self.enable_endo:
#             endo_feats = self.endo_adapter(vision_x)
#             endo_tokens = self.endo_qformer(endo_feats)
#             endo_tokens = self.resampler(endo_tokens)
#             endo_tokens = torch.cat([self.endo_token.expand(B, T, -1, -1), endo_tokens], dim=2)  # add Endo source token
#
#         # 融合分支后的输出
#         fused_tokens = torch.cat([
#             token for token in [pmc_tokens, endo_tokens] if token is not None
#         ], dim=2)
#
#         return {
#             "fused_tokens": fused_tokens,  # (B, T, N, D), combined visual tokens
#             "pmc_tokens": pmc_tokens,  # (B, T, N, D), PMC visual tokens
#             "endo_tokens": endo_tokens  # (B, T, N, D), Endo visual tokens
#         }
class DualVisualAdapter(nn.Module):
    def __init__(self, cfg, endo_checkpoint_path, pmc_checkpoint_path,
                 target_hidden_dim=4096, num_latents=64,
                 perceiver_depth=2, perceiver_heads=8, perceiver_dim_head=64,
                 freeze_endo=True, freeze_pmc=True,
                 enable_endo=True, enable_pmc=True, add_branch_tokens=True,
                 num_queries=64, qformer_depth=2):

        super(DualVisualAdapter, self).__init__()
        # existing adapters
        if enable_endo:
            self.endo_adapter = EndoFMAdapter(cfg, endo_checkpoint_path,
                                              target_hidden_dim, num_latents, perceiver_depth,
                                              perceiver_heads, perceiver_dim_head, freeze_endo)
        if enable_pmc:
            self.pmc_adapter = PMCClipAdapter(pmc_checkpoint_path,
                                              target_hidden_dim, num_latents, perceiver_depth,
                                              perceiver_heads, perceiver_dim_head, freeze_pmc)

        # VisionQFormers
        self.enable_endo = enable_endo
        self.enable_pmc = enable_pmc

        if enable_endo:
            self.endo_qformer = VisionQFormer(dim=target_hidden_dim, num_queries=num_queries, depth=qformer_depth)
        if enable_pmc:
            self.pmc_qformer = VisionQFormer(dim=target_hidden_dim, num_queries=num_queries, depth=qformer_depth)

        self.add_branch_tokens = add_branch_tokens
        if add_branch_tokens:
            if enable_endo:
                self.endo_token = nn.Parameter(torch.zeros(1, 1, target_hidden_dim))
                nn.init.normal_(self.endo_token, std=0.02)
            if enable_pmc:
                self.pmc_token = nn.Parameter(torch.zeros(1, 1, target_hidden_dim))
                nn.init.normal_(self.pmc_token, std=0.02)
        self.output_dim = target_hidden_dim

    def forward(self, vision_x):
        B, T, C, H, W = vision_x.shape
        endo_tokens, pmc_tokens = None, None

        if self.enable_endo:
            endo_feats = self.endo_adapter(vision_x)  # shape (B, T, num_latents, dim)
            endo_tokens = self.endo_qformer(endo_feats)  # (B, T, num_queries, dim)

        if self.enable_pmc and T == 1:
            pmc_feats = self.pmc_adapter(vision_x)  # shape (B, T, num_latents, dim)
            pmc_tokens = self.pmc_qformer(pmc_feats)  # (B, T, num_queries, dim)

        if self.add_branch_tokens:
            if endo_tokens is not None:
                endo_tokens = torch.cat([
                    self.endo_token.expand(B, T, 1, -1), endo_tokens
                ], dim=2)  # shape (B, T, num_queries+1, dim)
            if pmc_tokens is not None:
                pmc_tokens = torch.cat([
                    self.pmc_token.expand(B, T, 1, -1), pmc_tokens
                ], dim=2)  # shape (B, T, num_queries+1, dim)

        if endo_tokens is not None and pmc_tokens is not None:
            fused_tokens = torch.cat([endo_tokens, pmc_tokens], dim=2)  # shape (B, T, 2*(num_queries+1), dim)
        elif endo_tokens is not None:
            fused_tokens = endo_tokens
        elif pmc_tokens is not None:
            fused_tokens = pmc_tokens
        else:
            raise ValueError("Both endo_tokens and pmc_tokens are None. Check your enable flags.")

        # 返回明确的字典结构
        return {
            "fused_tokens": fused_tokens,
            "pmc_tokens": pmc_tokens,
            "endo_tokens": endo_tokens
        }


# 语义蒸馏损失函数
def semantic_distill_loss(pmc_tokens, endo_tokens):
    assert pmc_tokens is not None and endo_tokens is not None, "PMC and Endo tokens must not be None"
    return nn.functional.mse_loss(endo_tokens, pmc_tokens)