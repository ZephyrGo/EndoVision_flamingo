import torch
import torch.nn as nn
from flamingo_core.pmc_clip_adapter import PMCClipAdapter
from flamingo_core.endo_adapter import EndoFMAdapter

class DualVisualAdapter(nn.Module):
    def __init__(self,
                 cfg,
                 endo_checkpoint_path: str,
                 pmc_checkpoint_path: str,
                 target_hidden_dim: int = 4096,
                 num_latents: int = 64,
                 perceiver_depth: int = 2,
                 perceiver_heads: int = 8,
                 perceiver_dim_head: int = 64,
                 freeze_endo: bool = True,
                 freeze_pmc: bool = True,
                 enable_endo: bool = False,
                 enable_pmc: bool = True,
                 add_branch_tokens: bool = True):
        super(DualVisualAdapter, self).__init__()

        self.enable_endo = enable_endo
        self.enable_pmc = enable_pmc

        if enable_endo:
            self.endo_adapter = EndoFMAdapter(
                cfg=cfg,
                endo_checkpoint_path=endo_checkpoint_path,
                target_hidden_dim=target_hidden_dim,
                num_latents=num_latents,
                perceiver_depth=perceiver_depth,
                perceiver_heads=perceiver_heads,
                perceiver_dim_head=perceiver_dim_head,
                freeze_endo=freeze_endo
            )

        if enable_pmc:
            self.pmc_adapter = PMCClipAdapter(
                pmc_checkpoint_path=pmc_checkpoint_path,
                target_hidden_dim=target_hidden_dim,
                num_latents=num_latents,
                perceiver_depth=perceiver_depth,
                perceiver_heads=perceiver_heads,
                perceiver_dim_head=perceiver_dim_head,
                freeze_pmc=freeze_pmc
            )

        self.add_branch_tokens = add_branch_tokens

        if add_branch_tokens:
            if enable_endo:
                self.endo_token = nn.Parameter(torch.zeros(1, 1, target_hidden_dim))
                nn.init.normal_(self.endo_token, std=0.02)
            if enable_pmc:
                self.pmc_token = nn.Parameter(torch.zeros(1, 1, target_hidden_dim))
                nn.init.normal_(self.pmc_token, std=0.02)

        self.output_dim = target_hidden_dim

    def forward(self, vision_x: torch.Tensor) -> torch.Tensor:
        if vision_x is None:
            raise ValueError("vision_x cannot be None.")

        if vision_x.ndim == 4:
            vision_x = vision_x.unsqueeze(1)  # (B, T=1, C, H, W)

        B, T, C, H, W = vision_x.shape

        endo_tokens, pmc_tokens = None, None

        if self.enable_endo:
            endo_tokens = self.endo_adapter(vision_x)

        if self.enable_pmc and T == 1:
            pmc_tokens = self.pmc_adapter(vision_x)

        if self.add_branch_tokens:
            if endo_tokens is not None:
                endo_token_exp = self.endo_token.expand(B, T, 1, self.output_dim)
                endo_tokens = torch.cat([endo_token_exp, endo_tokens], dim=2)

            if pmc_tokens is not None:
                pmc_token_exp = self.pmc_token.expand(B, T, 1, self.output_dim)
                pmc_tokens = torch.cat([pmc_token_exp, pmc_tokens], dim=2)

        if endo_tokens is not None and pmc_tokens is not None:
            fused_tokens = torch.cat([endo_tokens, pmc_tokens], dim=2)
        elif endo_tokens is not None:
            fused_tokens = endo_tokens
        elif pmc_tokens is not None:
            fused_tokens = pmc_tokens
        else:
            raise ValueError("Both endo_tokens and pmc_tokens are None. Check your enable flags.")

        return fused_tokens


# class DualVisualAdapter(nn.Module):
#     """
#     DualVisualAdapter 模块融合 PMC-CLIP 图像分支和 Endo-FM 视频分支:
#     - 分别加载 PMC-CLIP 图像编码器和 Endo-FM 视频编码器, 提取视觉特征;
#     - 为每个分支应用独立的 PerceiverResampler 将特征序列压缩成固定数量的 latent tokens (默认64个);
#     - forward 方法根据 enable_pmc / enable_endo 控制选择使用哪条分支，并可选在每条分支 token 序列前添加一个可学习的分支标识 token;
#     - 输出 token 序列按 [PMC tokens, Endo tokens] 顺序在 token 维度拼接，形状与 EndoFMAdapter 输出保持一致 (B, T, N, target_hidden_dim)&#8203;:contentReference[oaicite:2]{index=2}。
#     """
#     def __init__(self, endo_cfg, endo_checkpoint_path: str, pmc_checkpoint_path: str,
#                  target_hidden_dim: int = 4096, num_latents: int = 64,
#                  perceiver_depth: int = 2, perceiver_heads: int = 8, perceiver_dim_head: int = 64,
#                  freeze_endo: bool = True, freeze_pmc: bool = True):
#         super(DualVisualAdapter, self).__init__()
#         # Initialize Endo-FM video branch adapter
#         self.endo_adapter = EndoFMAdapter(cfg=endo_cfg, endo_checkpoint_path=endo_checkpoint_path,
#                                           target_hidden_dim=target_hidden_dim, num_latents=num_latents,
#                                           perceiver_depth=perceiver_depth, perceiver_heads=perceiver_heads,
#                                           perceiver_dim_head=perceiver_dim_head, freeze_endo=freeze_endo)
#         # Initialize PMC-CLIP image branch adapter
#         self.pmc_adapter = PMCClipAdapter(pmc_checkpoint_path=pmc_checkpoint_path,
#                                           target_hidden_dim=target_hidden_dim, num_latents=num_latents,
#                                           perceiver_depth=perceiver_depth, perceiver_heads=perceiver_heads,
#                                           perceiver_dim_head=perceiver_dim_head, freeze_pmc=freeze_pmc)
#         # Learnable source identifier tokens for each branch (1 x 1 x 1 x hidden_dim)
#         self.pmc_branch_token = nn.Parameter(torch.zeros(1, 1, 1, target_hidden_dim))
#         self.endo_branch_token = nn.Parameter(torch.zeros(1, 1, 1, target_hidden_dim))
#         # Output token dimension (should match language model hidden size, e.g., 4096)
#         self.output_dim = target_hidden_dim
#
#     def forward(self, video_frames: torch.Tensor, images: torch.Tensor,
#                 enable_endo: bool = True, enable_pmc: bool = True,
#                 add_branch_tokens: bool = False) -> torch.Tensor:
#         """
#         :param video_frames: 视频帧序列 (B, T, C, H, W)，传入 Endo-FM 分支进行处理
#         :param images: 图像序列 (B, T, C, H, W)，传入 PMC-CLIP 分支进行处理 (可与视频帧相同或不同)
#         :param enable_endo: 是否启用 Endo-FM 视频分支 (False 时跳过该分支)
#         :param enable_pmc: 是否启用 PMC-CLIP 图像分支 (False 时跳过该分支)
#         :param add_branch_tokens: 是否在每个分支输出前添加可学习的来源标识 token
#         :return: 融合后的视觉 token 序列 (B, T, N, target_hidden_dim)，其中 N 取决于启用的分支数和是否添加标识 token (可为 64, 65, 128, 或 130)
#         """
#         B, T = video_frames.shape[0], video_frames.shape[1]  # batch size and sequence length
#         outputs = []
#
#         # PMC-CLIP image branch
#         if enable_pmc:
#             # 获得 PMC-CLIP 分支的视觉 token (B, T, 64, 4096)
#             pmc_tokens = self.pmc_adapter(images)  # shape: (B, T, num_latents, target_hidden_dim)
#             if add_branch_tokens:
#                 # 插入 PMC 分支标识 token (在 token 序列前，形状: B x T x 1 x hidden_dim)
#                 pmc_token_prefix = self.pmc_branch_token.expand(B, T, -1, -1)
#                 pmc_tokens = torch.cat([pmc_token_prefix, pmc_tokens], dim=2)
#             outputs.append(pmc_tokens)
#
#         # Endo-FM video branch
#         if enable_endo:
#             # 获得 Endo-FM 分支的视觉 token (B, T, 64, 4096)
#             endo_tokens = self.endo_adapter(video_frames)  # shape: (B, T, num_latents, target_hidden_dim)
#             if add_branch_tokens:
#                 # 插入 Endo 分支标识 token (形状: B x T x 1 x hidden_dim)
#                 endo_token_prefix = self.endo_branch_token.expand(B, T, -1, -1)
#                 endo_tokens = torch.cat([endo_token_prefix, endo_tokens], dim=2)
#             outputs.append(endo_tokens)
#
#         # 至少应启用一个分支，否则无法生成有效输出
#         if len(outputs) == 0:
#             raise ValueError("DualVisualAdapter: At least one of enable_endo or enable_pmc must be True.")
#
#         # 按顺序拼接 [PMC tokens, Endo tokens] (在 token 维度上拼接)
#         if len(outputs) == 2:
#             # 确保 PMC 分支输出位于前 (输出列表按上述顺序添加)
#             fused_tokens = torch.cat(outputs, dim=2)
#         else:
#             # 只有一个分支启用，直接使用该分支输出
#             fused_tokens = outputs[0]
#
#         return fused_tokens
