import torch
import torch.nn as nn
from flamingo_core.helpers import PerceiverResampler
from flamingo_core.pmc_clip_adapter import PMCClipAdapter


class DualVisualAdapter(nn.Module):
    """
    DualVisualAdapter 模块融合两个视觉分支 (内窥镜视频Endo-FM和PMC-CLIP图像)：
    - 包含 EndoFMAdapter 分支用于处理视频帧序列;
    - 包含 PMCClipAdapter 分支用于处理静态图像;
    - 支持通过 enable_endo 和 enable_pmc 参数选择启用哪个分支（可单独或同时启用）;
    - 支持 add_branch_tokens 参数，在每个分支输出的视觉token序列前添加一个可学习的源标识token;
    - 当两路分支都启用时，在 token 维度将它们的输出序列进行拼接。
    输出张量维度为 (B, T, N, target_hidden_dim)，其中 N 的取值取决于启用的分支数量以及是否添加源标识token：
    若仅单一路径且不添加标识token，则 N=64；单一路径且添加token则 N=65；
    若双路径且未添加token，则 N=128；双路径且添加token则 N=130。
    该模块输出的维度和格式与原有 EndoFMAdapter 保持一致，可直接替换 Flamingo 模型中的 vision_encoder。
    """

    def __init__(self,
                 cfg, endo_checkpoint_path: str, pmc_checkpoint_path: str,
                 target_hidden_dim: int = 4096, num_latents: int = 64,
                 perceiver_depth: int = 2, perceiver_heads: int = 8, perceiver_dim_head: int = 64,
                 freeze_endo: bool = True, freeze_pmc: bool = True,
                 enable_endo: bool = True, enable_pmc: bool = True,
                 add_branch_tokens: bool = False):
        super(DualVisualAdapter, self).__init__()
        assert enable_endo or enable_pmc, "DualVisualAdapter: 至少启用一个视觉分支 (enable_endo 或 enable_pmc 必须为 True)。"
        # 保存启用状态和标识token选项
        self.enable_endo = enable_endo
        self.enable_pmc = enable_pmc
        self.add_branch_tokens = add_branch_tokens
        # 初始化 Endo-FM 适配器 (内窥镜视频分支)
        self.endo_adapter = None
        if enable_endo:
            from flamingo_core.endo_adapter import EndoFMAdapter  # 导入现有 EndoFMAdapter 类
            self.endo_adapter = EndoFMAdapter(cfg=cfg, endo_checkpoint_path=endo_checkpoint_path,
                                              target_hidden_dim=target_hidden_dim, num_latents=num_latents,
                                              perceiver_depth=perceiver_depth, perceiver_heads=perceiver_heads,
                                              perceiver_dim_head=perceiver_dim_head, freeze_endo=freeze_endo)
        # 初始化 PMC-CLIP 适配器 (静态图像分支)
        self.pmc_adapter = None
        if enable_pmc:
            self.pmc_adapter = PMCClipAdapter(pmc_checkpoint_path=pmc_checkpoint_path,
                                              target_hidden_dim=target_hidden_dim, num_latents=num_latents,
                                              perceiver_depth=perceiver_depth, perceiver_heads=perceiver_heads,
                                              perceiver_dim_head=perceiver_dim_head, freeze_pmc=freeze_pmc)
        # 如果需要添加分支标识token，定义可学习参数
        if add_branch_tokens:
            # 每个分支一个标识token向量（1 x 1 x target_hidden_dim），将在forward时广播到 (B, T, 1, target_hidden_dim)
            self.endo_token = nn.Parameter(torch.zeros(1, 1, target_hidden_dim))
            self.pmc_token = nn.Parameter(torch.zeros(1, 1, target_hidden_dim))
            # 初始化为小随机值以打破对称性
            nn.init.normal_(self.endo_token, std=0.02)
            nn.init.normal_(self.pmc_token, std=0.02)
        # 记录输出维度 (与子适配器的 target_hidden_dim 相同)
        self.output_dim = target_hidden_dim

    def forward(self, video_frames: torch.Tensor = None, images: torch.Tensor = None) -> torch.Tensor:
        """
        :param video_frames: 视频帧序列张量 (B, T, C, H, W)，仅在 enable_endo=True 时需要提供。
        :param images: 静态图像序列张量 (B, T, C, H, W)，仅在 enable_pmc=True 时需要提供。
        :return: 融合后的视觉token序列 (B, T, N, target_hidden_dim)
        """
        # 检查输入有效性
        if self.enable_endo:
            assert video_frames is not None, "DualVisualAdapter: video_frames 输入不能为空 (启用了Endo分支)。"
        if self.enable_pmc:
            assert images is not None, "DualVisualAdapter: images 输入不能为空 (启用了PMC分支)。"

        # 获取各分支的视觉tokens输出
        endo_tokens = None
        pmc_tokens = None
        B = None  # 将用于扩展标识token的维度
        T = None
        if self.enable_endo:
            # 通过 Endo-FM 适配器提取视频帧特征 (shape: B, T, 64, target_hidden_dim)
            endo_tokens = self.endo_adapter(video_frames)  # (B, T, 64, d)
            B, T = endo_tokens.shape[0], endo_tokens.shape[1]
        if self.enable_pmc:
            # 通过 PMC-CLIP 适配器提取图像特征 (shape: B, T, 64, target_hidden_dim)
            pmc_tokens = self.pmc_adapter(images)  # (B, T, 64, d)
            # 如果此前未取到 B, T（如 Endo 未启用），在这里获取
            if B is None or T is None:
                B, T = pmc_tokens.shape[0], pmc_tokens.shape[1]

        # 在需要时添加分支标识 token
        if self.add_branch_tokens:
            if endo_tokens is not None:
                # 扩展endo_token到 (B, T, 1, d)，并在第2维(序列开头)拼接
                endo_token_expanded = self.endo_token.expand(B, T, 1, self.output_dim)
                endo_tokens = torch.cat([endo_token_expanded, endo_tokens], dim=2)  # (B, T, 65, d)
            if pmc_tokens is not None:
                # 同样处理pmc_token
                pmc_token_expanded = self.pmc_token.expand(B, T, 1, self.output_dim)
                pmc_tokens = torch.cat([pmc_token_expanded, pmc_tokens], dim=2)  # (B, T, 65, d)

        # 融合启用的分支输出
        if self.enable_endo and self.enable_pmc:
            # 双分支启用：在token维度拼接，保持(B, T)维度不变
            fused_tokens = torch.cat([endo_tokens, pmc_tokens], dim=2)  # (B, T, N_concat, d)
        elif self.enable_endo:
            fused_tokens = endo_tokens  # 仅Endo分支
        elif self.enable_pmc:
            fused_tokens = pmc_tokens  # 仅PMC分支
        else:
            # 理论上不会进入此分支，保护性返回
            fused_tokens = torch.zeros((0, 0, 0, self.output_dim),
                                       device=video_frames.device if video_frames is not None else images.device)

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
