import torch
import torch.nn as nn
from flamingo_core.helpers import PerceiverResampler
from endo_fm_backbone.timesformer import get_vit_base_patch16_224
from endo_fm_backbone.parser import load_config


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
        :param video_frames: (B, T, C, H, W):
            B: batch大小
            T: 视频帧数
            C: 通道数（通常为3，RGB图像）
            H, W: 图像的高度与宽度
        :return:
            (B, T, num_latents, target_hidden_dim)
        """
        B, T, C, H, W = video_frames.shape

        # 调用Endo-FM模型提取视频时空特征，处理可能的维度排列问题（C与T位置互换）
        try:
            features = self.endo_model.forward_features(video_frames, get_all=True)

        except RuntimeError:
            # features = self.endo_model(video_frames.permute(0, 2, 1, 3, 4), get_all=True)
            features = self.endo_model.forward_features(video_frames.permute(0, 2, 1, 3, 4), get_all=True)

        if features.dim() == 4:
            patch_tokens = features[:, :, 1:, :]  # 若输出特征为4维，去掉CLS token
        else:
            seq_len = features.shape[1]
            if T > 0 and seq_len % T != 0:
                patch_tokens = features[:, 1:, :]
                seq_len = patch_tokens.shape[1]
            else:
                patch_tokens = features
            N_patch = seq_len // T if T > 0 else seq_len
            patch_tokens = patch_tokens.view(B, T, N_patch, self.endo_output_dim)  # 若为3维，则去掉CLS token后reshape成(B, T, N_patch, 特征维度)

        mapped_tokens = self.linear_proj(patch_tokens)  # 特征维度由Endo-FM输出维度映射到目标维度
        image_embeds = self.perceiver_resampler(mapped_tokens)  # 压缩每个视频序列到固定数量的视觉token
        return image_embeds


# import torch
# import torch.nn as nn
# from helpers import PerceiverResampler
# from endo_fm_backbone.parser import load_config
#
# class EndoFMAdapter(nn.Module):
#     """
#     EndoFMAdapter 模块封装了对 Endo-FM 预训练内窥镜视频Transformer 的调用，并通过线性映射和 Perceiver Resampler
#     将可变长度的视频帧特征压缩为固定数量的视觉 token。
#
#     功能:
#     - 加载 Endo-FM 模型 (预训练的视频Transformer，用于提取胃镜视频的空间-时间特征)；
#     - 从输入的视频帧序列提取时空特征；
#     - 将特征通过线性层映射到目标隐藏维度（例如与语言模型的hidden size对齐）；
#     - 使用 Perceiver Resampler 模块将不定长的视觉特征序列压缩成固定数量的视觉tokens (image_embeds)，以便融入Flamingo多模态架构。
#     """
#
#     def __init__(self, endo_model=None, endo_checkpoint_path: str = None,
#                  target_hidden_dim: int = 4096, num_latents: int = 64,
#                  perceiver_depth: int = 2, perceiver_heads: int = 8, perceiver_dim_head: int = 64,
#                  freeze_endo: bool = True,
#                  backbone_cfg=None):  # 新增参数
#
#         """
#         初始化 EndoFMAdapter。
#         参数:
#         - endo_model: 可以直接传入一个已实例化的 Endo-FM 模型 (nn.Module)。若为 None，则需要提供 endo_checkpoint_path 来加载模型权重。
#         - endo_checkpoint_path: Endo-FM 预训练权重文件路径。如果提供了 endo_model，此参数可为 None。
#         - target_hidden_dim: 映射后的目标隐藏维度大小（应与语言模型的hidden size一致，例如 LLaMA-7B 为4096）。
#         - num_latents: Perceiver Resampler 生成的视觉token数量。
#         - perceiver_depth: Perceiver Resampler中 Transformer 层的层数(depth)。
#         - perceiver_heads: Perceiver Resampler中 multi-head attention 的头数。
#         - perceiver_dim_head: Perceiver Resampler中每个注意力头的维度。
#         - freeze_endo: 是否冻结 Endo-FM 模型的参数（默认 True，即在训练多模态模型时不更新视觉编码器参数）。
#         """
#         super(EndoFMAdapter, self).__init__()
#
#         # 若直接通过factory传入 config
#         if endo_model is None and backbone_cfg is not None:
#             from endo_fm_backbone.timesformer import get_vit_base_patch16_224
#             endo_model = get_vit_base_patch16_224(cfg=backbone_cfg, no_head=True)
#
#         # 加载或设置 Endo-FM 视频Transformer模型
#         if endo_model is not None:
#             self.endo_model = endo_model
#         else:
#             # 从checkpoint加载模型权重（假定 Endo-FM 模型架构可从代码导入）
#             # 这里假设存在类似 get_vit_base_patch16_224 的构造函数，根据需要调整
#             from endo_fm_backbone.timesformer import get_vit_base_patch16_224  # 假设Endo-FM提供了模型构造函数
#             # 创建Endo-FM骨干模型，无分类头
#             self.endo_model = get_vit_base_patch16_224(no_head=True)
#             # 加载预训练权重
#             ckpt = torch.load(endo_checkpoint_path, map_location='cpu')
#             # 如果权重dict中有teacher/student等结构，处理前缀
#             if "teacher" in ckpt:
#                 ckpt = ckpt["teacher"]
#             # 移除可能存在的 "backbone." 前缀，使之匹配模型参数
#             state_dict = {k[len("backbone."):]: v for k, v in ckpt.items() if k.startswith("backbone.")}
#             if len(state_dict) == 0:
#                 state_dict = ckpt
#             missing = self.endo_model.load_state_dict(state_dict, strict=False)
#             print(
#                 f"[EndoFMAdapter] Loaded Endo-FM checkpoint. Missing keys: {missing.missing_keys}, Unexpected keys: {missing.unexpected_keys}")
#         # 冻结 Endo-FM 模型参数（如果不需要fine-tune视觉编码器）
#         if freeze_endo:
#             for param in self.endo_model.parameters():
#                 param.requires_grad = False
#
#         # 获取Endo-FM输出的特征维度 (embed_dim)
#         if hasattr(self.endo_model, "embed_dim"):
#             endo_output_dim = getattr(self.endo_model, "embed_dim")
#         elif hasattr(self.endo_model, "hidden_dim"):
#             endo_output_dim = getattr(self.endo_model, "hidden_dim")
#         else:
#             # 如果模型没有明确属性，尝试从第一个参数形状推断
#             try:
#                 for name, param in self.endo_model.named_parameters():
#                     # 找到第一个权重形状 (embedding层权重形状 [embed_dim, ...])
#                     endo_output_dim = param.shape[0]
#                     break
#             except Exception:
#                 raise RuntimeError("无法确定 Endo-FM 模型的输出维度，请提供endo_output_dim参数或检查模型定义。")
#
#         # 线性映射，将 Endo-FM 输出特征维度对齐到目标hidden_dim
#         self.linear_proj = nn.Linear(endo_output_dim, target_hidden_dim)
#
#         self.output_dim = target_hidden_dim  # 将 hidden_dim 设置为语言模型的隐藏维度
#
#         # Perceiver Resampler：将长序列的视觉特征压缩为固定数量(num_latents)的latent tokens
#         self.perceiver_resampler = PerceiverResampler(dim=target_hidden_dim, depth=perceiver_depth,
#                                                       heads=perceiver_heads, dim_head=perceiver_dim_head,
#                                                       num_latents=num_latents)
#         # 上述 PerceiverResampler 实现假设存在，如果没有定义则需要实现。
#         # PerceiverResampler 将输入形状 (B, T, N_patch, target_hidden_dim) 映射到 (B, T, num_latents, target_hidden_dim).
#         # 兼容 Flamingo 封装中 vision_encoder.visual 的访问
#         self.visual = self
#
#     def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
#         """
#         从视频帧提取视觉token。
#         输入:
#         - video_frames: 张量形状 (B, T, C, H, W)，代表 B 批次的视频，每个视频 T 帧，每帧图像通道数 C（通常3），高 H，宽 W。
#         输出:
#         - image_embeds: 张量形状 (B, T, num_latents, target_hidden_dim)，表示每个视频帧序列被压缩得到的固定数量视觉tokens，可供语言模型跨模态注意力使用。
#         """
#         B, T, C, H, W = video_frames.shape
#         # 将 video_frames 输入 Endo-FM 模型，提取时空特征
#         # 注意: Endo-FM模型可能需要输入形状 (B, C, T, H, W) 或 (B, T, C, H, W) 具体取决于实现。
#         # 这里假定 Endo-FM forward 接受 (B, C, T, H, W)（根据 Endo-FM 模型实现调整）。
#         try:
#             features = self.endo_model(
#                 video_frames)  # shape (B, seq_len, endo_output_dim) 或 (B, T, patch_count+1, endo_output_dim)
#         except RuntimeError:
#             # 如果 Endo-FM 实现需要 (B, C, T, H, W) 顺序，则调整维度后再试
#             features = self.endo_model(video_frames.permute(0, 2, 1, 3, 4))  # 变换为 (B, C, T, H, W)
#         # 检查features形状
#         if features.dim() == 4:
#             # 如果模型直接输出4维 (B, T, N_patch+cls, dim)
#             # Bf, Tf, Nf, Df = features.shape
#             # 若存在分类token，将其移除
#             # 更合理判断：如果每帧多出一个 patch，说明存在 cls token
#             patch_tokens = features[:, :, 1:, :]
#
#         else:
#             # features 为3维 (B, seq_len, dim)
#             Bf, seq_len, Df = features.shape
#             # 判断是否包含分类token: 如果 seq_len 不整除 T，说明多出一个CLS token
#             if T > 0 and seq_len % T != 0:
#                 # 有CLS token的情况: seq_len = T * patch_count + 1
#                 # 移除第一个token，然后将剩余序列reshape回 (B, T, patch_count, D)
#                 patch_tokens = features[:, 1:, :]  # 去掉CLS
#                 seq_len = patch_tokens.shape[1]
#             else:
#                 patch_tokens = features
#             # 将patch_tokens reshape为 (B, T, N_patch, Df)
#             N_patch = seq_len // T if T > 0 else seq_len  # 每帧patch数
#             patch_tokens = patch_tokens.view(Bf, T, N_patch, Df)
#         # 将Endo-FM输出特征映射到语言模型hidden维度
#         mapped_tokens = self.linear_proj(patch_tokens)  # shape (B, T, N_patch, target_hidden_dim)
#         # 使用Perceiver Resampler将每帧的patch序列压缩为固定数量的latent tokens
#         image_embeds = self.perceiver_resampler(mapped_tokens)  # 输出 shape (B, T, num_latents, target_hidden_dim)
#         return image_embeds