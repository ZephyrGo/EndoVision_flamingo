import torch
import torch.nn as nn
from flamingo_core.helpers import PerceiverResampler
# Import PMC-CLIP vision backbone classes
from pmc_clip_backbone import ModifiedResNet, AttentionPool2d

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
                                        heads=heads, input_resolution=224, width=64)
        # Load PMC-CLIP pre-trained weights
        ckpt = torch.load(pmc_checkpoint_path, map_location='cpu')
        # Extract vision branch weights (keys starting with "visual.") and skip attnpool/projection layers
        state_dict = {}
        for k, v in ckpt.items():
            # Remove DataParallel prefix if present
            if k.startswith("module.visual."):
                key = k[len("module.visual."):]
            elif k.startswith("visual."):
                key = k[len("visual."):]
            else:
                continue
            # Skip AttentionPool2d and projection (text/mlm projection, logit_scale) weights
            if key.startswith("attnpool") or key.startswith("text_projection") \
               or key.startswith("mlm_projection") or key.startswith("logit_scale"):
                continue
            state_dict[key] = v
        # Load weights into the model (ignore missing keys for skipped layers)
        missing = self.pmc_model.load_state_dict(state_dict, strict=False)
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
        images_flat = images.view(B * T, C, H, W)
        # Forward through ResNet50 backbone (conv layers only, exclude attnpool)
        x = images_flat.type(self.pmc_model.conv1.weight.dtype)
        # Stem: 3 conv layers with ReLU and AvgPool (anti-aliased downsampling)
        for conv, bn in [(self.pmc_model.conv1, self.pmc_model.bn1),
                         (self.pmc_model.conv2, self.pmc_model.bn2),
                         (self.pmc_model.conv3, self.pmc_model.bn3)]:
            x = self.pmc_model.relu(bn(conv(x)))
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

#
# # 定义 Modified ResNet50 主干网络所需的基本模块 (瓶颈块)
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes: int, planes: int, stride: int = 1):
#         super(Bottleneck, self).__init__()
#         # 1x1 降维卷积
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         # 3x3 卷积 (保持通道数)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         # 使用平均池化实现 Blur Pooling（当 stride > 1 时进行下采样）
#         self.avgpool = nn.AvgPool2d(kernel_size=stride) if stride > 1 else nn.Identity()
#         # 1x1 卷积 (升维卷积，输出通道 = planes * 4)
#         self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         # 下采样层 (当输入输出维度或步幅不一致时使用)
#         self.downsample = None
#         if stride > 1 or inplanes != planes * Bottleneck.expansion:
#             # 下采样顺序: 先平均池化 (anti-alias), 再 1x1 卷积调整通道数, 最后批归一化
#             self.downsample = nn.Sequential(
#                 nn.AvgPool2d(kernel_size=stride),
#                 nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=1, bias=False),
#                 nn.BatchNorm2d(planes * Bottleneck.expansion)
#             )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x  # 残差连接输入
#         # 主分支的3层卷积操作
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.avgpool(out)  # 如果 stride > 1，这一步执行下采样（Blur Pool）
#         out = self.bn3(self.conv3(out))
#         # 残差连接: 若需要则下采样
#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)
#         return out
#
#
# # 定义 ModifiedResNet 类，实现 PMC-CLIP 的 ResNet50 主干（不含Attention Pool和投影层）
# class ModifiedResNet(nn.Module):
#     """
#     ModifiedResNet 实现了 ResNet50 主干的修改版本:
#     - 使用3层卷积(3x3)构成stem替代原始ResNet的单层7x7卷积，且使用AvgPool代替MaxPool进行下采样 (抗混叠)。
#     - 在残差块的下采样中引入Blur Pool（平均池化在stride卷积前）。
#     - 移除最终的全局池化和分类/投影层，仅保留卷积特征图输出。
#     """
#
#     def __init__(self, layers: list, width: int = 64):
#         """
#         :param layers: 列表，指定每个层（layer1-4）包含的Bottleneck块数量，例如 [3,4,6,3] 对应 ResNet50。
#         :param width: 第一层卷积的输出通道基数 (默认64，对于ResNet50一般为64)。
#         """
#         super(ModifiedResNet, self).__init__()
#         # Stem部分：3个卷积层 (3x3)，通道逐步从3增至 `width`
#         self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(width // 2)
#         self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(width // 2)
#         self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(width)
#         self.relu = nn.ReLU(inplace=True)
#         # 使用AvgPool进行下采样（代替原ResNet的max pool）
#         self.avgpool = nn.AvgPool2d(kernel_size=2)
#         # 设置初始输入通道数 (后续逐层更新)
#         self._inplanes = width
#         # 构建ResNet的4个阶段(layer1-layer4)
#         self.layer1 = self._make_layer(width, layers[0], stride=1)  # 第一层，不减小尺寸
#         self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  # 第二层，输出通道翻倍，尺寸1/2
#         self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  # 第三层，输出通道翻倍，尺寸1/2
#         self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  # 第四层，输出通道翻倍，尺寸1/2 (总下采样32x)
#
#     def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
#         """
#         构建包含多个 Bottleneck 的残差层。
#         :param planes: 当前层第一个 Bottleneck 块卷积的基准通道数 (不乘expansion)。
#         :param blocks: 当前层包含的 Bottleneck 块数。
#         :param stride: 当前层第一个 Bottleneck 块的步幅 (用于下采样)。
#         """
#         layers = []
#         # 第一个Bottleneck块，可能需要下采样
#         layers.append(Bottleneck(self._inplanes, planes, stride=stride))
#         # 更新当前输出通道作为下一块的输入通道
#         self._inplanes = planes * Bottleneck.expansion
#         # 剩余的Bottleneck块，stride默认为1
#         for _ in range(1, blocks):
#             layers.append(Bottleneck(self._inplanes, planes, stride=1))
#             self._inplanes = planes * Bottleneck.expansion
#         return nn.Sequential(*layers)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Stem：3层卷积 + ReLU + BN + AvgPool
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.relu(self.bn3(self.conv3(x)))
#         x = self.avgpool(x)
#         # ResNet 层叠
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # 返回最终的卷积特征 (未经过任何池化或投影)
#         return x
#
#
# class PMCClipAdapter(nn.Module):
#     """
#     PMCClipAdapter 封装 PMC-CLIP 预训练的 ResNet50 图像编码器:
#     - 加载 PMC-CLIP 图像模型的本地权重（state_dict），移除分类/投影头，仅保留ResNet50视觉主干;
#     - 使用线性层将提取的视觉特征映射到指定的 target_hidden_dim（默认4096，与语言模型隐藏尺寸对齐）;
#     - 使用 PerceiverResampler 将变长的图像patch序列压缩为固定长度的视觉token（默认64个）;
#     - 输出张量形状为 (B, T, num_latents, target_hidden_dim)，与 EndoFMAdapter 输出对齐以便后续融合。
#     """
#
#     def __init__(self, pmc_checkpoint_path: str,
#                  target_hidden_dim: int = 4096, num_latents: int = 64,
#                  perceiver_depth: int = 2, perceiver_heads: int = 8, perceiver_dim_head: int = 64,
#                  freeze_pmc: bool = True):
#         super(PMCClipAdapter, self).__init__()
#         # 初始化 PMC-CLIP ResNet50 主干模型 (不含分类头)
#         self.pmc_model = ModifiedResNet(layers=[3, 4, 6, 3], width=64)  # ResNet50 架构
#
#         # 加载 PMC-CLIP 预训练权重
#         ckpt = torch.load(pmc_checkpoint_path, map_location='cpu')
#         # 提取视觉分支相关权重 (key 以 "visual." 开头)，并去除不需要的部分 (例如 attention pooling 层和投影层)
#         state_dict = {}
#         for k, v in ckpt.items():
#             # 兼容 DataParallel 保存格式，去除前缀 "module."
#             if k.startswith("module.visual."):
#                 key = k[len("module.visual."):]
#             elif k.startswith("visual."):
#                 key = k[len("visual."):]
#             else:
#                 continue
#             # 跳过 AttentionPool2d 层和后续的文本投影等权重
#             if key.startswith("attnpool") or key.startswith("text_projection") or key.startswith(
#                     "mlm_projection") or key.startswith("logit_scale"):
#                 continue
#             state_dict[key] = v
#         # 加载权重到模型 (忽略不匹配的权重，例如不存在的分类头)
#         missing = self.pmc_model.load_state_dict(state_dict, strict=False)
#         print(
#             f"Loaded PMC-CLIP checkpoint. Missing keys: {missing.missing_keys}, Unexpected keys: {missing.unexpected_keys}")
#
#         # 冻结 PMC-CLIP 主干参数（如果不希望在训练中更新图像编码器）
#         if freeze_pmc:
#             for param in self.pmc_model.parameters():
#                 param.requires_grad = False
#
#         # 获取ResNet输出通道维度 (ResNet50 最后一层输出 2048 通道)
#         self.pmc_output_dim = 2048
#         # 特征维度映射到目标隐藏维度 (如4096)
#         self.linear_proj = nn.Linear(self.pmc_output_dim, target_hidden_dim)  # 将 ResNet 提取的特征投射到 target_hidden_dim
#
#         # Perceiver Resampler：将可变长度patch序列压缩为固定数量(num_latents)的视觉tokens
#         self.perceiver_resampler = PerceiverResampler(dim=target_hidden_dim,
#                                                       depth=perceiver_depth,
#                                                       heads=perceiver_heads,
#                                                       dim_head=perceiver_dim_head,
#                                                       num_latents=num_latents)
#         # 输出维度记录
#         self.output_dim = target_hidden_dim
#
#     def forward(self, images: torch.Tensor) -> torch.Tensor:
#         """
#         :param images: 图像张量 (B, T, C, H, W)，B为batch大小，T为每组输入的图像数，C通道数(3)，H,W为图像高宽。
#                        如果没有时间序列概念，可以将 T 设为1。
#         :return: 图像特征token序列 (B, T, num_latents, target_hidden_dim)
#         """
#         B, T, C, H, W = images.shape
#         # 将 (B, T) 视作整体批次，展平成 (B*T, C, H, W)，以便送入 ResNet 提取特征
#         images_flat = images.view(B * T, C, H, W)
#         # 提取 ResNet50 最后一层卷积特征 (输出 shape: (B*T, 2048, H_out, W_out))
#         conv_features = self.pmc_model(images_flat)  # ResNet主干输出特征图
#         # 展平空间维度，将每个图像的卷积特征映射为 patch 序列
#         # (B*T, 2048, H_out, W_out) -> (B*T, patch_count, 2048)
#         patch_tokens = conv_features.flatten(2).transpose(1, 2)
#         # 恢复 batch 和 T 维度: (B, T, patch_count, 2048)
#         patch_tokens = patch_tokens.view(B, T, -1, self.pmc_output_dim)
#         # 线性映射特征维度到 target_hidden_dim: (B, T, patch_count, target_hidden_dim)
#         mapped_tokens = self.linear_proj(patch_tokens)
#         # Perceiver Resampler 压缩每个图像的patch序列到固定数量的latent tokens: (B, T, num_latents, target_hidden_dim)
#         image_embeds = self.perceiver_resampler(mapped_tokens)
#         return image_embeds
