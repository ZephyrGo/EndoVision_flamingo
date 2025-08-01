from transformers import LlamaForCausalLM, LlamaConfig
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

class GatedCrossAttentionBlock(nn.Module):
    """
    Flamingo 风格的门控跨模态注意力模块。
    包括一个跨模态注意力层（语言 -> 图像特征）和一个后续前馈层，每层输出通过可学习门控参数(tanh gating)控制。
    """

    def __init__(self, hidden_dim, num_heads, dim_head, only_attend_immediate_media=True):
        super(GatedCrossAttentionBlock, self).__init__()
        self.only_attend_immediate_media = only_attend_immediate_media
        # 规范化层
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        # 多头跨模态注意力 (queries: 文本hidden_states, keys/values: 图像embedding)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = (dim_head ** -0.5)
        # 定义投影矩阵
        self.to_q = nn.Linear(hidden_dim, num_heads * dim_head, bias=False)
        self.to_k = nn.Linear(hidden_dim, num_heads * dim_head, bias=False)
        self.to_v = nn.Linear(hidden_dim, num_heads * dim_head, bias=False)
        self.to_out = nn.Linear(num_heads * dim_head, hidden_dim, bias=False)
        # 前馈层
        inner_dim = hidden_dim * 4  # 可根据实际LLaMA设置，例如7B的中间层11008
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim)
        )
        # 门控参数 (每层各一个)，初始为0使得初始输出被tanh(0)=0抑制
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_ff = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, image_embeds: torch.Tensor,
                media_locations: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, seq_len, hidden_dim)
        # image_embeds: (B, T, L, hidden_dim) 这里T为图像序列数，L为每个图像的latent数
        # media_locations: (B, seq_len) 布尔张量，标记文本序列中哪些位置是(或跟随)图像。
        B, seq_len, D = hidden_states.shape
        # 对文本hidden_states做LayerNorm
        x = self.norm_attn(hidden_states)  # shape (B, seq_len, D)
        # 将图像latent展开作为KV
        if image_embeds is None:
            # 若无图像，直接返回原hidden_states（无变化）
            return hidden_states
        # image_embeds shape: (B, T, L, D), 合并T和L维
        B_img, T, L, D_img = image_embeds.shape
        assert B_img == B, "图像批次大小应与文本批次匹配"
        kv = image_embeds.view(B, T * L, D_img)  # (B, M, D) M = T*L
        # 计算 Q, K, V
        Q = self.to_q(x)  # (B, seq_len, num_heads*dim_head)
        K = self.to_k(kv)  # (B, M, num_heads*dim_head)
        V = self.to_v(kv)  # (B, M, num_heads*dim_head)
        # 将 Q, K, V 拆分为多个head并转置维度顺序为 (B, num_heads, seq_len, dim_head)
        Q = Q.view(B, seq_len, self.num_heads, self.dim_head).transpose(1, 2)  # (B, num_heads, seq_len, dim_head)
        K = K.view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)  # (B, num_heads, M, dim_head)
        V = V.view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)  # (B, num_heads, M, dim_head)
        # 计算注意力分数
        # Q @ K^T -> (B, num_heads, seq_len, M)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # 构建注意力mask: 如果only_attend_immediate_media，则对不该看的图像tokens位置加上很大的负值mask
        if self.only_attend_immediate_media and media_locations is not None:
            # media_locations shape (B, seq_len). True表示该位置有对应图像可attend，否则不应attend
            # 我们扩展维度以便与attn_scores匹配 (B, 1, seq_len, 1)
            mask = (~media_locations).unsqueeze(1).unsqueeze(-1)  # False->媒体存在的位置, ~取反后True表示需要mask掉（无媒体）的位置
            # 将mask为True的位置分数设置为一个非常大的负值 (使softmax后接近0权重)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, num_heads, seq_len, M)
        # 根据权重获取加权值
        attn_out = torch.matmul(attn_weights, V)  # (B, num_heads, seq_len, dim_head)
        # 合并heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len,
                                                              self.num_heads * self.dim_head)  # (B, seq_len, D)
        attn_out = self.to_out(attn_out)  # 线性输出投影回 hidden_dim
        # 门控机制: 通过tanh(gamma)控制跨注意力输出的幅度
        gated_attn_out = torch.tanh(self.gamma_attn) * attn_out
        # 仅将跨模态注意力输出添加到存在媒体的位置 (如果只允许immediate，在mask中已经处理，不存在媒体的地方attn_out为0)
        if self.only_attend_immediate_media and media_locations is not None:
            # 将 gated_attn_out 在无媒体的位置强制为0 (冗余安全，多重保险)
            gated_attn_out = gated_attn_out * media_locations.unsqueeze(-1).float()
        # 残差连接
        x = hidden_states + gated_attn_out
        # 前馈网络 (对跨模态融合后的隐状态继续变换)
        ff_in = self.norm_ff(x)
        ff_out = self.ff(ff_in)
        gated_ff_out = torch.tanh(self.gamma_ff) * ff_out
        # 第二次残差
        output = x + gated_ff_out
        return output


class BenTsaoWithFlamingoCrossAttention(LlamaForCausalLM):
    """
    「媒体特征与位置缓存机制由 Flamingo 层负责；
    语言模型 (BenTsaoWithFlamingoCrossAttention) 仅被动接受缓存数据。」
    """
    def __init__(self, config: LlamaConfig, cross_attn_every_n_layers: int = 4,
                 only_attend_immediate_media: bool = True, finetune_lm: bool = False):
        super().__init__(config)

        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        dim_head = hidden_size // num_heads

        self.cross_attn_every_n = cross_attn_every_n_layers
        self.gated_cross_attn_layers = nn.ModuleList([
            GatedCrossAttentionBlock(hidden_dim=hidden_size, num_heads=num_heads,
                                     dim_head=dim_head, only_attend_immediate_media=only_attend_immediate_media)
            if layer_idx % cross_attn_every_n_layers == 0 else None
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.cached_media_locations = None  # 明确初始化缓存变量
        self.cached_image_embeds = None

        if not finetune_lm:
            for param in self.model.parameters():
                param.requires_grad = False
            for layer in self.gated_cross_attn_layers:
                if layer is not None:
                    layer.requires_grad_(True)

    def init_flamingo(self, *args, **kwargs):
        # 跨模态层初始化已在构造函数完成，无需调用此方法。
        print("BenTsaoWithFlamingoCrossAttention 已内置跨模态层，跳过init_flamingo")

    def forward(self, input_ids=None, attention_mask=None, image_embeds=None, media_locations=None, labels=None, **kwargs):
        inputs_embeds = self.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for idx, layer_module in enumerate(self.model.layers):
            cross_attn_block = self.gated_cross_attn_layers[idx]
            if cross_attn_block is not None:
                hidden_states = cross_attn_block(hidden_states, image_embeds, media_locations)

            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return CausalLMOutputWithPast(loss=loss, logits=logits)

        return CausalLMOutputWithPast(logits=logits)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # 调用generate时务必确保传入image_embeds与media_locations或提前缓存，否则无法跨模态推理。
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_embeds": kwargs.get("image_embeds"),
            "media_locations": kwargs.get("media_locations"),
        }

    def condition_media_locations(self, media_locations):
        # 允许flamingo缓存媒体位置掩码
        self.cached_media_locations = media_locations

    def condition_vis_x(self, image_embeds):
        # 允许flamingo缓存视觉特征
        self.cached_image_embeds = image_embeds

    def clear_conditioned_layers(self):
        # 允许flamingo清除缓存
        self.cached_media_locations = None
        self.cached_image_embeds = None

