from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from torch import nn
from contextlib import nullcontext

# -----------------------------
# Gated Cross-Attention Block
# -----------------------------
class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dim_head,
                 only_attend_immediate_media: bool = True):
        super().__init__()
        self.only_attend_immediate_media = only_attend_immediate_media

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)

        self.to_q = nn.Linear(hidden_dim, num_heads * dim_head, bias=False)
        self.to_k = nn.Linear(hidden_dim, num_heads * dim_head, bias=False)
        self.to_v = nn.Linear(hidden_dim, num_heads * dim_head, bias=False)
        self.to_out = nn.Linear(num_heads * dim_head, hidden_dim, bias=False)

        inner = hidden_dim * 4
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_dim)
        )

        # 门控参数始终保持FP32
        self.gamma_attn = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.gamma_ff = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, hidden_states, image_embeds, media_locations):
        if image_embeds is None:
            return hidden_states
        
        # 保存原始dtype（可能是fp16或bf16）
        orig_dtype = hidden_states.dtype
        device = hidden_states.device
        
        # 确保门控参数在正确的设备上（不要重新赋值，只修改data）
        if self.gamma_attn.device != device:
            self.gamma_attn.data = self.gamma_attn.data.to(device)
            self.gamma_ff.data = self.gamma_ff.data.to(device)
        
        # 确保门控参数保持FP32
        if self.gamma_attn.dtype != torch.float32:
            self.gamma_attn.data = self.gamma_attn.data.float()
        if self.gamma_ff.dtype != torch.float32:
            self.gamma_ff.data = self.gamma_ff.data.float()
        
        # 限制范围
        with torch.no_grad():
            self.gamma_attn.data.clamp_(-3, 3)
            self.gamma_ff.data.clamp_(-3, 3)

        B, S, D = hidden_states.shape
        T, L = image_embeds.size(1), image_embeds.size(2)
        M = T * L

        # 正常的注意力计算（在当前dtype下进行）
        x = self.norm_attn(hidden_states)
        
        q = self.to_q(x).view(B, S, self.num_heads, self.dim_head).transpose(1, 2)
        kv = image_embeds.reshape(B, M, D)
        k = self.to_k(kv).view(B, M, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.to_v(kv).view(B, M, self.num_heads, self.dim_head).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if self.only_attend_immediate_media and media_locations is not None:
            mask = (~media_locations).unsqueeze(1).unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask, -1e4)

        attn = torch.softmax(attn_scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.to_out(out)

        # 门控：计算tanh并转换到目标dtype
        gamma_attn_val = torch.tanh(self.gamma_attn).to(orig_dtype)
        gated = gamma_attn_val * out
        
        if self.only_attend_immediate_media and media_locations is not None:
            media_mask = media_locations.unsqueeze(-1).to(orig_dtype)
            gated = gated * media_mask

        x = hidden_states + gated
        
        # FF层
        x_normed = self.norm_ff(x)
        ff_out = self.ff(x_normed)
        
        gamma_ff_val = torch.tanh(self.gamma_ff).to(orig_dtype)
        x = x + gamma_ff_val * ff_out
        
        return x
    
    def _apply(self, fn):
        """重写_apply以防止gamma参数被转换dtype"""
        # 应用函数到其他参数
        super()._apply(fn)
        
        # 强制gamma参数保持float32
        # 注意：修改.data而不是重新赋值整个Parameter
        self.gamma_attn.data = self.gamma_attn.data.float()
        self.gamma_ff.data = self.gamma_ff.data.float()
        
        return self


# ------------------------------------------------
# LLaMA + 每 n 层插入 Gated Cross-Attention
# ------------------------------------------------
class BenTsaoWithFlamingoCrossAttention(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig,
                 cross_attn_every_n_layers: int = 4,
                 only_attend_immediate_media: bool = True,
                 finetune_lm: bool = False):
        super().__init__(config)

        h = config.hidden_size
        n_heads = config.num_attention_heads
        d_head = h // n_heads

        self.cross_attn_every_n = cross_attn_every_n_layers
        self.gated_cross_attn_layers = nn.ModuleList([
            GatedCrossAttentionBlock(h, n_heads, d_head, only_attend_immediate_media)
            if i % cross_attn_every_n_layers == 0 else None
            for i in range(config.num_hidden_layers)
        ])

        # <<< 新增：保存 finetune 标志，后面 forward 要用 >>>
        self.finetune_lm = finetune_lm

        if not finetune_lm:
            for p in self.model.parameters():
                p.requires_grad = False
            for blk in self.gated_cross_attn_layers:
                if blk is not None:
                    blk.requires_grad_(True)

        # （可选）若你需要显存优化，可以在冻结 LLM 时开启 gradient checkpointing
        # if not finetune_lm and hasattr(self.model, "gradient_checkpointing_enable"):
        #     self.model.gradient_checkpointing_enable()  # 省显存但会稍慢

        self.cached_image_embeds = None
        self.cached_media_locations = None

    ...
    # ---------- 前向传播 ----------
    def forward(self,
                input_ids=None,
                attention_mask=None,
                image_embeds=None,
                media_locations=None,
                labels=None,
                position_ids=None,
                past_key_values=None,
                use_cache: bool = False,
                **kwargs):

        # 允许外层 no_grad() 的情况下，训练时重新开启梯度跟踪
        # 语义：LM 参数仍然 requires_grad=False 不会更新；
        # 但计算图会被记录，使梯度能回传到跨模态/门控/视觉侧可训练参数
        need_grad = (self.training and not self.finetune_lm)
        grad_ctx = torch.enable_grad() if need_grad else nullcontext()
        with grad_ctx:
            # 使用缓存的视觉特征
            if image_embeds is None:
                image_embeds = self.cached_image_embeds
            if media_locations is None:
                media_locations = self.cached_media_locations

            B, S = input_ids.shape
            device = input_ids.device

            # position_ids
            if position_ids is None:
                past_len = 0
                if past_key_values is not None and len(past_key_values) > 0:
                    if past_key_values[0] is not None and len(past_key_values[0]) > 0:
                        past_len = past_key_values[0][0].size(-2)
                position_ids = torch.arange(
                    past_len, past_len + S, device=device
                ).unsqueeze(0).expand(B, -1)

            # 词嵌入（LM 参数冻结，但允许构图传梯度到门控）
            inputs_embeds = self.model.embed_tokens(input_ids)

            # 4D attention mask
            attn_mask_4d = self._build_4d_attn_mask(attention_mask, inputs_embeds, past_key_values)

            # 前向传播
            hidden_states = inputs_embeds
            new_past = [] if use_cache else None

            for idx, layer_module in enumerate(self.model.layers):
                # 跨模态注意力（门控层，需要梯度）
                blk = self.gated_cross_attn_layers[idx]
                if blk is not None and image_embeds is not None:
                    hidden_states = blk(hidden_states, image_embeds, media_locations)

                # 标准 Transformer 层（参数冻结，但要保留计算图）
                layer_past = None
                if past_key_values is not None and idx < len(past_key_values):
                    if past_key_values[idx] is not None:
                        layer_past = past_key_values[idx]

                layer_out = layer_module(
                    hidden_states,
                    attention_mask=attn_mask_4d,
                    position_ids=position_ids[:, -hidden_states.size(1):],
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                hidden_states = layer_out[0]

                if use_cache:
                    new_past.append(layer_out[1])

            # 最终的 LayerNorm 和输出
            hidden_states = self.model.norm(hidden_states)
            logits = self.lm_head(hidden_states)

            # 计算损失
            loss = None
            if labels is not None:
                valid_labels = (labels != -100).sum()
                if valid_labels == 0:
                    # <<< 修改：兜底 loss 也“依赖 logits”，确保 requires_grad=True >>>
                    loss = logits.mean() * 0.0 + logits.new_tensor(0.01)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()

                    # 交叉熵本身会产生 float32；这里不强制 cast，让 AMP 处理
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                    # 若出现 NaN/Inf，用“有梯度的兜底”
                    if torch.isnan(loss) or torch.isinf(loss):
                        loss = shift_logits.mean() * 0.0 + shift_logits.new_tensor(0.01)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=tuple(new_past) if use_cache else None
            )

    # ---------- generate() 辅助函数 ----------
    def prepare_inputs_for_generation(self,
                                     input_ids,
                                     past_key_values=None,
                                     attention_mask=None,
                                     position_ids=None,
                                     **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
            if kwargs.get("media_locations") is not None:
                media_locations = kwargs.get("media_locations")
                new_location = torch.zeros(
                    (media_locations.shape[0], 1),
                    dtype=torch.bool,
                    device=media_locations.device
                )
                media_locations = torch.cat([media_locations, new_location], dim=1)
                kwargs["media_locations"] = media_locations[:, -1:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "image_embeds": kwargs.get("image_embeds"),
            "media_locations": kwargs.get("media_locations"),
        }

    def init_flamingo(self, *args, **kwargs):
        print("[Info] Cross-attention layers already initialized.")