import torch
from einops import rearrange
from torch import nn
from flamingo_core.helpers import PerceiverResampler
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from flamingo_core.utils import apply_with_stopping_condition


class Flamingo(nn.Module):
    def __init__(self, vision_encoder, lang_encoder, eoc_token_id, media_token_id, vis_dim, cross_attn_every_n_layers=1):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim
        self.perceiver = nn.Identity()

        self.cached_image_embeds = None
        self.cached_media_locations = None
        self._use_cached_vision_x = False

    def cache_media(self, vision_x, input_ids):
        """
        Flamingo架构本身不直接处理视频的“帧级”信息（F维度），
        而是依赖视觉编码器（如Endo-FM）来理解视频的帧序列（T维度）信息。
        :param vision_x:
        :param input_ids:
        :return:
        """
        if vision_x.ndim == 5:  # (B, T, C, H, W)
            vision_x = vision_x.unsqueeze(2)  # 转换为 (B, T, N=1, C, H, W)
        assert vision_x.ndim == 6, f"Expected 6D tensor, got {vision_x.ndim}D"
        b, T, F, c, h, w = vision_x.shape

        if F == 1:
            vision_x = vision_x.squeeze(2)  # (b, T, c, h, w)
        else:
            vision_x = vision_x.mean(dim=2)  # (b, T, c, h, w)

        # 明确将维度调整为 (B, C, T, H, W)，与Endo-FM模型兼容
        vision_x = rearrange(vision_x, "b T c h w -> b c T h w")

        # 调用Endo-FM视觉编码器，明确输出维度：(b, num_latents, hidden_dim)
        vision_x = self.vision_encoder(vision_x)["fused_tokens"]  # EndoFMAdapter输出：(b, num_latents, d)

        # print(f"[Before rearrange] vision_x.shape={vision_x.shape}")
        if vision_x.ndim == 4:
            b, T, v, d = vision_x.shape
            vision_x = vision_x.reshape(b * T, v, d)
        # print(f"[After reshape] vision_x.shape={vision_x.shape}")

        # 恢复T维度到原有shape：(b, T, num_latents, hidden_dim)
        vision_x = rearrange(vision_x, "(b T) v d -> b T v d", b=b, T=T)

        self.cached_image_embeds = vision_x

        media_locations = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            inside_media_block = False
            for t_idx in range(input_ids.size(1)):
                if input_ids[b_idx, t_idx] == self.media_token_id:
                    inside_media_block = True
                elif input_ids[b_idx, t_idx] == self.eoc_token_id:
                    inside_media_block = False
                media_locations[b_idx, t_idx] = inside_media_block

        self.cached_media_locations = media_locations
        self._use_cached_vision_x = True

    def clear_conditioned_layers(self):
        # 清空缓存
        self.cached_image_embeds = None
        self.cached_media_locations = None
        self._use_cached_vision_x = False

    def forward(self, vision_x=None, lang_x=None, attention_mask=None, labels=None, clear_conditioned_layers=True):
        if not self._use_cached_vision_x:
            self.cache_media(vision_x, lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            image_embeds=self.cached_image_embeds,
            media_locations=self.cached_media_locations
        )

        if clear_conditioned_layers:
            self.clear_conditioned_layers()

        return output

    def generate(self, vision_x, lang_x, attention_mask=None, **kwargs):
        self.cache_media(vision_x, lang_x)

        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            image_embeds=self.cached_image_embeds,
            media_locations=self.cached_media_locations,
            eos_token_id=kwargs.pop("eos_token_id", self.eoc_token_id),
            **kwargs
        )

        self.clear_conditioned_layers()

        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder(vision_x)  # 去掉索引[1]

        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
        vision_x = self.perceiver(vision_x)  # Identity，无变化

        for layer in self.lang_encoder._get_decoder_layers():
            if hasattr(layer, 'condition_vis_x'):
                layer.condition_vis_x(vision_x)

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_encoder.get_input_embeddings().requires_grad_(True)
            ```
        """
        # unfreeze the decoder layers
        for block in self.lang_encoder.old_decoder_blocks:
            block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver = wrap(wrap(self.perceiver))
            self.lang_encoder.old_decoder_blocks = nn.ModuleList(
                wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
            )
            self.lang_encoder.gated_cross_attn_layers = nn.ModuleList(
                wrap(wrap(layer)) if layer is not None else None
                for layer in self.lang_encoder.gated_cross_attn_layers
            )
            self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            self.lang_encoder.set_input_embeddings(
                wrap(wrap(self.lang_encoder.get_input_embeddings()))
            )
            self.lang_encoder.set_output_embeddings(
                wrap(wrap(self.lang_encoder.get_output_embeddings()))
            )
            self.vision_encoder = wrap(wrap(self.vision_encoder))  # frozen

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_encoder
        apply_with_stopping_condition(
            module=self.lang_encoder,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # exclude the original decoder layers from the optimizer
        for block in self.lang_encoder.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            for layer in self.lang_encoder.gated_cross_attn_layers:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_

    def _condition_media_locations(self, input_ids: torch.Tensor):
        media_locations = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(input_ids.size(0)):
            inside_media_block = False
            for t in range(input_ids.size(1)):
                if input_ids[b, t] == self.media_token_id:
                    inside_media_block = True
                elif input_ids[b, t] == self.eoc_token_id:
                    inside_media_block = False
                media_locations[b, t] = inside_media_block

        for layer in self.lang_encoder._get_decoder_layers():
            if hasattr(layer, 'condition_media_locations'):
                layer.condition_media_locations(media_locations)

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
