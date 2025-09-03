"""
基于flamingo_lm.py的Bio-Medical-Llama-3-8B集成
完全替代bentsao_model.py
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from flamingo_core.flamingo_lm import FlamingoLMMixin, FlamingoLayer
from flamingo_core.helpers import GatedCrossAttentionBlock


class BioMedicalLlamaFlamingo(FlamingoLMMixin, LlamaForCausalLM):
    """
    使用FlamingoLMMixin替代bentsao_model的实现
    保持与现有factory.py和flamingo.py的接口兼容
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        cross_attn_every_n_layers: int = 4,
        only_attend_immediate_media: bool = True,
        finetune_lm: bool = False,
        **kwargs
    ):
        # 初始化LlamaForCausalLM
        super().__init__(config)
        
        # 设置decoder层的路径（Llama的层在model.layers）
        self.set_decoder_layers_attr_name("model.layers")
        
        # 保存配置
        self.cross_attn_every_n = cross_attn_every_n_layers
        self.only_attend_immediate_media = only_attend_immediate_media
        
        # 这些属性是为了兼容性
        self.cached_image_embeds = None
        self.cached_media_locations = None
        self._use_cached_vision_x = False
        
        # 是否微调语言模型
        if not finetune_lm:
            # 冻结原始LLM参数
            for param in self.model.parameters():
                param.requires_grad = False
            # lm_head通常也要冻结
            for param in self.lm_head.parameters():
                param.requires_grad = False
                
        self.initialized_flamingo = False
        
    def init_flamingo_cross_attention(
        self,
        media_token_id: int,
        vis_dim: int = 4096,
        gradient_checkpointing: bool = False
    ):
        """
        初始化Flamingo交叉注意力层
        这个方法会在factory.py中调用
        """
        # 获取语言模型的隐藏维度
        lang_hidden_size = self.config.hidden_size
        
        # 初始化Flamingo层
        self.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=lang_hidden_size,
            vis_hidden_size=vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n,
            gradient_checkpointing=gradient_checkpointing
        )
        
        # 自定义门控交叉注意力层的参数
        for i, layer in enumerate(self.gated_cross_attn_layers):
            if layer is not None:
                # 设置only_attend_immediate_media
                layer.only_attend_immediate_media = self.only_attend_immediate_media
                
                # 初始化门控参数为小值而非0
                nn.init.constant_(layer.attn_gate, 0.1)
                nn.init.constant_(layer.ff_gate, 0.1)
        
        # 确保所有交叉注意力层都是可训练的
        for layer in self.gated_cross_attn_layers:
            if layer is not None:
                for param in layer.parameters():
                    param.requires_grad = True
                    
        self.initialized_flamingo = True
        
    def condition_vis_x(self, image_embeds):
        """兼容方法：条件化视觉特征"""
        if not self.initialized_flamingo:
            raise ValueError("Flamingo not initialized. Call init_flamingo_cross_attention first.")
            
        self.cached_image_embeds = image_embeds
        for layer in self._get_decoder_layers():
            if hasattr(layer, 'condition_vis_x'):
                layer.condition_vis_x(image_embeds)
                
    def condition_media_locations(self, media_locations):
        """兼容方法：条件化媒体位置"""
        if not self.initialized_flamingo:
            raise ValueError("Flamingo not initialized. Call init_flamingo_cross_attention first.")
            
        self.cached_media_locations = media_locations
        for layer in self._get_decoder_layers():
            if hasattr(layer, 'condition_media_locations'):
                layer.condition_media_locations(media_locations)
                
    def clear_conditioned_layers(self):
        """清除条件化的层"""
        self.cached_image_embeds = None
        self.cached_media_locations = None
        self._use_cached_vision_x = False
        
        if self.initialized_flamingo:
            super().clear_conditioned_layers()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        image_embeds=None,
        media_locations=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        position_ids=None,
        **kwargs
    ):
        """
        前向传播，兼容现有接口
        """
        # 如果提供了image_embeds，进行条件化
        if image_embeds is not None:
            self.condition_vis_x(image_embeds)
            
        # 如果提供了media_locations，进行条件化
        if media_locations is not None:
            self.condition_media_locations(media_locations)
        # 否则从input_ids中计算
        elif input_ids is not None and self.initialized_flamingo:
            media_locations = (input_ids == self.media_token_id)
            self.condition_media_locations(media_locations)
            
        # 调用父类的forward（会触发FlamingoLMMixin的forward）
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )
        
        # 计算loss
        loss = None
        if labels is not None:
            # Shift预测和标签
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def generate(self, *args, image_embeds=None, media_locations=None, **kwargs):
        """生成方法的包装"""
        if image_embeds is not None:
            self.condition_vis_x(image_embeds)
        if media_locations is not None:
            self.condition_media_locations(media_locations)
            
        # 设置生成模式
        self._use_cached_vision_x = True
        
        result = super().generate(*args, **kwargs)
        
        # 清理
        self.clear_conditioned_layers()
        
        return result