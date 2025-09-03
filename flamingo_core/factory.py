from flamingo_core.adapters import DualVisualAdapter  # adapters
from flamingo_core.bentsao_model import BenTsaoWithFlamingoCrossAttention
from flamingo_core.flamingo import Flamingo

from transformers import AutoTokenizer, LlamaConfig
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import torch
import torch.nn as nn
from flamingo_core.bio_llama_flamingo import BioMedicalLlamaFlamingo  # 新导入

class DummyCfg:
    """视觉编码器配置"""

    class DATA:
        TRAIN_CROP_SIZE = 224
        NUM_FRAMES = 8

    class MODEL:
        NUM_CLASSES = 0

    class TIMESFORMER:
        ATTENTION_TYPE = 'divided_space_time'
        PRETRAINED_MODEL = ''


def print_trainable_parameters(model):
    """打印模型参数统计"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params:,} ({trainable_params / 1e6:.1f}M) || "
        f"All params: {all_param:,} ({all_param / 1e6:.1f}M) || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def create_model_and_transforms(
        lang_encoder_path: str,
        tokenizer_path: str,
        endo_checkpoint_path: str,
        pmc_checkpoint_path: str,
        cross_attn_every_n_layers: int = 4,
        enable_endo: bool = True,
        enable_pmc: bool = True,
        add_branch_tokens: bool = True,
        target_hidden_dim: int = 4096,
        num_latents: int = 64,
        perceiver_depth: int = 2,
        perceiver_heads: int = 8,
        perceiver_dim_head: int = 64,
        freeze_endo: bool = True,
        freeze_pmc: bool = True,
        num_queries: int = 64,
        qformer_depth: int = 2,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        gradient_checkpointing: bool = False,  # 新增参数
        **kwargs
):
    """
    使用新的BioMedicalLlamaFlamingo创建模型
    """
    
    # 1. 加载分词器并添加特殊token
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")
    special_tokens = {
        "additional_special_tokens": ["<|endofchunk|>", "<image>"],
        "pad_token": "<pad>"
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 获取特殊token的ID
    media_token_id = tokenizer.convert_tokens_to_ids("<image>")
    eoc_token_id = tokenizer.convert_tokens_to_ids("<|endofchunk|>")
    
    # 2. 使用新的BioMedicalLlamaFlamingo
    llama_config = LlamaConfig.from_pretrained(lang_encoder_path)
    
    # 创建语言模型
    lang_encoder = BioMedicalLlamaFlamingo.from_pretrained(
        lang_encoder_path,
        config=llama_config,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        only_attend_immediate_media=False,  # 允许跨媒体注意力
        finetune_lm=False,  # 冻结LM主体
        torch_dtype=dtype,
        device_map={"": 0} if device == "cuda" else device,
        low_cpu_mem_usage=True
    )
    
    # 调整词嵌入大小
    lang_encoder.resize_token_embeddings(len(tokenizer))
    
    # 3. 初始化Flamingo交叉注意力层
    lang_encoder.init_flamingo_cross_attention(
        media_token_id=media_token_id,
        vis_dim=target_hidden_dim,
        gradient_checkpointing=gradient_checkpointing
    )
    
    # 4. 初始化双轨视觉编码器（保持不变）
    vision_encoder = DualVisualAdapter(
        cfg=DummyCfg(),
        endo_checkpoint_path=endo_checkpoint_path,
        pmc_checkpoint_path=pmc_checkpoint_path,
        target_hidden_dim=target_hidden_dim,
        num_latents=num_latents,
        perceiver_depth=perceiver_depth,
        perceiver_heads=perceiver_heads,
        perceiver_dim_head=perceiver_dim_head,
        freeze_endo=freeze_endo,
        freeze_pmc=freeze_pmc,
        enable_endo=enable_endo,
        enable_pmc=enable_pmc,
        add_branch_tokens=add_branch_tokens,
        num_queries=num_queries,
        qformer_depth=qformer_depth,
    )
    
    # 5. 创建Flamingo模型
    model = Flamingo(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        eoc_token_id=eoc_token_id,
        media_token_id=media_token_id,
        vis_dim=target_hidden_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers
    )
    
    # 6. 设置可训练参数（优化）
    model.requires_grad_(False)  # 先冻结所有
    
    # 解冻交叉注意力层（已经在init_flamingo_cross_attention中处理）
    
    # 解冻视觉适配器的关键组件
    trainable_patterns = [
        "perceiver_resampler",
        "linear_proj",
        "qformer",
        "endo_token",
        "pmc_token"
    ]
    
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in trainable_patterns):
            param.requires_grad = True
            
    # 7. 移动到设备并设置精度
    model = model.to(device=device, dtype=dtype)
    
    # 8. 打印参数统计
    print_trainable_parameters(model)
    
    return model, tokenizer


def create_model_for_training(
        base_model_path: str,
        endo_checkpoint_path: str,
        pmc_checkpoint_path: str,
        stage: str = "stage1",
        use_gradient_checkpointing: bool = False,
        **kwargs
):
    """
    根据训练阶段创建模型，使用新架构
    """
    if stage == "stage1":
        return create_model_and_transforms(
            lang_encoder_path=base_model_path,
            tokenizer_path=base_model_path,
            endo_checkpoint_path=endo_checkpoint_path,
            pmc_checkpoint_path=pmc_checkpoint_path,
            enable_endo=False,  # Stage 1不使用Endo
            enable_pmc=True,
            cross_attn_every_n_layers=6,
            gradient_checkpointing=use_gradient_checkpointing,
            **kwargs
        )
    else:  # stage2
        return create_model_and_transforms(
            lang_encoder_path=base_model_path,
            tokenizer_path=base_model_path,
            endo_checkpoint_path=endo_checkpoint_path,
            pmc_checkpoint_path=pmc_checkpoint_path,
            enable_endo=True,  # Stage 2启用双轨
            enable_pmc=True,
            cross_attn_every_n_layers=4,
            gradient_checkpointing=use_gradient_checkpointing,
            **kwargs
        )