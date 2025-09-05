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
        gradient_checkpointing: bool = False,  # 梯度检查
        use_mixed_precision: bool = True,   # 使用混合精度
        **kwargs
):
    """
    使用新的BioMedicalLlamaFlamingo创建模型
        创建具有正确混合精度设置的模型：
    - 冻结的基础模型使用 FP16 以节省显存
    - 可训练参数使用 FP32 以保证梯度稳定性
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
    
    # 4. 初始化双轨视觉编码器（frozen parts stay FP16）
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

    # 将冻结的视觉编码器参数放入fp16   
    vision_encoder = vision_encoder.to(device=device)

    
    # 5. 创建Flamingo模型
    model = Flamingo(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        eoc_token_id=eoc_token_id,
        media_token_id=media_token_id,
        vis_dim=target_hidden_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers
    )
    model = model.to(device=device)  # 确保整个模型在GPU上

    # 6. 设置可训练参数（优化）
    model.requires_grad_(False)  # 先冻结所有
     
    # 解冻视觉适配器的关键组件和交叉注意力层
    trainable_patterns = [
        "gated_cross_attn_layers",  # Flamingo cross-attention
        "perceiver_resampler",
        "linear_proj",
        "qformer",
        "endo_token",
        "pmc_token"
    ]
            
    # 7. 移动到设备并设置精度
    if use_mixed_precision:
        print("Setting up mixed precision training...")
        print("  - Frozen parameters (base model): FP16")
        print("  - Trainable parameters: FP32")
        
        # Keep frozen parts in FP16 to save memory
        for name, param in model.named_parameters():
            if not any(pattern in name for pattern in trainable_patterns):
                # Frozen parameters stay in FP16
                param.data = param.data.to(dtype=dtype)
            else:
                # Trainable parameters must be FP32 for GradScaler
                param.data = param.data.to(dtype=torch.float32)
                param.requires_grad = True
                
        # Special handling for gated attention parameters
        if hasattr(model.lang_encoder, 'gated_cross_attn_layers'):
            for layer in model.lang_encoder.gated_cross_attn_layers:
                if layer is not None:
                    for param in layer.parameters():
                        param.data = param.data.float()  # Ensure FP32
                        param.requires_grad = True
    else:
        # No mixed precision - all parameters in specified dtype
        print(f"Standard precision training with dtype={dtype}")
        model = model.to(device=device, dtype=dtype)
        
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns):
                param.requires_grad = True
    
    # 8. 打印参数统计
    print("\nParameter dtype configuration:")
    fp16_params = 0
    fp32_params = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.float16:
            fp16_params += param.numel()
        elif param.dtype == torch.float32:
            fp32_params += param.numel()
            
    print(f"  FP16 parameters: {fp16_params:,} ({fp16_params/1e6:.1f}M)")
    print(f"  FP32 parameters: {fp32_params:,} ({fp32_params/1e6:.1f}M)")
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    # 9. Create a custom forward wrapper to handle mixed dtypes
    original_forward = model.forward
    
    def mixed_precision_forward(vision_x=None, lang_x=None, attention_mask=None, labels=None, **kwargs):
        # Ensure inputs are in correct dtype
        if vision_x is not None:
            vision_x = vision_x.to(dtype=torch.float16)  # Vision always FP16 for autocast
        return original_forward(vision_x=vision_x, lang_x=lang_x, attention_mask=attention_mask, labels=labels, **kwargs)
    
    if use_mixed_precision:
        model.forward = mixed_precision_forward
    
    return model, tokenizer


def create_model_for_training(
        base_model_path: str,
        endo_checkpoint_path: str,
        pmc_checkpoint_path: str,
        stage: str = "stage1",
        use_gradient_checkpointing: bool = False,
        use_mixed_precision: bool = True,  # Default to mixed precision
        **kwargs
):
    """
    根据训练阶段创建模型
    """
    if stage == "stage1":
        return create_model_and_transforms(
            lang_encoder_path=base_model_path,
            tokenizer_path=base_model_path,
            endo_checkpoint_path=endo_checkpoint_path,
            pmc_checkpoint_path=pmc_checkpoint_path,
            enable_endo=False,  # Stage 1 doesn't use Endo
            enable_pmc=True,
            cross_attn_every_n_layers=6,
            gradient_checkpointing=use_gradient_checkpointing,
            use_mixed_precision=use_mixed_precision,
            dtype=torch.float16 if use_mixed_precision else torch.float32,
            **kwargs
        )
    else:  # stage2
        return create_model_and_transforms(
            lang_encoder_path=base_model_path,
            tokenizer_path=base_model_path,
            endo_checkpoint_path=endo_checkpoint_path,
            pmc_checkpoint_path=pmc_checkpoint_path,
            enable_endo=True,  # Stage 2 uses both branches
            enable_pmc=True,
            cross_attn_every_n_layers=4,
            gradient_checkpointing=use_gradient_checkpointing,
            use_mixed_precision=use_mixed_precision,
            dtype=torch.float16 if use_mixed_precision else torch.float32,
            **kwargs
        )