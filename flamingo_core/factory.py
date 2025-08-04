from flamingo_core.dual_visual_adapter import DualVisualAdapter
from flamingo_core.bentsao_model import BenTsaoWithFlamingoCrossAttention
from flamingo_core.flamingo import Flamingo

from transformers import AutoTokenizer, LlamaConfig
from peft import PeftModel, PeftConfig

def debug_trainable_parameters(model):
    # 打印可训练参数，确认LoRA加载状态
    print("Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

def create_model_and_transforms(
    lang_encoder_path,
    tokenizer_path,
    endo_checkpoint_path,
    pmc_checkpoint_path,
    lora_path=None,
    merge_lora=False,
    cross_attn_every_n_layers=1,
    enable_endo=True,
    enable_pmc=True,
    add_branch_tokens=True,
    target_hidden_dim=4096,
    num_latents=64,
    perceiver_depth=2,
    perceiver_heads=8,
    perceiver_dim_head=64,
    freeze_endo=True,
    freeze_pmc=True,
    **kwargs
):
    # 加载语言模型配置
    llama_config = LlamaConfig.from_pretrained(lang_encoder_path)

    # 初始化语言编码器（带跨模态注意力）
    lang_encoder = BenTsaoWithFlamingoCrossAttention.from_pretrained(
        lang_encoder_path,
        config=llama_config,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        finetune_lm=False
    )

    # 可选LoRA加载
    if lora_path:
        peft_config = PeftConfig.from_pretrained(lora_path)
        assert peft_config.base_model_name_or_path == lang_encoder_path, "LoRA与基础模型不匹配"
        lang_encoder = PeftModel.from_pretrained(lang_encoder, lora_path)
        print(f"Loaded LoRA from {lora_path}")

        if merge_lora:
            lang_encoder = lang_encoder.merge_and_unload()
            print("LoRA weights merged into base model.")

    # Tokenizer 加载与特殊token添加
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    special_tokens = {"additional_special_tokens": ["<|endofchunk|>", "<image>"], "pad_token": "<PAD>"}
    tokenizer.add_special_tokens(special_tokens)
    lang_encoder.resize_token_embeddings(len(tokenizer))

    # 初始化双轨视觉编码器
    vision_encoder = DualVisualAdapter(
        cfg=kwargs.get('endo_cfg', {}),
        endo_checkpoint_path=endo_checkpoint_path,
        pmc_checkpoint_path=pmc_checkpoint_path,
        target_hidden_dim=target_hidden_dim,
        num_latents=num_latents,
        perceiver_depth=perceiver_depth,
        perceiver_heads=perceiver_heads,
        perceiver_dim_head=perceiver_dim_head,
        freeze_endo=freeze_endo,
        freeze_pmc=freeze_pmc,
    )

    # 创建最终的 Flamingo 模型实例，双轨视觉特征通过vision_encoder接入
    model = Flamingo(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        eoc_token_id=tokenizer.convert_tokens_to_ids("<|endofchunk|>"),
        media_token_id=tokenizer.convert_tokens_to_ids("<image>"),
        vis_dim=vision_encoder.output_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers
    )

    model.requires_grad_(False)

    # 激活需要训练的模块参数
    for name, param in model.named_parameters():
        if "gated_cross_attn_layers" in name:
            param.requires_grad = True
        if ("vision_encoder.pmc_adapter.perceiver_resampler" in name or
            "vision_encoder.pmc_adapter.linear_proj" in name):
            param.requires_grad = True
        if ("vision_encoder.endo_adapter.perceiver_resampler" in name or
            "vision_encoder.endo_adapter.linear_proj" in name):
            param.requires_grad = True
        if add_branch_tokens and ("vision_encoder.pmc_branch_token" in name or
                                  "vision_encoder.endo_branch_token" in name):
            param.requires_grad = True

    # 打印训练参数以确认正确加载
    debug_trainable_parameters(model)

    return model, tokenizer
