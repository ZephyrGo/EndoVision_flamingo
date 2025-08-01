from flamingo_core.endo_adapter import EndoFMAdapter
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
    lora_path=None,
    merge_lora=False,
    cross_attn_every_n_layers=1,
    **kwargs
):
    llama_config = LlamaConfig.from_pretrained(lang_encoder_path)

    lang_encoder = BenTsaoWithFlamingoCrossAttention.from_pretrained(
        lang_encoder_path,
        config=llama_config,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        finetune_lm=False
    )

    if lora_path:
        peft_config = PeftConfig.from_pretrained(lora_path)
        assert peft_config.base_model_name_or_path == lang_encoder_path, "LoRA与基础模型不匹配"
        lang_encoder = PeftModel.from_pretrained(lang_encoder, lora_path)
        print(f"Loaded LoRA from {lora_path}")

        if merge_lora:
            lang_encoder = lang_encoder.merge_and_unload()
            print("LoRA weights merged into base model.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    special_tokens = {"additional_special_tokens": ["<|endofchunk|>", "<image>"], "pad_token": "<PAD>"}
    tokenizer.add_special_tokens(special_tokens)
    lang_encoder.resize_token_embeddings(len(tokenizer))

    vision_encoder = EndoFMAdapter(**kwargs)

    model = Flamingo(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        eoc_token_id=tokenizer.convert_tokens_to_ids("<|endofchunk|>"),
        media_token_id=tokenizer.convert_tokens_to_ids("<image>"),
        vis_dim=vision_encoder.output_dim,
        cross_attn_every_n_layers=cross_attn_every_n_layers
    )

    model.requires_grad_(False)

    for name, param in model.named_parameters():
        if "gated_cross_attn_layers" in name:
            param.requires_grad = True
        if "vision_encoder.perceiver_resampler" in name or "vision_encoder.linear_proj" in name:
            param.requires_grad = True

    return model, tokenizer
