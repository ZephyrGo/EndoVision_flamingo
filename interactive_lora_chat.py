import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def validate_paths(base_model_path, lora_path):
    # 检查基础模型路径是否存在必要文件
    required_files = ["config.json", "tokenizer.model"]
    for f in required_files:
        if not os.path.exists(os.path.join(base_model_path, f)):
            raise FileNotFoundError(f"❌ 缺失基础模型文件: {f} in {base_model_path}")

    # 检查 LoRA 权重路径
    lora_files = ["adapter_model.bin", "adapter_config.json"]
    for f in lora_files:
        if not os.path.exists(os.path.join(lora_path, f)):
            raise FileNotFoundError(f"❌ 缺失 LoRA 文件: {f} in {lora_path}")

def load_model(base_model_path, lora_path):
    print("🚀 正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("📦 正在加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print("🔧 正在加载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("🧩 合并 LoRA 到主模型中（提升推理速度）...")
    model = model.merge_and_unload()

    return model, tokenizer

def chat_loop(model, tokenizer, max_new_tokens=256, temperature=0.7):
    print("\n🩺 欢迎使用医学大模型问答系统！输入 'exit' 可退出对话。\n")
    while True:
        user_input = input("👤 你：")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 已退出。")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = response[len(user_input):].strip()
        print("🤖 模型：", reply)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医学 LoRA 聊天助手")
    parser.add_argument("--base_model", type=str, default="./checkpoints/llama-7b-hf")
    parser.add_argument("--lora_path", type=str, default="./checkpoints/lora-llama-med")
    args = parser.parse_args()

    validate_paths(args.base_model, args.lora_path)
    model, tokenizer = load_model(args.base_model, args.lora_path)
    chat_loop(model, tokenizer)
