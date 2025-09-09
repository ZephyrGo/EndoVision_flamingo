from transformers import AutoTokenizer, AutoModelForCausalLM

# 本地路径（假设你已经上传到这个目录）
local_model_path = "/home/zzk01/lqy_proj/EndoVision_flamingo/checkpoints/models--ContactDoctor--Bio-Medical-Llama-3-8B/snapshots/d71486ab8c920f34afe37ec8a21c2035b83d7b2d"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
