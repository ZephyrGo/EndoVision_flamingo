import torch

# 加载 checkpoint
ckpt = torch.load('endo_fm_convert.pth', map_location='cpu')

# # 保存干净的视觉编码器权重
# student = ckpt["student"]
# torch.save(student, "endo_vit_backbone.pth")

# 打印 key 和 shape
for k, v in ckpt.items():
    print(f"{k:60s} → {tuple(v.shape)}")

with open('endo_fm_convert_keys.txt', 'w') as f:
    for k, v in ckpt.items():
        print(f"{k}: {v.shape}")
        f.write(f"{k}: {tuple(v.shape)}\n")  # 将张量维度转换为 tuple 便于可读


# def inspect_pth_file(pth_path):
#     """
#     查看并打印 PyTorch 模型权重文件(.pth)的内部结构和参数shape。
#     """
#     checkpoint = torch.load(pth_path, map_location='cpu')
#
#     print(f"Inspecting {pth_path}\n{'-'*60}")
#
#     if isinstance(checkpoint, dict):
#         for key, value in checkpoint.items():
#             if isinstance(value, torch.Tensor):
#                 print(f"{key:<60} -> {tuple(value.shape)}")
#             elif isinstance(value, dict):
#                 print(f"Nested dictionary found at '{key}', exploring nested structure...")
#                 for subkey, subvalue in value.items():
#                     if isinstance(subvalue, torch.Tensor):
#                         print(f"  {subkey:<58} -> {tuple(subvalue.shape)}")
#                     else:
#                         print(f"  {subkey:<58} -> (non-tensor value)")
#             else:
#                 print(f"{key:<60} -> (non-tensor value)")
#     else:
#         print(f"Checkpoint type: {type(checkpoint)}, value: {checkpoint}")
#
#
# if __name__ == '__main__':
#     pth_file_path = "endo_fm.pth"
#     inspect_pth_file(pth_file_path)


# def convert_pth(pth_path, new_pth_path):
#     checkpoint = torch.load(pth_path, map_location='cpu')
#
#     # 如果权重结构嵌套在student或teacher中，则提取出来
#     if 'student' in checkpoint:
#         checkpoint = checkpoint['student']
#     elif 'teacher' in checkpoint:
#         checkpoint = checkpoint['teacher']
#
#     # 明确去除 module.backbone. 或 backbone. 前缀
#     new_checkpoint = {}
#     for k, v in checkpoint.items():
#         new_key = k.replace('module.backbone.', '').replace('backbone.', '')
#         new_checkpoint[new_key] = v
#
#     # 保存新的权重
#     torch.save(new_checkpoint, new_pth_path)
#     print(f"Converted checkpoint saved to {new_pth_path}")
#
# # 使用方法 (注意修改为你的实际路径)
# convert_pth('endo_fm.pth', 'endo_fm_convert.pth')

# 查看pmc_clip.pt结构
data = torch.load('pmc_clip.pt')
print(type(data))
if isinstance(data, dict):
    for key in data.keys():
        print(key)
with open('pmc_clip_state_dict_keys.txt', 'w') as f:
    for k, v in data['state_dict'].items():
        print(f"{k}: {v.shape}")
        f.write(f"{k}: {tuple(v.shape)}\n")  # 将张量维度转换为 tuple 便于可读
