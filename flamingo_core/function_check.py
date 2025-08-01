import os
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import transforms, io
from endo_adapter import EndoFMAdapter
from endo_fm_backbone.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from endo_fm_backbone.parser import load_config
import yaml
from yacs.config import CfgNode as CN


def load_and_preprocess_video(video_path, num_frames=8, device='cpu'):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video, _, _ = io.read_video(video_path, pts_unit="sec")
    total = video.shape[0]
    idxs = torch.linspace(0, total - 1, steps=num_frames).long()
    frames = video[idxs]

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])

    processed = [tf(Image.fromarray(f.numpy())) for f in frames]
    stacked = torch.stack(processed, dim=1).unsqueeze(0).to(device)
    return stacked


def load_model(cfg_path, checkpoint_path, device='cpu'):
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        cfg = CN(cfg_dict)

    # 直接在这里补充缺失的字段
    cfg.TIMESFORMER.PRETRAINED_MODEL = checkpoint_path

    adapter = EndoFMAdapter(
        cfg=cfg,
        endo_checkpoint_path=checkpoint_path,
        target_hidden_dim=4096,
        num_latents=64,
        perceiver_depth=2,
        perceiver_heads=8,
        perceiver_dim_head=64,
        freeze_endo=True
    ).to(device)
    return adapter, adapter.endo_model


def extract_patch_tokens(endo_model, frames):
    with torch.no_grad():
        # 明确使用 get_all=True 获取完整的patch tokens序列
        feats = endo_model.forward_features(frames, get_all=True)

    B, C, T, H, W = frames.shape
    if feats.dim() == 4:
        feats = feats[:, :, 1:, :]
    elif feats.dim() == 3:
        seq_len = feats.shape[1]
        feats = feats[:, 1:, :] if seq_len % T != 0 else feats
        N_patch = feats.shape[1] // T
        feats = feats.view(B, T, N_patch, feats.shape[-1])
    else:
        raise RuntimeError(f"Unexpected feature dimensions: {feats.shape}")

    return feats



def extract_adapter_tokens(adapter, frames):
    adapter.eval()
    with torch.no_grad():
        image_embeds = adapter(frames)
    return image_embeds


def visualize_patch_distribution(patch_tokens, adapter, frame_idx=0):
    proj = adapter.linear_proj(patch_tokens)[0, frame_idx].detach().cpu().numpy()
    pca = PCA(n_components=2).fit_transform(proj)
    grid_size = int(np.sqrt(proj.shape[0]))
    colors = np.arange(proj.shape[0]) // grid_size if grid_size ** 2 == proj.shape[0] else 'blue'

    plt.figure(figsize=(5, 5))
    plt.scatter(pca[:, 0], pca[:, 1], c=colors, cmap='viridis', alpha=0.8)
    if isinstance(colors, np.ndarray):
        plt.colorbar(label='Patch row index')
    plt.title(f"PCA of Patch Tokens (Frame {frame_idx})")
    plt.tight_layout()
    plt.show()




def visualize_temporal_norm(patch_tokens, adapter_tokens, adapter):
    patch_proj = adapter.linear_proj(patch_tokens)[0]
    adapter_proj = adapter_tokens[0]
    T = min(patch_proj.shape[0], adapter_proj.shape[0])

    p_norm = [torch.norm(patch_proj[t], dim=1).mean().item() for t in range(T)]
    a_norm = [torch.norm(adapter_proj[t], dim=1).mean().item() for t in range(T)]

    plt.figure(figsize=(6, 4))
    plt.plot(range(T), p_norm, marker='o', label="Patch tokens")
    plt.plot(range(T), a_norm, marker='s', label="Adapter tokens")
    plt.title("Mean Token L2 Norm per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean L2 Norm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def summarize_dimensions(video_frames, patch_tokens, adapter_tokens):
    B, C, T, H, W = video_frames.shape
    _, _, N_patch, D_orig = patch_tokens.shape
    _, _, num_latents, D_target = adapter_tokens.shape

    print("\n=== Feature Dimension Summary ===")
    print(f"Input: B={B}, C={C}, T={T}, H={H}, W={W}")
    print(f"Patch token shape: (B, T, {N_patch}, {D_orig})")
    print(f"Adapter output shape: (B, T, {num_latents}, {D_target})")


# =================== 主程序调用 ===================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_path = "../data/check_func/endoscope_video.mp4"
    checkpoint_path = "../checkpoints/endo_fm_convert.pth"
    cfg_path = "../endo_fm_backbone/configs/TimeSformer_divST_8x32_224.yaml"

    video_frames = load_and_preprocess_video(video_path, device=device)
    adapter, endo_model = load_model(cfg_path, checkpoint_path, device=device)
    patch_tokens = extract_patch_tokens(endo_model, video_frames)
    adapter_tokens = extract_adapter_tokens(adapter, video_frames)

    summarize_dimensions(video_frames, patch_tokens, adapter_tokens)
    visualize_patch_distribution(patch_tokens, adapter)
    visualize_temporal_norm(patch_tokens, adapter_tokens, adapter)
