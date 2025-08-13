import os
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import transforms, io
from adapters import DualVisualAdapter
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


def load_dual_adapter(cfg_path, endo_checkpoint_path, pmc_checkpoint_path, device='cpu'):
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        cfg = CN(cfg_dict)

    # 明确补充缺失的PRETRAINED_MODEL字段
    if 'PRETRAINED_MODEL' not in cfg.TIMESFORMER:
        cfg.TIMESFORMER.PRETRAINED_MODEL = endo_checkpoint_path

    adapter = DualVisualAdapter(
        cfg=cfg,
        endo_checkpoint_path=endo_checkpoint_path,
        pmc_checkpoint_path=pmc_checkpoint_path,
        target_hidden_dim=4096,
        num_latents=64,
        perceiver_depth=2,
        perceiver_heads=8,
        perceiver_dim_head=64,
        freeze_endo=True,
        freeze_pmc=True,
        enable_endo=True,
        enable_pmc=True,
        add_branch_tokens=True
    ).to(device)

    return adapter



def extract_adapter_tokens(adapter, frames):
    adapter.eval()
    with torch.no_grad():
        output = adapter(frames)
    return output


def visualize_token_distribution(tokens, title="Token Distribution"):
    B, T, N, D = tokens.shape
    tokens = tokens[0, 0].detach().cpu().numpy()
    pca = PCA(n_components=2).fit_transform(tokens)

    plt.figure(figsize=(6, 6))
    plt.scatter(pca[:, 0], pca[:, 1], alpha=0.8)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# def visualize_token_distribution(patch_tokens, adapter, frame_idx=0):
#     proj = adapter.linear_proj(patch_tokens)[0, frame_idx].detach().cpu().numpy()
#     pca = PCA(n_components=2).fit_transform(proj)
#     grid_size = int(np.sqrt(proj.shape[0]))
#     colors = np.arange(proj.shape[0]) // grid_size if grid_size ** 2 == proj.shape[0] else 'blue'
#
#     plt.figure(figsize=(5, 5))
#     plt.scatter(pca[:, 0], pca[:, 1], c=colors, cmap='viridis', alpha=0.8)
#     if isinstance(colors, np.ndarray):
#         plt.colorbar(label='Patch row index')
#     plt.title(f"PCA of Patch Tokens (Frame {frame_idx})")
#     plt.tight_layout()
#     plt.show()

def visualize_token_norm(tokens_dict):
    plt.figure(figsize=(6, 4))
    for name, tokens in tokens_dict.items():
        if tokens is not None:
            norm = torch.norm(tokens[0], dim=-1).mean(dim=-1).cpu().numpy()
            plt.plot(norm, marker='o', label=name)
        else:
            print(f"Warning: {name} is None, skipped.")

    plt.title("Mean Token L2 Norm per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean L2 Norm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def summarize_dimensions(video_frames, output):
    B, C, T, H, W = video_frames.shape
    print("\n=== Feature Dimension Summary ===")
    print(f"Input: B={B}, C={C}, T={T}, H={H}, W={W}")

    for key, tokens in output.items():
        if tokens is not None:
            _, _, N, D = tokens.shape
            print(f"{key} shape: (B, T, {N}, {D})")
        else:
            print(f"{key} is None.")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_path = "../data/check_func/endoscope_video.mp4"
    endo_checkpoint_path = "../checkpoints/endo_fm_convert.pth"
    pmc_checkpoint_path = "../checkpoints/pmc_clip_visual_only.pt"
    cfg_path = "../endo_fm_backbone/configs/TimeSformer_divST_8x32_224.yaml"

    video_frames = load_and_preprocess_video(video_path, device=device)
    adapter = load_dual_adapter(cfg_path, endo_checkpoint_path, pmc_checkpoint_path, device=device)

    output = extract_adapter_tokens(adapter, video_frames)

    summarize_dimensions(video_frames, output)

    visualize_token_distribution(output["fused_tokens"], title="Fused Token Distribution")

    visualize_token_norm({
        "Fused tokens": output["fused_tokens"],
        "PMC tokens": output["pmc_tokens"],
        "Endo tokens": output["endo_tokens"]
    })
