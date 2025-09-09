import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from flamingo_core.adapters import DualVisualAdapter
import torch
import unittest
import torch.nn as nn
from flamingo_core.flamingo import Flamingo
from flamingo_core.bentsao_model import BenTsaoWithFlamingoCrossAttention
from transformers import AutoTokenizer, LlamaConfig

LOCAL_LM_PATH = "/home/zzk01/lqy_proj/EndoVision_flamingo/checkpoints/models--ContactDoctor--Bio-Medical-Llama-3-8B/snapshots/d71486ab8c920f34afe37ec8a21c2035b83d7b2d"
class DummyCfg:
    class DATA:
        TRAIN_CROP_SIZE = 224
        NUM_FRAMES = 8

    class MODEL:
        NUM_CLASSES = 0

    class TIMESFORMER:
        ATTENTION_TYPE = 'divided_space_time'
        PRETRAINED_MODEL = ''

class DummyConfig:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

class DummyLangEncoder(nn.Module):
    def __init__(self, vocab_size=32000):
        super(DummyLangEncoder, self).__init__()
        self.config = type('DummyConfig', (), {'vocab_size': vocab_size})

    def forward(self, input_ids=None, attention_mask=None, labels=None, image_embeds=None, media_locations=None, **kwargs):
        B, seq_len = input_ids.shape
        logits = torch.randn(B, seq_len, self.config.vocab_size, device=input_ids.device)
        return type('DummyOutput', (), {'logits': logits})

    def generate(self, input_ids=None, attention_mask=None, image_embeds=None, media_locations=None, **kwargs):
        return ["mocked generated text"] * input_ids.shape[0]


class DummyBatchEncoding(dict):
    def to(self, device):
        return {k: v.to(device) for k, v in self.items()}


class DummyTokenizer:
    def __init__(self):
        self.token_to_id = {
            "<pad>": 0, "<bos>": 1, "<eos>": 2,
            "<|endofchunk|>": 3, "<image>": 4, "hello": 5, "world": 6, "test": 7
        }
        self.pad_token = "<pad>"
        self.additional_special_tokens = set()

    def __call__(self, texts, return_tensors=None, padding=False):
        texts = [texts] if isinstance(texts, str) else texts
        encoded_batches = []
        for text in texts:
            ids = [self.token_to_id.get("<bos>", 1)]
            for word in text.lower().split():
                ids.append(self.token_to_id.get(word, 99))
            ids.append(self.token_to_id.get("<eos>", 2))
            encoded_batches.append(ids)

        if padding:
            max_len = max(len(ids) for ids in encoded_batches)
            padded = [ids + [self.token_to_id[self.pad_token]] * (max_len - len(ids)) for ids in encoded_batches]
        else:
            padded = encoded_batches

        input_ids = torch.tensor(padded, dtype=torch.long)

        result = DummyBatchEncoding({"input_ids": input_ids})

        return result if return_tensors == 'pt' else dict(result)

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, 99)

    def add_special_tokens(self, special_tokens_dict):
        tokens = special_tokens_dict.get("additional_special_tokens", [])
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                self.additional_special_tokens.add(token)
        pad_token = special_tokens_dict.get("pad_token")
        if pad_token:
            self.pad_token = pad_token
            if pad_token not in self.token_to_id:
                self.token_to_id[pad_token] = len(self.token_to_id)


class TestDualVisualAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---------- 视觉适配器 ----------
        cls.adapter = (
            DualVisualAdapter(
                cfg=DummyCfg(),
                endo_checkpoint_path="checkpoints/endo_fm_convert.pth",
                pmc_checkpoint_path="checkpoints/pmc_clip_visual_only.pt",
                target_hidden_dim=4096,
                num_latents=64,
                perceiver_depth=2,
                perceiver_heads=8,
                perceiver_dim_head=64,
                enable_endo=True,
                enable_pmc=True,
                freeze_endo=True,
                freeze_pmc=True,
                num_queries=64,
                qformer_depth=2,
            )
            .to(cls.device)
            .half()                                        # 🔑 适配器统一 FP16
            .eval()
        )

        # ---------- 分词器 ----------
        cls.tokenizer = AutoTokenizer.from_pretrained(LOCAL_LM_PATH, padding_side="right")
        cls.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": ["<|endofchunk|>", "<image>"],
                "pad_token": "<pad>",
            }
        )

        # ---------- 语言模型 ----------
        cfg = LlamaConfig.from_pretrained(LOCAL_LM_PATH)
        cls.lang_encoder = BenTsaoWithFlamingoCrossAttention.from_pretrained(
            LOCAL_LM_PATH,
            config=cfg,
            cross_attn_every_n_layers=2,
            torch_dtype=torch.float16,                     # 先流式加载为 FP16
            device_map={"": 0},                            # 单 GPU
            low_cpu_mem_usage=True,
            only_attend_immediate_media=False,
            finetune_lm=False,
        )
        cls.lang_encoder.resize_token_embeddings(len(cls.tokenizer))
        cls.lang_encoder.half()                            # 统一 LayerNorm 等剩余 FP32 权重
        cls.lang_encoder.eval()

        # ---------- Flamingo 容器 ----------
        cls.model = (
            Flamingo(
                vision_encoder=cls.adapter,
                lang_encoder=cls.lang_encoder,
                eoc_token_id=cls.tokenizer.convert_tokens_to_ids("<|endofchunk|>"),
                media_token_id=cls.tokenizer.convert_tokens_to_ids("<image>"),
                vis_dim=cls.lang_encoder.config.hidden_size,
                cross_attn_every_n_layers=4,
            )
            .to(cls.device)
            .half()                                        
            .eval()
        )

        def _force_layernorm_to_half(module: torch.nn.Module):
            for m in module.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    # 权重/偏置转成 half
                    m.to(dtype=torch.float16, device=next(module.parameters()).device)

        _force_layernorm_to_half(cls.lang_encoder)

    # --------------------------------------------------
    #                单元测试开始
    # --------------------------------------------------
    def _dummy_images(self, B, T=1, C=3, H=224, W=224):
        return torch.randn(B, T, C, H, W, device=self.device, dtype=torch.float16)

    def test_full_integration_forward(self):
        B = 2
        imgs = self._dummy_images(B)
        txt = self.tokenizer(
            ["This is a test medical report.", "Endoscopy findings are normal."],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        out = self.model(
            vision_x=imgs,
            lang_x=txt["input_ids"],
            attention_mask=txt["attention_mask"], 
        )
        self.assertEqual(
            out.logits.shape,
            (B, txt["input_ids"].shape[1], self.lang_encoder.config.vocab_size),
        )

    def test_generation(self):
        B = 2
        imgs = self._dummy_images(B)
        prompts = self.tokenizer(
            ["<image> Findings:", "<image> Impressions:"],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        gen = self.model.generate(vision_x=imgs, lang_x=prompts["input_ids"], max_length=50)
        self.assertEqual(len(gen), B)
        for i, t in enumerate(gen): print(f"[Gen {i}] {t}")

    def test_memory_efficiency(self):
        B = 2
        imgs = self._dummy_images(B)
        torch.cuda.reset_peak_memory_stats(self.device)

        txt = self.tokenizer(["test"] * B, return_tensors="pt").to(self.device)  # 保证 batch 对齐
        _ = self.model(vision_x=imgs, lang_x=txt["input_ids"])

        peak = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        print(f"Peak GPU mem = {peak:.1f} MB")
        self.assertLess(peak, 28000)  # 根据实际显存适当调整

    def test_adapter_output_shapes(self):
        B, T = 2, 1
        outs = self.adapter(self._dummy_images(B, T))
        self.assertEqual(outs["fused_tokens"].shape, (B, T, 130, 4096))

    def test_trainable_parameters(self):
        train = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        freeze = sum(p.numel() for p in self.adapter.parameters() if not p.requires_grad)
        print(f"Trainable {train/1e6:.1f} M / Frozen {freeze/1e6:.1f} M")
        self.assertGreater(train, 0)
        self.assertGreater(freeze, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
