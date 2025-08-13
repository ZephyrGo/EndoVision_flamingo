from adapters import DualVisualAdapter
import torch
import unittest
import torch.nn as nn
from flamingo_core.flamingo import Flamingo
from flamingo_core.bentsao_model import BenTsaoWithFlamingoCrossAttention
from transformers import AutoTokenizer


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

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.adapter = DualVisualAdapter(
            cfg=DummyCfg(),
            endo_checkpoint_path='../checkpoints/endo_fm_convert.pth',
            pmc_checkpoint_path='../checkpoints/pmc_clip_visual_only.pt',
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
            qformer_depth=2
        ).to(self.device)
        # 使用扩展后的DummyTokenizer
        self.tokenizer = DummyTokenizer()
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|endofchunk|>", "<image>"],
            "pad_token": "<pad>"
        })

        # DummyLangEncoder
        self.lang_encoder = DummyLangEncoder().to(self.device)

        self.model = Flamingo(
            vision_encoder=self.adapter,
            lang_encoder=self.lang_encoder,
            eoc_token_id=self.tokenizer.convert_tokens_to_ids("<|endofchunk|>"),
            media_token_id=self.tokenizer.convert_tokens_to_ids("<image>"),
            vis_dim=4096,
            cross_attn_every_n_layers=1
        ).to(self.device)

        self.adapter.eval()

    def test_full_integration_forward(self):
        B, T, C, H, W = 2, 1, 3, 224, 224
        video_tensor = torch.randn(B, T, C, H, W, device=self.device)
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)

        # Tokenizer真实文本输入
        inputs = self.tokenizer(
            ["This is a test medical report.", "Endoscopy findings are normal."],
            return_tensors='pt',
            padding=True
        ).to(self.device)

        input_ids = inputs['input_ids']

        output = self.model(
            vision_x=image_tensor,
            lang_x=input_ids
        )

        # 验证输出logits的维度
        self.assertEqual(output.logits.shape, (B, input_ids.shape[1], self.lang_encoder.config.vocab_size))
        print("完整模型输出logits维度:", output.logits.shape)

    def test_generation(self):
        B, T, C, H, W = 2, 1, 3, 224, 224
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)

        inputs = self.tokenizer(
            ["<image> Findings:", "<image> Impressions:"],
            return_tensors='pt',
            padding=True
        ).to(self.device)

        generated_texts = self.model.generate(
            vision_x=image_tensor,
            lang_x=inputs['input_ids'],
            max_length=50
        )

        self.assertEqual(len(generated_texts), B)
        for idx, text in enumerate(generated_texts):
            print(f"Generated text {idx + 1}: {text}")

    def test_memory_efficiency(self):
        B, T, C, H, W = 2, 1, 3, 224, 224
        video_tensor = torch.randn(B, T, C, H, W, device=self.device)
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)

        torch.cuda.reset_peak_memory_stats(self.device)
        inputs = self.tokenizer(["<image> Test memory usage"], return_tensors='pt').to(self.device)

        _ = self.model(
            vision_x=image_tensor,
            lang_x=inputs['input_ids']
        )

        max_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        print(f"Peak GPU memory usage: {max_mem:.2f} MB")
        self.assertLess(max_mem, 12000, "GPU内存占用超过预期阈值")

    def test_adapter_output_shapes(self):
        B, T, C, H, W = 2, 1, 3, 224, 224
        video_tensor = torch.randn(B, T, C, H, W, device=self.device)

        outputs = self.adapter(video_tensor)

        fused_tokens = outputs["fused_tokens"]
        pmc_tokens = outputs["pmc_tokens"]
        endo_tokens = outputs["endo_tokens"]

        expected_shape_single_branch = (B, T, 65, 4096)
        expected_shape_fused = (B, T, 130, 4096)

        self.assertEqual(fused_tokens.shape, expected_shape_fused, "Fused tokens shape mismatch")
        self.assertEqual(pmc_tokens.shape, expected_shape_single_branch, "PMC tokens shape mismatch")
        self.assertEqual(endo_tokens.shape, expected_shape_single_branch, "Endo tokens shape mismatch")

        print("DualVisualAdapter output shapes test passed:")
        print("Fused tokens shape:", fused_tokens.shape)
        print("PMC tokens shape:", pmc_tokens.shape)
        print("Endo tokens shape:", endo_tokens.shape)

    def test_trainable_parameters(self):
        trainable_params = [name for name, param in self.adapter.named_parameters() if param.requires_grad]
        frozen_params = [name for name, param in self.adapter.named_parameters() if not param.requires_grad]

        print("Number of trainable parameters:", len(trainable_params))
        print("Number of frozen parameters:", len(frozen_params))

        self.assertGreater(len(trainable_params), 0, "There should be some trainable parameters.")
        self.assertGreater(len(frozen_params), 0, "There should be some frozen parameters.")

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
