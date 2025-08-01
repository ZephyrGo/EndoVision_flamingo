import torch
import torch.nn as nn
import unittest
from flamingo_core.flamingo import Flamingo
from flamingo_core.endo_adapter import EndoFMAdapter

class DummyLangEncoder(nn.Module):
    def forward(self, input_ids=None, attention_mask=None, labels=None, image_embeds=None, media_locations=None, **kwargs):
        B, seq_len = input_ids.shape
        logits = torch.randn(B, seq_len, 32000, device=input_ids.device)
        return type('DummyOutput', (), {'logits': logits})

    def generate(self, input_ids=None, attention_mask=None, image_embeds=None, media_locations=None, **kwargs):
        return ["mocked generated text"] * input_ids.shape[0]

class DummyTokenizer:
    token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "hello": 3, "world": 4, "test": 5}

    def __call__(self, texts, return_tensors=None):
        texts = [texts] if isinstance(texts, str) else texts
        encoded_batches = []
        for text in texts:
            ids = [self.token_to_id.get(word, 99) for word in ["<bos>"] + text.lower().split() + ["<eos>"]]
            encoded_batches.append(ids)
        max_len = max(len(ids) for ids in encoded_batches)
        padded = [ids + [0]*(max_len - len(ids)) for ids in encoded_batches]
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}

class TestEndoFlamingoIntegration(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        class DummyCfg:
            class DATA:
                TRAIN_CROP_SIZE = 224
                NUM_FRAMES = 8
            class MODEL:
                NUM_CLASSES = 0
            class TIMESFORMER:
                ATTENTION_TYPE = 'divided_space_time'
                PRETRAINED_MODEL = ''

        self.dummy_cfg = DummyCfg()
        self.adapter = EndoFMAdapter(
            cfg=self.dummy_cfg,
            endo_checkpoint_path='checkpoints/endo_fm_convert.pth',
            target_hidden_dim=4096,
            num_latents=64,
            perceiver_depth=2,
            perceiver_heads=8,
            perceiver_dim_head=64,
            freeze_endo=True
        ).to(self.device)

        self.lang_encoder = DummyLangEncoder().to(self.device)
        self.flamingo = Flamingo(
            vision_encoder=self.adapter,
            lang_encoder=self.lang_encoder,
            eoc_token_id=2,
            media_token_id=3,
            vis_dim=4096
        ).to(self.device)

        self.model = self.flamingo
        self.adapter.eval()
        self.model.eval()

    def test_visual_token_shape(self):
        B, T, C, H, W = 2, 8, 3, 224, 224
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)
        print("Adapter输入维度:", image_tensor.shape)
        visual_tokens = self.adapter(image_tensor)
        print("Adapter输出视觉tokens维度:", visual_tokens.shape)
        self.assertEqual(visual_tokens.shape, (B, T, 64, 4096))

    def test_cross_attention_integration(self):
        B, T, C, H, W = 2, 8, 3, 224, 224
        image_tensor = torch.randn(B, T, 1, C, H, W, device=self.device)
        tokenizer = DummyTokenizer()
        tokens = tokenizer(["hello world"] * B, return_tensors='pt')['input_ids'].to(self.device)
        print("Cross-attention输入维度:", image_tensor.shape, tokens.shape)
        output = self.model(image_tensor, tokens)
        print("Cross-attention输出类型:", type(output))
        self.assertIsNotNone(output)

    def test_empty_input(self):
        image_tensor = torch.randn(0, 8, 3, 224, 224, device=self.device)
        with self.assertRaises(Exception):
            self.adapter(image_tensor)

    def test_output_dim_type(self):
        print("endo_output_dim 类型:", type(self.adapter.endo_output_dim))
        self.assertIsInstance(self.adapter.endo_output_dim, int)

    def test_memory_usage(self):
        B, T, C, H, W = 2, 8, 3, 224, 224
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        _ = self.adapter(image_tensor)
        max_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        print(f"Peak GPU memory usage: {max_mem:.2f} MB")
        self.assertLess(max_mem, 3000, "GPU memory占用超过预期阈值")

    def test_trainable_parameters(self):
        trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
        frozen = [name for name, p in self.model.named_parameters() if not p.requires_grad]
        print(f"可训练参数数量: {len(trainable)}")
        print(f"冻结参数数量: {len(frozen)}")
        self.assertGreater(len(trainable), 0)
        self.assertGreater(len(frozen), 0)

    def tearDown(self):
        del self.adapter, self.lang_encoder, self.flamingo
        torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
