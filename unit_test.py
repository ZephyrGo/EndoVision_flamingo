import torch
import unittest
from flamingo_core.flamingo import Flamingo
from flamingo_core.dual_visual_adapter import DualVisualAdapter
from flamingo_core.bentsao_model import BenTsaoWithFlamingoCrossAttention
from transformers import AutoTokenizer

class TestDualVisualFlamingoWithLM(unittest.TestCase):
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

        # 加载双轨视觉适配器
        self.adapter = DualVisualAdapter(
            endo_cfg=self.dummy_cfg,
            endo_checkpoint_path='checkpoints/endo_fm_convert.pth',
            pmc_checkpoint_path='checkpoints/pmc_clip.pth',
            target_hidden_dim=4096,
            num_latents=64,
            perceiver_depth=2,
            perceiver_heads=8,
            perceiver_dim_head=64,
            freeze_endo=True,
            freeze_pmc=True
        ).to(self.device)

        # 加载真实语言模型
        lang_encoder_path = 'path/to/lang_model'
        tokenizer_path = 'path/to/tokenizer'

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": ["<|endofchunk|>", "<image>"],
            "pad_token": "<PAD>"
        })

        self.lang_encoder = BenTsaoWithFlamingoCrossAttention.from_pretrained(
            lang_encoder_path,
            cross_attn_every_n_layers=1,
            finetune_lm=False
        ).to(self.device)

        self.lang_encoder.resize_token_embeddings(len(self.tokenizer))

        self.model = Flamingo(
            vision_encoder=self.adapter,
            lang_encoder=self.lang_encoder,
            eoc_token_id=self.tokenizer.convert_tokens_to_ids("<|endofchunk|>"),
            media_token_id=self.tokenizer.convert_tokens_to_ids("<image>"),
            vis_dim=4096,
            cross_attn_every_n_layers=1
        ).to(self.device)

        self.model.eval()

    def test_full_integration_forward(self):
        B, T, C, H, W = 2, 8, 3, 224, 224
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
            vision_x={'video_frames': video_tensor, 'images': image_tensor},
            lang_x=input_ids
        )

        # 验证输出logits的维度
        self.assertEqual(output.logits.shape, (B, input_ids.shape[1], self.lang_encoder.config.vocab_size))
        print("完整模型输出logits维度:", output.logits.shape)

    def test_generation(self):
        B, T, C, H, W = 2, 8, 3, 224, 224
        video_tensor = torch.randn(B, T, C, H, W, device=self.device)
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)

        inputs = self.tokenizer(
            ["<image> Findings:", "<image> Impressions:"],
            return_tensors='pt',
            padding=True
        ).to(self.device)

        generated_texts = self.model.generate(
            vision_x={'video_frames': video_tensor, 'images': image_tensor},
            lang_x=inputs['input_ids'],
            max_length=50
        )

        self.assertEqual(len(generated_texts), B)
        for idx, text in enumerate(generated_texts):
            print(f"Generated text {idx+1}: {text}")

    def test_memory_efficiency(self):
        B, T, C, H, W = 2, 8, 3, 224, 224
        video_tensor = torch.randn(B, T, C, H, W, device=self.device)
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)

        torch.cuda.reset_peak_memory_stats(self.device)
        inputs = self.tokenizer(["<image> Test memory usage"], return_tensors='pt').to(self.device)

        _ = self.model(
            vision_x={'video_frames': video_tensor, 'images': image_tensor},
            lang_x=inputs['input_ids']
        )

        max_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        print(f"Peak GPU memory usage: {max_mem:.2f} MB")
        self.assertLess(max_mem, 12000, "GPU内存占用超过预期阈值")

    def test_adapter_output_shape(self):
        B, T, C, H, W = 1, 8, 3, 224, 224
        video_tensor = torch.randn(B, T, C, H, W, device=self.device)
        image_tensor = torch.randn(B, T, C, H, W, device=self.device)

        visual_tokens = self.adapter(
            video_frames=video_tensor,
            images=image_tensor,
            enable_endo=True,
            enable_pmc=True,
            add_branch_tokens=True
        )

        expected_shape = (B, T, 130, 4096)  # 确认与你 init 参数一致(enable_endo=True, enable_pmc=True, add_branch_tokens=True)
        self.assertEqual(visual_tokens.shape, expected_shape)
        print("DualVisualAdapter输出token维度验证成功:", visual_tokens.shape)

    def test_trainable_parameters(self):
        trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
        frozen = [name for name, p in self.model.named_parameters() if not p.requires_grad]

        print("可训练参数：", len(trainable))
        print("冻结参数：", len(frozen))
        self.assertGreater(len(trainable), 0, "未找到可训练参数")
        self.assertGreater(len(frozen), 0, "未找到冻结参数")

    def tearDown(self):
        del self.adapter, self.lang_encoder, self.model
        torch.cuda.empty_cache()

if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
