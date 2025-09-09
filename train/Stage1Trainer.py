"""
Stage 1 Training Script for EndoVision-Flamingo with Open-PMC-18M
第一阶段：使用Open-PMC-18M进行通用医学数据预训练（流式加载）
修复版本：正确处理WebDataset数据格式
"""

import sys
import os

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上级目录
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import json
import wandb
from PIL import Image
from torchvision import transforms
from typing import Dict, Optional
import webdataset as wds
import io
import time
from datetime import datetime

# 导入模型工厂
from flamingo_core.factory import create_model_for_training


class PMC18MStreamingDataset:
    """
    Open-PMC-18M 流式数据集处理类
    修复版：正确处理WebDataset的PIL Image格式
    """
    
    def __init__(
        self,
        base_url: str,
        tokenizer,
        num_shards: int = 1000,
        max_length: int = 512,
        image_size: int = 224,
        mode: str = "train",
        shuffle_buffer: int = 10000,
        use_mirror: bool = False,
        mirror_endpoint: str = "https://hf-mirror.com"
    ):
        """
        初始化流式数据集
        
        Args:
            base_url: 数据集基础URL或本地路径
            tokenizer: 文本分词器
            num_shards: 分片数量
            max_length: 最大序列长度
            image_size: 图像尺寸
            mode: train/val/test模式
            shuffle_buffer: 随机打乱缓冲区大小
            use_mirror: 是否使用中国镜像
            mirror_endpoint: 镜像站点地址
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.shuffle_buffer = shuffle_buffer
        
        # 中国镜像配置
        if use_mirror:
            os.environ['HF_ENDPOINT'] = mirror_endpoint
            # 替换URL中的域名
            if "huggingface.co" in base_url:
                base_url = base_url.replace("https://huggingface.co", mirror_endpoint)
            print(f"使用镜像站点: {mirror_endpoint}")
            print(f"实际URL: {base_url}")
        
        # 构建URL列表 - 根据实际文件格式调整
        if base_url.startswith("http"):
            # Open-PMC-18M使用的文件名格式
            self.urls = []
            for i in range(num_shards):
                # 尝试不同的文件名格式
                # 格式1: data_00000.tar
                url = f"{base_url}/data_{i:05d}.tar"
                self.urls.append(url)
        else:
            # 本地文件加载
            import glob
            tar_files = glob.glob(os.path.join(base_url, "*.tar"))
            tar_files.sort()
            self.urls = [f"file://{os.path.abspath(f)}" for f in tar_files[:num_shards]]
        
        print(f"配置 {len(self.urls)} 个数据分片")
        if self.urls:
            print(f"示例URL: {self.urls[0]}")
        
        # 图像预处理管道
        if mode == "train":
            # 训练时使用数据增强
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.48145466, 0.4578275, 0.40821073],
                    std  = [0.26862954, 0.26130258, 0.27577711]

                )
            ])
        else:
            # 验证/测试时不做增强
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.48145466, 0.4578275, 0.40821073],
                    std  = [0.26862954, 0.26130258, 0.27577711]
                )
            ])
    
    def process_image(self, image):
        """
        处理医学图像
        修复：直接处理PIL Image对象，而不是字节流
        
        Args:
            image: PIL Image对象或字节流
            
        Returns:
            处理后的图像张量
        """
        try:
            # 检查输入类型
            if isinstance(image, bytes):
                # 如果是字节流，先转换为PIL Image
                image = Image.open(io.BytesIO(image)).convert("RGB")
            elif not isinstance(image, Image.Image):
                # 如果既不是字节流也不是PIL Image，尝试转换
                print(f"未知的图像类型: {type(image)}")
                return torch.zeros(1, 3, 224, 224)
            
            # 确保是RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 应用变换
            image_tensor = self.image_transform(image)
            
            # 添加batch维度 (C, H, W) -> (1, C, H, W)
            return image_tensor.unsqueeze(0)
            
        except Exception as e:
            print(f"图像处理错误: {e}")
            # 返回黑图作为fallback
            return torch.zeros(1, 3, 224, 224)
    
    def extract_caption(self, metadata):
        """
        从元数据中提取标题/描述
        适配PMC-18M的实际数据结构
        """
        try:
            if isinstance(metadata, bytes):
                metadata = metadata.decode('utf-8')
            
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    return metadata
            
            # PMC-18M的实际字段是 "caption"
            if isinstance(metadata, dict):
                caption = metadata.get('caption', '')
                
                if not caption:
                    # 备用字段
                    caption = metadata.get('text', '') or metadata.get('description', '')
                
                # 清理文本 - 去除多余的空格和换行
                caption = ' '.join(caption.split())
                
                return caption if caption else "Medical figure from research paper."
                
        except Exception as e:
            print(f"标题提取错误: {e}")
            return "Medical image without description."
    
    def process_sample(self, sample):
        """
        处理单个样本
        修复：适配WebDataset的实际输出格式
        
        Args:
            sample: WebDataset样本元组
            
        Returns:
            处理后的字典
        """
        try:
            # WebDataset通常返回元组 (image, metadata)
            if isinstance(sample, tuple) and len(sample) >= 2:
                image, metadata = sample[0], sample[1]
            elif isinstance(sample, dict):
                # 有时候可能是字典格式
                image = sample.get('jpg') or sample.get('png') or sample.get('jpeg')
                metadata = sample.get('json') or sample.get('txt') or sample.get('caption')
            else:
                print(f"未知的样本格式: {type(sample)}")
                raise ValueError("Invalid sample format")
            
            # 处理图像
            image_tensor = self.process_image(image)
            
            # 处理文本
            caption = self.extract_caption(metadata)
            
            # PMC-18M的caption通常很长，可能需要智能截断
            if len(caption) > 2000:  # 如果太长
                # 保留开头和结尾部分（通常包含关键信息）
                words = caption.split()
                if len(words) > 200:
                    caption = ' '.join(words[:150] + ['...'] + words[-50:])
            
            # 构建输入 - 使用更简洁的prompt以留出空间给caption
            input_text = "<image> Describe this figure:"  # 更短的prompt
            target_text = caption
            
            full_text = f"{input_text} {target_text} <|endofchunk|>"
            
            # Tokenization
            encodings = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 创建标签
            labels = encodings.input_ids.clone()
            
            # 计算输入部分的长度
            input_encoding = self.tokenizer(
                input_text,
                add_special_tokens=True,
                return_tensors="pt"
            )
            input_len = input_encoding.input_ids.shape[1]
            
            # 屏蔽输入和padding
            labels[0, :input_len] = -100
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            
            # 统计有效标签
            valid_labels = (labels != -100).sum()
            
            # 如果有效标签太少，说明文本被截断太多
            # 质量检查 - 跳过低质量样本
            if valid_labels < 20:
                print(f"跳过低质量样本: 只有 {valid_labels} 个有效标签")
                return None
            
            return {
                "images": image_tensor,
                "input_ids": encodings.input_ids.squeeze(0),
                "attention_mask": encodings.attention_mask.squeeze(0),
                "labels": labels.squeeze(0)
            }
            
        except Exception as e:
            print(f"样本处理错误: {e}")
            # 返回一个有效的dummy数据
            dummy_text = "<image> Describe the medical findings: This is a placeholder medical image description used for error handling. <|endofchunk|>"
            dummy_encoding = self.tokenizer(
                dummy_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            dummy_labels = dummy_encoding.input_ids.clone()
            # 只将前10个token设为-100，确保有足够的有效标签
            dummy_labels[0, :10] = -100
            dummy_labels[dummy_labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "images": torch.zeros(1, 3, self.image_transform.transforms[0].size, self.image_transform.transforms[0].size),
                "input_ids": dummy_encoding.input_ids.squeeze(0),
                "attention_mask": dummy_encoding.attention_mask.squeeze(0),
                "labels": dummy_labels.squeeze(0)
            }
    
    def create_webdataset(self, batch_size: int = 32, num_workers: int = 4):
        """
        创建WebDataset数据加载器
        修复：更健壮的错误处理和正确的解码顺序
        
        Args:
            batch_size: 批次大小
            num_workers: 工作进程数
            
        Returns:
            WebDataset DataLoader
        """
        print(f"创建WebDataset，batch_size={batch_size}, num_workers={num_workers}")

        # 初始化统计
        self.label_stats = []
        
        # 过滤函数
        def filter_none(sample):
            """过滤掉None样本"""
            return sample is not None
        
        # 创建WebDataset管道
        dataset = (
            wds.WebDataset(
                self.urls,
                shardshuffle=1000 if self.mode == "train" else False,
                handler=wds.warn_and_continue,  # 错误处理：跳过损坏的样本
                nodesplitter=wds.split_by_node,  # 多节点训练支持
            )
            .shuffle(self.shuffle_buffer if self.mode == "train" else 0)
            # 关键修改：decode("pil")会将图像解码为PIL Image对象
            .decode("pil", handler=wds.warn_and_continue)
            # 适配可能的文件扩展名
            .to_tuple("jpg;png;jpeg;JPG;PNG;JPEG", "json;txt;caption", handler=wds.warn_and_continue)
            .select(filter_none)  # 过滤掉None值
            .map(self.process_sample, handler=wds.warn_and_continue)
            .batched(batch_size, partial=False)
        )
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=None,  # WebDataset已经处理了批次
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return dataloader


class Stage1PMCTrainer:
    """
    第一阶段训练器 - 适配新的BioMedicalLlamaFlamingo架构
    增强：支持fp16混合精度，优化显存使用
    """

    def __init__(
        self,
        model,
        tokenizer,
        config
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.bad_epochs = 0
        self.best_recent_loss = float('inf')
        self.patience = 3

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"当前使用的 GPU ID: {torch.cuda.current_device()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # 重要：模型已经在factory中设置为fp16，这里不再移动
        # self.model已经在正确的设备和精度上

        # [数据集初始化代码保持不变...]
        print("\n初始化Open-PMC-18M流式数据集...")

        # PMC-18M配置
        pmc_train_config = {
            "base_url": config.get("pmc_base_url"),
            "num_shards": config.get("num_shards", 100),
            "max_length": config["max_length"],
            "image_size": config.get("image_size", 224),
            "mode": "train",
            "shuffle_buffer": config.get("shuffle_buffer", 10000),
            "use_mirror": config.get("use_chinese_mirror", False),
            "mirror_endpoint": config.get("mirror_endpoint", "https://hf-mirror.com")
        }

        pmc_val_config = pmc_train_config.copy()
        pmc_val_config["mode"] = "val"
        pmc_val_config["num_shards"] = min(10, pmc_train_config["num_shards"])

        # 创建数据集
        print("创建训练数据集...")
        self.train_dataset = PMC18MStreamingDataset(
            **pmc_train_config,
            tokenizer=tokenizer
        )

        print("创建验证数据集...")
        self.val_dataset = PMC18MStreamingDataset(
            **pmc_val_config,
            tokenizer=tokenizer
        )

        # 创建流式数据加载器
        self.train_loader = self.train_dataset.create_webdataset(
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 4)
        )
        self.val_loader = self.val_dataset.create_webdataset(
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 2)
        )

        # 估算训练步数
        estimated_samples_per_shard = 1500
        estimated_total_samples = pmc_train_config["num_shards"] * estimated_samples_per_shard
        self.steps_per_epoch = estimated_total_samples // config["batch_size"]

        print(f"\n训练配置:")
        print(f"  - 估算训练步数: {self.steps_per_epoch} steps/epoch")
        print(f"  - Batch size: {config['batch_size']}")
        print(f"  - 梯度累积步数: {config.get('gradient_accumulation_steps', 1)}")
        print(f"  - 有效batch size: {config['batch_size'] * config.get('gradient_accumulation_steps', 1)}")

        # 优化器设置 - 只传递可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(f"\n模型参数:")
        print(f"  - 总参数量: {total_params / 1e6:.2f}M")
        print(f"  - 可训练参数: {trainable_params_count / 1e6:.2f}M")
        print(f"  - 训练比例: {100 * trainable_params_count / total_params:.2f}%")

        # 创建优化器
        if config.get("use_8bit_adam", False):
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=config["learning_rate"],
                    weight_decay=config.get("weight_decay", 0.01),
                    betas=(0.9, 0.95),
                    eps=1e-5  # 增加epsilon以提高数值稳定性
                )
                print("使用8-bit AdamW优化器")
            except ImportError:
                print("bitsandbytes未安装，使用标准AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=config["learning_rate"],
                    weight_decay=config.get("weight_decay", 0.01)
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config["learning_rate"],
                weight_decay=config.get("weight_decay", 0.01)
            )

        # 学习率调度器
        num_training_steps = self.steps_per_epoch * config["num_epochs"]
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get("warmup_steps", 500),
            num_training_steps=num_training_steps,
            num_cycles=0.5  # 半个余弦周期，确保学习率持续下降
        )

        # 混合精度训练设置
        self.use_fp16 = config.get("fp16", True)
        if self.use_fp16:
            # 使用默认的GradScaler设置
            self.scaler = GradScaler(
                init_scale=2.**16,
                growth_factor=2.0,
                backoff_factor=0.25,
                growth_interval=1000,
                enabled=True
            )
            print("启用fp16混合精度训练（GradScaler）")
        else:
            self.scaler = None
            print("警告：未启用fp16，可能会导致显存不足！")

        # 梯度累积
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # 初始化wandb
        if config.get("use_wandb", False):
            wandb.init(
                project="endovision-flamingo",
                name=f"stage1_pmc18m_{config.get('exp_name', 'default')}",
                config=config
            )

        # 训练状态跟踪
        self.global_step = 0
        self.start_epoch = 0

        # 尝试恢复checkpoint
        self.resume_from_checkpoint()

    def train_epoch(self, epoch):
        """
        训练一个epoch - 使用混合精度
        """
        self.model.train()
        total_loss = 0
        num_steps = 0
        
        # 创建进度条
        progress_bar = tqdm(
            range(self.steps_per_epoch),
            desc=f"Epoch {epoch}"
        )
        
        # 流式训练循环
        data_iter = iter(self.train_loader)
        last_save_step = self.global_step
        
        for step in progress_bar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                try:
                    batch = next(data_iter)
                except:
                    print("无法获取数据，跳过此步")
                    continue
            except Exception as e:
                print(f"数据加载错误: {e}")
                continue
                
            try:
                # 移动到设备
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 混合精度前向传播
                if self.use_fp16 and self.scaler is not None:
                    # 使用autocast上下文
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(
                            vision_x=images,
                            lang_x=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss / self.gradient_accumulation_steps
                    
                    # 缩放损失并反向传播
                    self.scaler.scale(loss).backward()
                    
                    # 梯度累积完成后
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        # 先unscale梯度
                        self.scaler.unscale_(self.optimizer)
                        
                        # fp32下的梯度裁剪
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.get("max_grad_norm", 1.0)
                        )
                        
                        # 优化器步骤
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # 调度器和清零
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        self.global_step += 1
                    else:
                        grad_norm = 0.0  # 累积中不计算梯度范数
                        
                else:
                    # 全精度训练（不推荐）
                    outputs = self.model(
                        vision_x=images,
                        lang_x=input_ids, 
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.get("max_grad_norm", 1.0)
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.global_step += 1
                    else:
                        grad_norm = 0.0
                        
                # 记录损失
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_steps += 1
                avg_loss = total_loss / num_steps
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "step": self.global_step,
                    "grad_norm": f"{grad_norm:.2f}" if grad_norm > 0 else "N/A"
                })
                
                # 记录到wandb
                if self.config.get("use_wandb", False) and self.global_step % 10 == 0:
                    wandb.log({
                        "train_loss": loss.item() * self.gradient_accumulation_steps,
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "grad_norm": grad_norm if grad_norm > 0 else 0
                    })
                    
                # # 显存监控
                # if step % 100 == 0:
                #     self.monitor_memory()
                    
                # 定期保存checkpoint（只保存可训练参数）
                if self.global_step > 0 and self.global_step - last_save_step >= 1000:
                    self.save_checkpoint(
                        epoch,
                        avg_loss,
                        os.path.join(self.config["output_dir"], f"checkpoint_step_{self.global_step}.pt"),
                        save_full_model=False  # 只保存可训练参数
                    )
                    last_save_step = self.global_step

                # 每200步检查一次

                if self.global_step % 200 == 0:
                    if avg_loss > self.best_recent_loss:
                        self.bad_epochs += 1
                        if self.bad_epochs >= self.patience:
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            print(f"降低学习率到: {param_group['lr']}")
                            self.bad_epochs = 0
                    else:
                        self.best_recent_loss = avg_loss
                        self.bad_epochs = 0
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM错误，跳过此批次")
                    if self.scaler is not None:
                        self.scaler.update()  # 重置scaler状态
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                # elif "Expected all tensors to be on the same device" in error_msg:
                #     print(f"设备不匹配错误: {error_msg[:100]}...")
                #     # 打印调试信息
                #     self.debug_device_mismatch()
                else:
                    print(f"运行时错误: {e}")
                    # 重置状态
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    
                    # 【重要】在异常后不要调用scaler.update()
                    # 只在成功的迭代后更新scaler
                    continue
            except Exception as e:
                print(f"训练步骤错误: {e}")
                continue
                
            # 达到预定步数后停止
            if num_steps >= self.steps_per_epoch:
                break
        
                
        return total_loss / num_steps if num_steps > 0 else float('inf')

    @torch.no_grad()
    def validate(self, epoch, max_steps=100):
        """验证（流式，限制步数）- 使用fp16推理"""
        self.model.eval()
        total_loss = 0
        num_steps = 0

        progress_bar = tqdm(range(max_steps), desc="Validation")
        data_iter = iter(self.val_loader)

        for step in progress_bar:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"验证数据加载错误: {e}")
                continue

            try:
                # 始终使用fp16进行验证
                images = batch["images"].to(self.device, dtype=torch.float16)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(dtype=torch.float16):
                    outputs = self.model(
                        vision_x=images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                total_loss += loss.item()
                num_steps += 1

                # 更新进度条
                avg_loss = total_loss / num_steps
                progress_bar.set_postfix({"val_loss": f"{avg_loss:.4f}"})

            except Exception as e:
                print(f"验证步骤错误: {e}")
                continue

            if num_steps >= max_steps:
                break

        avg_loss = total_loss / num_steps if num_steps > 0 else float('inf')

        if self.config.get("use_wandb", False):
            wandb.log({
                "val_loss": avg_loss,
                "epoch": epoch
            })

        return avg_loss
    
    def resume_from_checkpoint(self):
        """从checkpoint恢复 - 支持多种checkpoint格式"""
        checkpoint_dir = self.config["output_dir"]
        if not os.path.exists(checkpoint_dir):
            print(f"检查点目录不存在: {checkpoint_dir}")
            return
        
        # 查找所有可能的checkpoint文件
        all_checkpoints = []
        
        # 1. 查找epoch checkpoints
        epoch_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                            if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
        for ckpt in epoch_checkpoints:
            try:
                epoch = int(ckpt.split("_")[-1].replace(".pt", ""))
                all_checkpoints.append({
                    'type': 'epoch',
                    'value': epoch,
                    'filename': ckpt,
                    'path': os.path.join(checkpoint_dir, ckpt)
                })
            except:
                continue
        
        # 2. 查找step checkpoints  
        step_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                        if f.startswith("checkpoint_step_") and f.endswith(".pt")]
        for ckpt in step_checkpoints:
            try:
                step = int(ckpt.split("_")[-1].replace(".pt", ""))
                all_checkpoints.append({
                    'type': 'step',
                    'value': step,
                    'filename': ckpt,
                    'path': os.path.join(checkpoint_dir, ckpt)
                })
            except:
                continue
        
        # 3. 查找interrupted checkpoints
        interrupted_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                                if f.startswith("interrupted_epoch_") and f.endswith(".pt")]
        for ckpt in interrupted_checkpoints:
            try:
                epoch = int(ckpt.split("_")[-1].replace(".pt", ""))
                all_checkpoints.append({
                    'type': 'interrupted',
                    'value': epoch,
                    'filename': ckpt,
                    'path': os.path.join(checkpoint_dir, ckpt)
                })
            except:
                continue
        
        if not all_checkpoints:
            print("没有找到任何checkpoint文件")
            return
        
        # 打印找到的所有checkpoints
        print(f"\n发现 {len(all_checkpoints)} 个checkpoint文件:")
        for ckpt in all_checkpoints:
            print(f"  - {ckpt['filename']} (类型: {ckpt['type']}, 值: {ckpt['value']})")
        
        # 选择最新的checkpoint
        # 优先级: interrupted > step > epoch
        # 在同类型中选择值最大的
        latest_checkpoint = None
        
        # 先按类型分组
        by_type = {}
        for ckpt in all_checkpoints:
            if ckpt['type'] not in by_type:
                by_type[ckpt['type']] = []
            by_type[ckpt['type']].append(ckpt)
        
        # 按优先级选择
        if 'interrupted' in by_type:
            latest_checkpoint = max(by_type['interrupted'], key=lambda x: x['value'])
            print(f"\n选择中断的checkpoint: {latest_checkpoint['filename']}")
        elif 'step' in by_type:
            latest_checkpoint = max(by_type['step'], key=lambda x: x['value'])
            print(f"\n选择最新的step checkpoint: {latest_checkpoint['filename']}")
        elif 'epoch' in by_type:
            latest_checkpoint = max(by_type['epoch'], key=lambda x: x['value'])
            print(f"\n选择最新的epoch checkpoint: {latest_checkpoint['filename']}")
        
        if latest_checkpoint is None:
            print("无法选择合适的checkpoint")
            return
        
        # 加载选中的checkpoint
        checkpoint_path = latest_checkpoint['path']
        print(f"正在加载: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 检查checkpoint类型
            is_full_model = checkpoint.get("is_full_model", True)
            
            if is_full_model:
                # 完整模型checkpoint
                print("加载完整模型权重...")
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                # 只有可训练参数的checkpoint
                print("加载可训练参数...")
                trainable_state_dict = checkpoint["trainable_state_dict"]
                
                # 更新模型中的可训练参数
                model_state_dict = self.model.state_dict()
                for name, param in trainable_state_dict.items():
                    if name in model_state_dict:
                        model_state_dict[name].copy_(param)
                    else:
                        print(f"警告：参数 {name} 在模型中不存在")
                
                print(f"成功加载 {len(trainable_state_dict)} 个可训练参数")
            
            # 加载优化器状态
            print("加载优化器状态...")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # 加载调度器状态
            print("加载学习率调度器状态...")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            for g in self.optimizer.param_groups:
                g["lr"] = self.config["learning_rate"]

            
            # 恢复训练状态
            self.start_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            
            # 如果是中断的checkpoint，不增加epoch
            if latest_checkpoint['type'] == 'interrupted':
                self.start_epoch = checkpoint.get("epoch", 0)  # 保持在同一个epoch
                print(f"从中断的Epoch {self.start_epoch} 继续训练")
            else:
                print(f"从Epoch {self.start_epoch} 开始训练")
            
            print(f"全局步数: {self.global_step}")
            print(f"上次训练损失: {checkpoint.get('loss', 'N/A')}")
            print(f"Checkpoint时间: {checkpoint.get('timestamp', 'N/A')}")
            
            print("\n✓ 成功恢复训练状态！")
            
        except Exception as e:
            print(f"\n✗ 加载checkpoint失败: {e}")
            print("详细错误信息:")
            import traceback
            traceback.print_exc()
            
            # 询问是否要从头开始
            response = input("\n是否要从头开始训练? (y/n): ")
            if response.lower() != 'y':
                raise RuntimeError("用户取消训练")
            print("将从头开始训练...")
    
    def monitor_memory(self):
        """监控GPU显存"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nGPU内存: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
    
    def save_checkpoint(self, epoch, loss, path, save_full_model=False, save_both=False):
        """
        灵活的checkpoint保存策略
        
        Args:
            path: 保存路径
            save_full_model: 是否保存完整模型
            save_both: 是否同时保存完整版和轻量版（用于最终模型）
        """
        if save_both:
            # 同时保存两个版本
            base_path = path.replace('.pt', '')
            
            # 1. 保存完整版本
            full_path = f"{base_path}_full.pt"
            self._save_single_checkpoint(
                epoch, loss, full_path, 
                save_full_model=True
            )
            
            # 2. 保存轻量版本
            light_path = f"{base_path}_light.pt"
            self._save_single_checkpoint(
                epoch, loss, light_path,
                save_full_model=False
            )
            
            print(f"已保存两个版本的checkpoint:")
            print(f"  - 完整版: {full_path}")
            print(f"  - 轻量版: {light_path}")
            
        else:
            # 保存单个版本
            self._save_single_checkpoint(epoch, loss, path, save_full_model)
    
    def _save_single_checkpoint(self, epoch, loss, path, save_full_model=False):
        """
        实际的保存逻辑
        """
        # 获取可训练参数
        trainable_state_dict = {
            name: param.data.cpu().clone()  # 保存到CPU以节省GPU内存
            for name, param in self.model.named_parameters() 
            if param.requires_grad
        }
        
        # 基础checkpoint信息
        base_checkpoint = {
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
            "global_step": self.global_step,
            "timestamp": datetime.now().isoformat(),
            "pytorch_version": torch.__version__,
            "trainable_params_count": len(trainable_state_dict),
            "trainable_params_names": list(trainable_state_dict.keys())
        }
        
        if save_full_model:
            # 保存完整模型
            checkpoint = {
                **base_checkpoint,
                "model_state_dict": self.model.state_dict(),
                "is_full_model": True,
                "checkpoint_type": "full"
            }
        else:
            # 只保存可训练参数
            checkpoint = {
                **base_checkpoint,
                "trainable_state_dict": trainable_state_dict,
                "is_full_model": False,
                "checkpoint_type": "light"
            }
        
        # 安全保存（先保存到临时文件）
        temp_path = path + ".tmp"
        torch.save(checkpoint, temp_path)
        
        import shutil
        shutil.move(temp_path, path)
        
        # 显示文件信息
        file_size = os.path.getsize(path) / (1024**3)  # GB
        print(f"✓ Checkpoint已保存: {os.path.basename(path)}")
        print(f"  - 大小: {file_size:.2f} GB")
        print(f"  - 类型: {checkpoint['checkpoint_type']}")
        print(f"  - 可训练参数: {checkpoint['trainable_params_count']}个")
    
    def train(self):
        """
        完整训练流程
        增强：更好的错误恢复和checkpoint管理
        """
        best_val_loss = float('inf')
        
        print("\n" + "="*50)
        print("开始训练")
        print("="*50)
        
        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            epoch_start_time = time.time()
            
            try:
                # 训练
                train_loss = self.train_epoch(epoch)
                print(f"训练损失: {train_loss:.4f}")
                
                # 验证
                val_loss = self.validate(epoch)
                print(f"验证损失: {val_loss:.4f}")
                
                # 保存最佳模型（同时保存两个版本）
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        epoch,
                        val_loss,
                        os.path.join(self.config["output_dir"], "best_model.pt"),
                        save_both=True  # 同时保存两个版本
                    )
                    print(f"新的最佳模型 (val_loss: {val_loss:.4f})")
                
                # 每个epoch都保存checkpoint（重要！）只保存轻量版
                self.save_checkpoint(
                    epoch,
                    val_loss,
                    os.path.join(self.config["output_dir"], f"checkpoint_epoch_{epoch+1}.pt")
                )
                
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch时间: {epoch_time/60:.2f} 分钟")
                
                # 估算剩余时间
                remaining_epochs = self.config["num_epochs"] - epoch - 1
                if remaining_epochs > 0:
                    estimated_time = (epoch_time * remaining_epochs) / 3600
                    print(f"预计剩余时间: {estimated_time:.2f} 小时")
                
            except KeyboardInterrupt:
                print("\n训练被用户中断")
                self.save_checkpoint(
                    epoch,
                    train_loss if 'train_loss' in locals() else float('inf'),
                    os.path.join(self.config["output_dir"], f"interrupted_epoch_{epoch+1}.pt")
                )
                print("已保存中断checkpoint")
                break
                
            except Exception as e:
                print(f"\nEpoch {epoch+1} 出现错误: {e}")
                print("尝试继续下一个epoch...")
                continue

        # 训练结束时保存最终模型（两个版本）
        print("\n保存最终模型...")
        self.save_checkpoint(
            self.config["num_epochs"] - 1,
            val_loss,
            os.path.join(self.config["output_dir"], "final_model.pt"),
            save_both=True
        )
        
        print("\n" + "="*50)
        print("训练完成！")
        print(f"最佳验证损失: {best_val_loss:.4f}")
        print("="*50)

def test_gradient_flow(model, tokenizer, device, amp_dtype=None):
    """测试模型是否能产生梯度"""
    print("创建测试数据...")
    
    # 创建一个简单的测试输入
    test_text = "<image> Describe the medical findings: This is a test medical image. <|endofchunk|>"
    inputs = tokenizer(test_text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    
    # 创建虚拟图像
    test_image = torch.randn(1, 1, 3, 224, 224).to(device)
    
    # 创建标签
    labels = inputs.input_ids.clone()
    labels[0, :10] = -100  # 屏蔽前10个token
    
    # 移动到设备
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    labels = labels.to(device)
    
    print("执行前向传播...")
    model.train()  # 确保在训练模式
    
    try:
        if amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                outputs = model(
                    vision_x=test_image,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
        else:
            outputs = model(
                vision_x=test_image,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        print(f"Loss值: {loss.item():.4f}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        
        if not loss.requires_grad:
            print("错误：Loss不需要梯度！")
            return False
        
        print("执行反向传播...")
        loss.backward()
        
        # 检查梯度
        has_gradient = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    print(f"参数 {name} 有梯度，norm={grad_norm:.6f}")
                    has_gradient = True
                    break
        
        if has_gradient:
            print("✓ 梯度流测试通过！")
        else:
            print("✗ 没有检测到梯度！")
        
        # 清理梯度
        model.zero_grad()
        
        return has_gradient
        
    except Exception as e:
        print(f"梯度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主训练函数 - 适配新架构"""

    # 训练配置 - 优化fp16设置
    config = {
        # 模型路径
        "base_model_path": "/home/zzk01/lqy_proj/EndoVision_flamingo/checkpoints/models--ContactDoctor--Bio-Medical-Llama-3-8B/snapshots/d71486ab8c920f34afe37ec8a21c2035b83d7b2d",
        "endo_checkpoint_path": "/home/zzk01/lqy_proj/EndoVision_flamingo/checkpoints/endo_fm_convert.pth",
        "pmc_checkpoint_path": "/home/zzk01/lqy_proj/EndoVision_flamingo/checkpoints/pmc_clip_visual_only.pt",

        # Open-PMC-18M配置
        "pmc_base_url": "https://hf-mirror.com/datasets/vector-institute/open-pmc-18m/resolve/main",
        "num_shards": 200,
        "use_chinese_mirror": True,
        "mirror_endpoint": "https://hf-mirror.com",
        "shuffle_buffer": 5000,

        # 训练参数 - 针对A6000 48GB优化
        "batch_size": 2,  # 小批次以避免OOM
        "gradient_accumulation_steps": 4,  # 有效batch size = 8
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "num_epochs": 5,
        "warmup_steps": 2000,
        "max_grad_norm": 0.25,
        "max_length": 1024,
        "image_size": 224,

        # 优化设置
        "fp16": True,  # fp16混合精度
        "use_8bit_adam": True,  # 8-bit优化器节省显存
        "use_mixed_precision": True,  # 混合精度
        "use_gradient_checkpointing": True,  # 梯度检查点

        # 保存设置
        "output_dir": "./checkpoints/stage1_pmc18m",
        "save_every": 1,

        # 其他
        "num_workers": 2,
        "use_wandb": False,
        "exp_name": "stage1_pmc18m_new_arch"
    }

    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)

    print("="*50)
    print("EndoVision-Flamingo Stage 1 Training")
    print("Architecture: BioMedicalLlamaFlamingo (New)")
    print("Dataset: Open-PMC-18M (Streaming)")
    print("="*50)

    # 打印配置
    print("\n训练配置:")
    for key, value in config.items():
        if not key.endswith("_path"):
            print(f"  {key}: {value}")

    # 创建模型 - 使用新的factory
    print("\n创建模型（新架构）...")
    try:
        model, tokenizer = create_model_for_training(
            base_model_path=config["base_model_path"],
            endo_checkpoint_path=config["endo_checkpoint_path"],
            pmc_checkpoint_path=config["pmc_checkpoint_path"],
            stage="stage1",
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
            # dtype=torch.float16,  # Base model in FP16, trainable in FP32
            use_mixed_precision=config.get("use_mixed_precision", True)  # Enable mixed precision
        )
        print("模型创建成功（使用fp16混合精度）")
        
        # 验证模型精度设置
        print("\n验证模型精度设置:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: dtype={param.dtype}, requires_grad={param.requires_grad}")
                break  # 只打印第一个可训练参数作为示例
                
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建训练器
    print("\n初始化训练器...")
    try:
        trainer = Stage1PMCTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config
        )
        print("训练器初始化成功")
    except Exception as e:
        print(f"训练器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 开始训练
    print("\n开始使用Open-PMC-18M进行流式训练...")
    print("提示：")
    print("  - 使用fp16混合精度以节省显存")
    print("  - 启用梯度检查点进一步减少显存使用")
    print("  - 可以使用Ctrl+C安全中断训练")
    print("\n")

    try:
        trainer.train()
    except Exception as e:
        print(f"\n训练过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查错误信息并重新运行")

    print("\n训练脚本执行完成！")


if __name__ == "__main__":
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 设置PyTorch以优化显存使用
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # 设置随机种子
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 运行主函数
    main()