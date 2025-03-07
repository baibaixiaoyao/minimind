import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer

from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # ! 结合 warmup 的余弦退火策略,每个 step 动态计算学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps
        # ! 缩放损失进行反向传
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    # ! Hugging Face Transformers 库中的一个方法，用于加载预训练的文本处理模型（Tokenizer），以便将文本数据转换为模型可以接受的输入格式
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')
    # ! 传入模型配置，并绑定到设备
    model = MiniMindLM(lm_config).to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


# ! 获取本地gpu，设置设备
def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# ! 单卡训练 python train_pretrain.py --device cuda:0 --batch_size 32
# ！多卡分布式训  torchrun --nproc_per_node 2 1-pretrain.py
# ！ 训练监督 --use_wandb  启用 WandB 可视化
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    # ! 模型输出目录
    parser.add_argument("--out_dir", type=str, default="out")
    # !  预训练数据路径
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")


    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    #！训练控制参数-训练轮次，总步数 = epochs * (数据集大小/batch_size)
    parser.add_argument("--epochs", type=int, default=1)
    #！训练控制参数-单卡批大小，实际批大小 = batch_size * accumulation_steps * GPU数量
    parser.add_argument("--batch_size", type=int, default=32)
    #！训练控制参数-基础学习率，实际学习率通过余弦退火动态调整
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    #！训练控制参数-梯度累积次数，模拟更大批次的训练，实际批大小 = batch_size * accumulation_steps * GPU数量
    parser.add_argument("--accumulation_steps", type=int, default=8)
    # ！训练控制参数-梯度裁剪阈值，用于防止梯度爆炸，torch.nn.utils.clip_grad_norm_
    parser.add_argument("--grad_clip", type=float, default=1.0)


    #！硬件相关参-设备类型，代码存在未导入 torch 的问题，支持 cpu、cuda、bfloat16
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    #！硬件相关参-浮点精度类型，可选：float32/float16/bfloat16
    parser.add_argument("--dtype", type=str, default="bfloat16")
    #！硬件相关参-数据加载线程数，建议设为 CPU 核心数的 50-75%，用于多线程加载数据
    parser.add_argument("--num_workers", type=int, default=1)


    #！分布式训练参数-启用分布式训练，需要 torchrun 启
    parser.add_argument("--ddp", action="store_true")
    #! 分布式训练参数-自动分配的本地 GPU 编号，当前进程在本节点（单机）上的 GPU 号,假设 4 张 GPU，local_rank=0,1,2,3
    parser.add_argument('--local_rank', type=int, default=-1)


    #！监控与调试-启用 WandB 监控
    parser.add_argument("--use_wandb", action="store_true")
    #！#！监控与调试-WandB 项目名称
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    #！监控与调试-学习率预热的迭代次数。预热的意思是，刚开始训练的时候，学习率很小，然后逐渐增加到预定的学习率，这样可以避免模型在初期因为学习率太高导致的不稳定。
    parser.add_argument("--warmup_iters", type=int, default=0)
    #！监控与调试-日志打印间隔（步），控制控制台输出频率
    parser.add_argument("--log_interval", type=int, default=100)
    #！监控与调试-模型保存间隔（步），定期保存检查点
    parser.add_argument("--save_interval", type=int, default=100)

    # ! 模型架构参数-模型隐藏层维度,影响模型容量
    parser.add_argument('--dim', default=512, type=int)
    # ! 模型架构参数-Transformer 层数,决定模型深度
    parser.add_argument('--n_layers', default=8, type=int)
    # ! 模型架构参数-最大序列长度,输入文本截断长
    parser.add_argument('--max_seq_len', default=512, type=int)
    # ! 模型架构参数-是否使用混合专家(MoE),稀疏激活配置
    parser.add_argument('--use_moe', default=False, type=bool)
    args = parser.parse_args()

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    # ! 自动混合精度上下文
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?  #! 当前进程在所有 GPU 训练中的编号,假设 8 张 GPU（2 机器，每台 4 张），rank=0~7
    ddp_local_rank, DEVICE = 0, "cuda:0"

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
    # ! https://blog.csdn.net/u013565133/article/details/145457047
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None
    # ! 返回模型和token配置
    model, tokenizer = init_model(lm_config)
    # ! 加载预训练数据
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    # ! 采样器只从整个数据集中加载部分数据，提高训练模型的效率或进行数据集的分批处理
    # ! 支持分布式训练
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    # ! 梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ! 支持分布式训练
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    # ！epoch 是训练过程中对整个数据集进行一轮完整的训练，过多的 epochs 可能导致过拟合，即模型在训练数据上表现很好，但在未见过的数据上的泛化能力较差。因此，在实际训练中需要找到合适的 epoch 数量。
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
