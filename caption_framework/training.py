"""基础训练、验证损失、checkpoint 工具。先保证正确性，不实现优化技巧。"""

from pathlib import Path

import torch
from torch.nn import functional as F


def run_epoch(model, loader, device, pad_id, optimizer=None):
    """执行一轮训练，或计算一轮 teacher-forcing 验证损失。

    loader 每次返回 (images, captions)：
      images 为预处理后的 float Tensor [B,3,H,W]；
      captions 为 long Tensor [B,T]，含 BOS、EOS，并在右侧用 PAD 补齐。
    optimizer 非空：更新参数，只能传训练集。
    optimizer 为空：不更新参数，可以传验证集。
    返回所有有效目标词的平均交叉熵，避免不同句长造成平均值偏差。

    注意：验证损失不是课程的语义相似度指标。最终选模型应另外生成验证描述，
    再调用老师提供的冻结 MiniLM 评估流程。
    """
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)
        if captions.ndim != 2 or captions.shape[1] < 2:
            raise ValueError("句子至少应包含 BOS 和 EOS")
        inputs, targets = captions[:, :-1], captions[:, 1:]
        valid_tokens = targets.ne(pad_id).sum().item()
        if valid_tokens == 0:
            raise ValueError("当前 batch 没有有效目标词，请检查数据处理")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images, inputs)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                ignore_index=pad_id, reduction="sum",
            )
            loss = loss_sum / valid_tokens
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss_sum.detach().item()
        total_tokens += valid_tokens
    if total_tokens == 0:
        raise ValueError("DataLoader 为空")
    return total_loss / total_tokens


def save_checkpoint(path, model, config, vocabulary, epoch=0, optimizer=None):
    """保存全部模块权重，以及重建模型所需的配置、词表。

    config 用普通字典保存架构名称、维度、布局、特殊词编号和图片预处理说明；
    vocabulary 建议保存为 单词->整数编号 的字典。
    不直接 pickle 整个模型对象，减少对 Python 类路径的依赖。
    optimizer 可选：恢复训练时保存；最终推理文件通常不包含它以控制体积。
    不要将数据集、MiniLM 或缓存视觉特征放进 checkpoint。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "vocabulary": vocabulary,
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    return path.stat().st_size


def load_checkpoint(path, model, device="cpu", optimizer=None):
    """加载到已构建的同结构模型，并返回配置、词表等信息。

    调用方先读取配置或使用已知配置构建对应模块，再加载权重。
    不会自动判断 encoder/decoder 类型；接入真实模型后可再增加模型工厂。
    model 应在创建 optimizer 之前移动到目标 device。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None:
        if "optimizer_state" not in checkpoint:
            raise ValueError("该文件未保存优化器状态，不能直接恢复原优化器")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
