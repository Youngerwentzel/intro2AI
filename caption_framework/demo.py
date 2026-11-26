"""无需下载权重或读取课程图片的可运行示例。

运行方法（在项目根目录）：python -m caption_framework.demo
这里使用随机图片和人工句子，只检查框架能否运行，不衡量真实效果。
TinyEncoder / ExampleDecoder 是临时替身；队友模块完成后可以直接替换。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import CaptionModel, FeatureBridge
from .training import load_checkpoint, run_epoch, save_checkpoint


class TinyEncoder(nn.Module):
    """临时全局编码器，输出 [B,16]；不使用任何预训练权重。"""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        )

    def forward(self, images):
        return self.network(images)


class ExampleDecoder(nn.Module):
    """用于展示接口的单层 LSTM，不替代队友的正式 decoder 工作。

    对视觉位置取平均，用结果初始化 LSTM 的 h、c。
    forward 用真实前词训练；decode_step 用上次生成的词逐步预测。
    两种调用共享相同的 embedding、LSTM、输出层和初始化逻辑。
    """

    def __init__(self, vocab_size, visual_dim=12, hidden_dim=16, pad_id=0):
        super().__init__()
        self.to_h = nn.Linear(visual_dim, hidden_dim)
        self.to_c = nn.Linear(visual_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, 8, padding_idx=pad_id)
        self.lstm = nn.LSTM(8, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, visual_features):
        context = visual_features.mean(dim=1)
        return (
            self.to_h(context).tanh().unsqueeze(0),
            self.to_c(context).tanh().unsqueeze(0),
        )

    def forward(self, visual_features, input_tokens):
        outputs, _ = self.lstm(
            self.embedding(input_tokens), self.init_state(visual_features)
        )
        return self.output(outputs)

    def decode_step(self, previous_token, visual_features, state):
        # 此示例已经通过初始状态使用图片；注意力 decoder 可每步使用 visual_features。
        outputs, state = self.lstm(
            self.embedding(previous_token).unsqueeze(1), state
        )
        return self.output(outputs[:, 0]), state


def build_demo_model(vocab_size):
    """接真实模块时修改这里；正式版本应根据 config 构建模型。"""
    return CaptionModel(
        TinyEncoder(), FeatureBridge(16, 12), ExampleDecoder(vocab_size),
        encoder_layout="global", freeze_encoder=True,
    )


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")  # 示例固定 CPU，方便无 GPU 的成员运行
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3,
             "a": 4, "dog": 5, "runs": 6, "cat": 7}
    # 人工数据仅供联调；不是外部图文训练集，也不接触课程 Test/。
    images = torch.randn(4, 3, 16, 16)
    captions = torch.tensor([
        [1, 4, 5, 6, 2, 0], [1, 4, 7, 2, 0, 0],
        [1, 4, 5, 2, 0, 0], [1, 4, 7, 6, 2, 0],
    ], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, captions), batch_size=2)
    model = build_demo_model(len(vocab)).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # 冻结检查包含 BatchNorm 缓冲区，避免参数不动但统计量变化的问题。
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    bridge_before = model.bridge.projection.weight.detach().clone()
    train_loss = run_epoch(model, loader, device, 0, optimizer)
    assert all(torch.equal(encoder_before[k], v)
               for k, v in model.encoder.state_dict().items())
    assert not torch.equal(bridge_before, model.bridge.projection.weight)
    assert model.bridge.projection.weight.grad is not None
    assert model.decoder.output.weight.grad is not None
    validation_loss = run_epoch(model, loader, device, 0)
    # 此处复用人工数据只为接口检查，正式训练需传独立的课程验证集 loader。
    print(f"训练损失: {train_loss:.4f}; 示例评估损失: {validation_loss:.4f}")

    model.eval()
    with torch.no_grad():
        expected = model(images, captions[:, :-1])
    assert expected.shape == (4, 5, len(vocab))

    # 检查训练前向与逐词接口是否一致，减少正式接入时的接口错误。
    with torch.no_grad():
        visual = model.encode(images)
        state = model.decoder.init_state(visual)
        steps = []
        for token in captions[:, :-1].unbind(dim=1):
            logits, state = model.decoder.decode_step(token, visual, state)
            steps.append(logits)
        torch.testing.assert_close(expected, torch.stack(steps, dim=1))

    # 自动清理临时 checkpoint，不把假模型误放进正式 model/final_model.pt。
    with TemporaryDirectory(prefix="caption_demo_") as directory:
        path = Path(directory) / "demo.pt"
        config = {"model_type": "demo", "encoder_dim": 16,
                  "decoder_dim": 12, "encoder_layout": "global"}
        size = save_checkpoint(path, model, config, vocab, epoch=1, optimizer=optimizer)
        restored = build_demo_model(len(vocab)).to(device)
        restored_optimizer = torch.optim.Adam(
            [p for p in restored.parameters() if p.requires_grad], lr=1e-3
        )
        checkpoint = load_checkpoint(path, restored, device, restored_optimizer)
        assert checkpoint["vocabulary"] == vocab
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(images, captions[:, :-1]))
        # 验证恢复后还可以继续训练。
        run_epoch(restored, loader, device, 0, restored_optimizer)
        print(f"保存、加载及恢复训练检查通过；临时文件大小: {size} bytes")

    model.train()
    ids = model.generate(images, bos_id=1, eos_id=2, pad_id=0, max_len=6)
    assert model.training and not model.encoder.training
    inverse_vocab = {index: word for word, index in vocab.items()}
    print("未充分训练模型的示例输出（不代表真实效果）：")
    for sentence in ids:
        assert len(sentence) <= 6 and all(word not in (0, 1, 2) for word in sentence)
        print(" ".join(inverse_vocab[word] for word in sentence) or "[空句子]")

    # 另外验证空间特征、位置序列两种布局也能接到相同 bridge。
    for layout, shape in [("spatial", (2, 16, 3, 3)), ("tokens", (2, 9, 16))]:
        adapter_model = CaptionModel(
            nn.Identity(), FeatureBridge(16, 12), ExampleDecoder(len(vocab)),
            encoder_layout=layout,
        )
        assert adapter_model.encode(torch.randn(*shape)).shape == (2, 9, 12)
    print("基础联调全部通过。未读取训练/验证/测试图片，未下载模型权重。")


if __name__ == "__main__":
    main()
