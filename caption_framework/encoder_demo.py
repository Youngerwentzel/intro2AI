"""真实 EfficientNet 结构的基础联调，不需要等正式 decoder 完成。

项目根目录运行：python -m caption_framework.encoder_demo
默认随机初始化编码器，仅验证结构；--pretrained 才会加载/下载 ImageNet 权重。
--image 可提供一张训练图片，默认用随机张量，不接触课程 Test/。
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from .demo import ExampleDecoder
from .encoder import build_caption_model
from .training import load_checkpoint, run_epoch, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true", help="首次使用可能下载权重")
    parser.add_argument("--output-mode", choices=["global", "spatial"], default="global")
    parser.add_argument("--image", type=Path, help="可选训练图片路径，勿使用课程测试图片")
    parser.add_argument("--device", default="cpu", help="cpu/cuda/mps，按实际环境填写")
    args = parser.parse_args()
    torch.manual_seed(42)
    # 这是 CPU 小批量联调，限制线程以避免大量线程调度开销。
    torch.set_num_threads(2)
    device = torch.device(args.device)
    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "a": 4, "dog": 5}
    model, transform, config = build_caption_model(
        ExampleDecoder(len(vocab), visual_dim=256), decoder_dim=256,
        pretrained=args.pretrained, output_mode=args.output_mode,
    )
    model.to(device)
    if args.image:
        with Image.open(args.image) as image:
            images = transform(image.convert("RGB")).unsqueeze(0).to(device)
    else:
        images = torch.randn(1, *model.encoder.data_config["input_size"], device=device)
    captions = torch.tensor([[1, 4, 5, 2]], device=device)

    model.eval()
    with torch.no_grad():
        raw = model.encoder(images)
        visual = model.encode(images)
    print("预训练权重:", args.pretrained)
    print("编码器配置:", config)
    print("编码器输出:", tuple(raw.shape), "Bridge 输出:", tuple(visual.shape))
    assert visual.shape[-1] == 256
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    before = model.bridge.projection.weight.detach().clone()
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    loss = run_epoch(model, [(images, captions)], device, 0, optimizer)
    assert not torch.equal(before, model.bridge.projection.weight)
    assert model.decoder.output.weight.grad is not None
    assert all(torch.equal(value, model.encoder.state_dict()[key])
               for key, value in encoder_before.items())
    model.eval()
    with torch.no_grad():
        expected = model(images, captions[:, :-1])
    # 补齐临时 decoder 的构造信息。正式 decoder 应保存自己的全部架构参数。
    config["decoder"] = {"name": "ExampleDecoder", "hidden_dim": 16, "vocab_size": len(vocab)}
    with TemporaryDirectory(prefix="efficientnet_smoke_") as directory:
        path = Path(directory) / "smoke.pt"
        size = save_checkpoint(path, model, config, vocab)
        # 保存文件已包含整个编码器，不要在恢复模型时重新下载预训练权重。
        restored, _, _ = build_caption_model(
            ExampleDecoder(len(vocab), visual_dim=256), 256,
            pretrained=False, output_mode=args.output_mode,
        )
        restored.to(device)
        load_checkpoint(path, restored, device)
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(images, captions[:, :-1]))
        assert size < 200_000_000
        print("联调 checkpoint 字节数:", size)
    result = model.generate(images, bos_id=1, eos_id=2, pad_id=0, max_len=5)
    print("示例损失:", loss, "未训练充分的输出编号:", result)
    print("真实编码器结构接入、梯度、冻结和保存加载检查通过。")


if __name__ == "__main__":
    main()
