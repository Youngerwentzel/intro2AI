"""用训练集前若干张图片观察接口输出，结果写入单独 JSON 文件。

示例（在仓库根目录运行）：
python -m caption_framework.preview_training_images --data-root "课程数据目录"

默认加载预训练 encoder，但 bridge 和临时 decoder 保持随机初始化。
本脚本不训练任何模块，不保存/覆盖正式 checkpoint，也不读取 Test/。
JSON 保存完整特征数组，便于后续直接用 json.load 读取、对比和绘图。
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from .demo import ExampleDecoder
from .encoder import build_caption_model


def tensor_record(tensor):
    """把某张图片的完整特征和摘要转成可写入 JSON 的普通类型。"""
    tensor = tensor.detach().float().cpu()
    if not torch.isfinite(tensor).all():
        raise ValueError("输出包含 NaN/Inf，请检查输入和模型")
    return {
        "shape": list(tensor.shape),
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "l2_norm": tensor.norm().item(),
        "values": tensor.tolist(),  # 完整数组，而不是截断预览
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-encoder", action="store_true",
                        help="只用于离线排错；启用后不再使用预训练权重")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count 必须大于 0")
    paths = sorted((args.data_root / "Training").glob("*.jpg"))[:args.count]
    if len(paths) != args.count:
        raise ValueError(f"Training 中不足 {args.count} 张图片，请检查数据路径")
    # 参考描述仅用于展示；词表从训练集构建，不使用 validation/test 的文字。
    with (args.data_root / "Training_Labels.csv").open(encoding="utf-8-sig", newline="") as f:
        references = {row["image_filename"]: row["description"] for row in csv.DictReader(f)}
    words = sorted({word for text in references.values()
                    for word in re.findall(r"[a-z]+(?:'[a-z]+)?|[.,!?]", text.lower())})
    vocabulary = {word: i for i, word in enumerate(["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + words)}
    inverse = {index: word for word, index in vocabulary.items()}
    torch.manual_seed(42)
    torch.set_num_threads(2)
    decoder = ExampleDecoder(len(vocabulary), visual_dim=256, hidden_dim=16)
    model, transform, config = build_caption_model(
        decoder, decoder_dim=256, pretrained=not args.random_encoder,
        output_mode="global", freeze_encoder=True,
    )
    model.eval()  # 禁止 BatchNorm 统计量更新，同时关闭训练模式的随机层
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    images = torch.stack(images)
    encoder_output = model.encoder(images)  # [10,320]，池化后的编码器输出
    bridge_output = model.bridge(encoder_output.unsqueeze(1))  # [10,1,256]
    assert encoder_output.shape == (len(paths), model.encoder.output_dim)
    assert bridge_output.shape == (len(paths), 1, 256)
    generated = model.generate(images, bos_id=1, eos_id=2, pad_id=0, max_len=20)

    result = {
        "purpose": "Temporary inference-only interface preview, not caption-quality evaluation",
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": 42,
        "pretrained_encoder": not args.random_encoder,
        "bridge_trained": False,
        "decoder_trained": False,
        "parameter_updates": 0,
        "warning": "Both bridge and decoder are random. Generated text is not expected to describe images correctly.",
        "config": config,
        "decoder_config": {"type": "ExampleDecoder", "hidden_dim": 16, "embedding_dim": 8},
        "vocabulary_source": "Training_Labels.csv only",
        "vocabulary": vocabulary,
        "max_generation_length": 20,
        "images": [],
    }
    for index, path in enumerate(paths):
        ids = generated[index]
        text = " ".join(inverse[word] for word in ids)
        row = {
            "image_filename": path.name,
            "image_path": str(path.resolve()),
            "reference_description": references[path.name],
            "generated_description": text,
            "generated_token_ids": ids,
            "reached_length_limit": len(ids) == 20,
            "encoder_output": tensor_record(encoder_output[index]),
            "bridge_output": tensor_record(bridge_output[index]),
        }
        result["images"].append(row)
        print(f"{path.name}: {text or '[empty: immediate EOS]'}")
    # 将简短对照表放在最前面，打开文件即可先看结果，再向下查看完整特征。
    result = {"preview_summary": [
        {"image_filename": row["image_filename"],
         "reference_description": row["reference_description"],
         "generated_description": row["generated_description"]}
        for row in result["images"]
    ], **result}
    destination = args.output or Path("temporary_results") / (
        "training_preview_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # x 模式避免误覆盖已有实验结果；全部内容放在这一个文件中。
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    # 写完后重新读取，确认图片数及完整特征数组确实保留。
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert len(saved["images"]) == len(paths)
    assert len(saved["images"][0]["encoder_output"]["values"]) == model.encoder.output_dim
    print("Saved:", destination.resolve())
    print("No training updates; no final checkpoint written.")


if __name__ == "__main__":
    main()
