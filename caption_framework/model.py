"""整体模型与 bridge。

本文件属于“模型搭建”工作，不依赖具体 ResNet 或 LSTM。
后续接入队友模块时，遵守 CaptionModel 的接口约定即可。
维度说明：B=批次大小，N=视觉位置数，C=编码器维度，D=解码器维度，V=词表大小。
"""

import torch
from torch import nn


class FeatureBridge(nn.Module):
    """将标准视觉特征 [B,N,C] 映射为 [B,N,D]。

    初版只使用线性层，目的在于解决两位队友模块维度不同的问题。
    不在这里写 LSTM 状态初始化或注意力：这些交给 decoder 内部处理。
    Linear 会作用在最后一维，因此不会混合 batch 或空间位置。
    """

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.projection = nn.Linear(encoder_dim, decoder_dim)

    def forward(self, features):
        if features.ndim != 3 or features.shape[-1] != self.encoder_dim:
            raise ValueError(
                f"Bridge 需要 [B,N,{self.encoder_dim}]，实际为 {tuple(features.shape)}"
            )
        return self.projection(features)


class CaptionModel(nn.Module):
    """把 encoder、bridge、decoder 组装为一个可训练模型。

    Encoder 约定：encoder(images) 返回 Tensor，支持以下三种布局：
      global: [B,C]，整张图一个向量；
      spatial: [B,C,H,W]，卷积特征图；
      tokens: [B,N,C]，已经整理好的位置序列。
    必须显式指定布局，避免仅凭形状猜错维度。

    Decoder 约定（具体实现由 decoder 同学提供）：
      forward(visual_features, input_tokens) -> logits [B,T,V]
      init_state(visual_features) -> 任意内部状态
      decode_step(previous_token, visual_features, state) -> (logits [B,V], 新状态)
    input_tokens/previous_token 是 torch.long；logits 是未经 softmax 的分数。
    decoder 必须使用传入的图片信息，训练与逐词推理的逻辑应保持一致。
    """

    def __init__(self, encoder, bridge, decoder,
                 encoder_layout="global", freeze_encoder=True):
        super().__init__()
        if encoder_layout not in {"global", "spatial", "tokens"}:
            raise ValueError("encoder_layout 必须是 global、spatial 或 tokens")
        self.encoder = encoder
        self.bridge = bridge
        self.decoder = decoder
        self.encoder_layout = encoder_layout
        self.set_encoder_frozen(freeze_encoder)

    def set_encoder_frozen(self, frozen=True):
        """冻结/解冻整个编码器；若之后解冻，需重新配置优化器以纳入新参数。"""
        self.freeze_encoder = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode=True):
        """整体进入训练模式时，仍保持冻结编码器的 BatchNorm/Dropout 为评估模式。

        requires_grad=False 只禁止参数梯度，并不会自动阻止 BatchNorm 统计量更新。
        因此冻结时需要同时调用 eval()。
        """
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def encode(self, images):
        """提取并适配视觉特征；bridge 始终位于 no_grad 之外，仍可以训练。"""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            features = self.encoder(images)
        if not isinstance(features, torch.Tensor):
            raise TypeError("Encoder 必须返回 Tensor；字典/元组请先由适配器取出特征")
        expected_ndim = {"global": 2, "spatial": 4, "tokens": 3}
        if features.ndim != expected_ndim[self.encoder_layout]:
            raise ValueError(f"编码器输出形状与 {self.encoder_layout} 布局不符")
        if self.encoder_layout == "global":
            features = features.unsqueeze(1)       # [B,C] -> [B,1,C]
        elif self.encoder_layout == "spatial":
            features = features.flatten(2).transpose(1, 2)  # [B,C,H,W] -> [B,H*W,C]
        return self.bridge(features)

    def forward(self, images, input_tokens):
        """训练入口：每个输入位置预测它后面的词，不在这里计算 loss。"""
        if input_tokens.ndim != 2 or input_tokens.dtype != torch.long:
            raise ValueError("input_tokens 必须为 [B,T] 的 torch.long 张量")
        logits = self.decoder(self.encode(images), input_tokens)
        if logits.ndim != 3 or logits.shape[:2] != input_tokens.shape:
            raise ValueError("Decoder 必须输出与输入 B、T 一致的 [B,T,V] 分数")
        return logits

    @torch.inference_mode()
    def generate(self, images, bos_id, eos_id, pad_id, max_len=30):
        """最简单的批量 greedy 生成，返回 list[list[int]]，不含 BOS/EOS/PAD。

        用于验证集生成和后续最终推理；调用方负责词表还原和 CSV 导出。
        每个句子单独跟踪结束状态，达到 max_len 也会终止，避免无限循环。
        临时切换 eval 后恢复原模式，方便在训练过程中调用。
        """
        if max_len < 1 or len({bos_id, eos_id, pad_id}) != 3:
            raise ValueError("max_len 需为正数，BOS/EOS/PAD 编号需互不相同")
        was_training = self.training
        self.eval()
        try:
            visual = self.encode(images)
            state = self.decoder.init_state(visual)
            token = torch.full((images.shape[0],), bos_id,
                               dtype=torch.long, device=images.device)
            finished = torch.zeros_like(token, dtype=torch.bool)
            sentences = [[] for _ in range(images.shape[0])]
            for _ in range(max_len):
                logits, state = self.decoder.decode_step(token, visual, state)
                logits = logits.clone()
                logits[:, [bos_id, pad_id]] = -torch.inf
                next_token = logits.argmax(dim=-1)
                for index, word in enumerate(next_token.tolist()):
                    if not finished[index].item() and word != eos_id:
                        sentences[index].append(word)
                finished |= next_token.eq(eos_id)
                token = next_token.masked_fill(finished, eos_id)
                if finished.all().item():
                    break
            return sentences
        finally:
            self.train(was_training)
