"""将队友 encoder_test.py 中的 EfficientNet 接入整体模型。

原脚本适合独立检查特征，但导入时会读硬编码路径、下载权重并执行检索。
本模块不导入该脚本，只复用它选定的模型名及最后一层特征提取方式。
"""

from torch import nn

ENCODER_NAME = "tf_efficientnet_b0.in1k"


class EfficientNetEncoder(nn.Module):
    """输入为预处理后的 RGB 浮点张量 [B,3,H,W]。

    output_mode='global'：最后一层空间平均，输出 [B,C]，与队友 embed 一致。
    output_mode='spatial'：保留 [B,C,h,w]，供后续注意力 decoder 使用。
    C 由 timm.feature_info 实际读取，不使用分类头的 num_features 猜测。
    pretrained=True 用于首次训练；加载整个 checkpoint 时设为 False，避免重下载。
    构造函数不冻结参数，冻结/微调统一由 CaptionModel 管理。
    """

    def __init__(self, pretrained=True, output_mode="global"):
        super().__init__()
        if output_mode not in {"global", "spatial"}:
            raise ValueError("output_mode 必须为 global 或 spatial")
        # 延迟导入：不安装 timm 仍可运行原有 TinyEncoder 示例。
        import timm
        from timm.data import resolve_data_config

        self.backbone = timm.create_model(
            ENCODER_NAME, pretrained=pretrained,
            features_only=True, out_indices=(4,),
        )
        self.output_mode = output_mode
        self.output_dim = self.backbone.feature_info.channels()[-1]
        # 用同一模型的配置生成预处理，保证训练、验证、推理输入规则一致。
        self.data_config = resolve_data_config({}, model=self.backbone)

    def build_transform(self):
        """返回确定性的图片预处理；在 Dataset 的 __getitem__ 中调用。

        用法：transform(Image.open(path).convert('RGB'))。
        timm 会按权重配置处理尺寸、插值、裁剪和归一化；初版不做随机增强。
        这与旧脚本直接 Resize 到方形略有区别，避免拉伸图片比例。
        后续对照实验必须统一预处理，不能混用旧特征缓存和新预处理。
        """
        from timm.data import create_transform
        return create_transform(**self.data_config, is_training=False)

    def forward(self, images):
        # features_only=True 返回列表；只取最后一层，转为框架要求的 Tensor。
        features = self.backbone(images)[-1]
        if self.output_mode == "global":
            return features.mean(dim=(2, 3))
        return features


def build_caption_model(decoder, decoder_dim, pretrained=True,
                        output_mode="global", freeze_encoder=True):
    """把真实 encoder、维度适配 bridge、队友 decoder 组装起来。

    decoder 必须遵守 model.py 中 forward/init_state/decode_step 接口。
    decoder_dim 是它接收的视觉特征维度，不一定等于 LSTM hidden_dim。
    返回 (model, transform, config)，供训练、数据处理及保存 checkpoint 使用。
    """
    from .model import CaptionModel, FeatureBridge

    encoder = EfficientNetEncoder(pretrained, output_mode)
    model = CaptionModel(
        encoder, FeatureBridge(encoder.output_dim, decoder_dim), decoder,
        encoder_layout=output_mode, freeze_encoder=freeze_encoder,
    )
    config = {
        "encoder_name": ENCODER_NAME,
        "encoder_dim": encoder.output_dim,
        "encoder_layout": output_mode,
        "decoder_visual_dim": decoder_dim,
        "freeze_encoder": freeze_encoder,
        "preprocessing": dict(encoder.data_config),
    }
    # 保存前由调用方补充 decoder 架构、hidden_dim、词表与特殊词编号等。
    return model, encoder.build_transform(), config
