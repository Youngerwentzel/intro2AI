"""图片描述项目的基础组装框架；从项目根目录使用 python -m 调用示例。"""

from .model import CaptionModel, FeatureBridge
from .encoder import EfficientNetEncoder, build_caption_model

__all__ = ["CaptionModel", "FeatureBridge", "EfficientNetEncoder", "build_caption_model"]
