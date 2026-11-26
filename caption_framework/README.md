# Image Captioning Model Framework

This package provides a modular encoder–bridge–decoder framework for the course project. It includes model assembly, basic training, checkpoint management, and integration examples. It is not a completed captioning system.

## Installation

Run commands from the repository root:

```powershell
python -m pip install -r caption_framework/requirements.txt
```

Use a PyTorch CPU/CUDA build appropriate for your environment. The original small-network example only requires PyTorch; EfficientNet integration also uses timm, torchvision, and Pillow.

## EfficientNet-B0 Integration

`encoder.py` uses the model selected in the team's `encoder_test.py`: `tf_efficientnet_b0.in1k`, with `features_only=True`, extracting the last feature map. It does not import or execute the original script's hard-coded Mac data path, dataset traversal, or retrieval experiment. The original script remains available for reference.

Run integration checks without downloading weights:

```powershell
python -m caption_framework.encoder_demo
python -m caption_framework.encoder_demo --output-mode spatial
```

Load pretrained weights and optionally supply a training image:

```powershell
python -m caption_framework.encoder_demo --pretrained --image "D:\111\Intro to AI\AIAA2205Project\AIAA2205Project\Training\train_00001.jpg"
```

Replace the image path as needed. The first use of `--pretrained` may download ImageNet weights. Without this flag, the example uses random encoder weights to check structure only. Do not use those random weights as a frozen encoder for actual training.

### Model Assembly

```python
from caption_framework import build_caption_model

model, transform, config = build_caption_model(
    decoder=your_decoder,
    decoder_dim=256,           # Visual dimension accepted by the decoder.
    pretrained=True,           # Load pretrained weights for initial training.
    output_mode="global",     # Choose "spatial" to preserve image locations.
    freeze_encoder=True,
)
```

Supply a decoder implementing the interface below. Its visual input dimension must match `decoder_dim`, which need not equal its LSTM hidden dimension.

For a 224 x 224 input, the last feature map has shape `[B,320,7,7]`. Channel count is read from `feature_info`, rather than assuming the classification head's 1280-dimensional representation.

| Mode | Encoder output | Standardized features | Bridge output for D=256 |
| --- | --- | --- | --- |
| global | `[B,320]` | `[B,1,320]` | `[B,1,256]` |
| spatial | `[B,320,7,7]` | `[B,49,320]` | `[B,49,256]` |

Global mode follows the original script's spatial averaging. Spatial mode supports a future attention decoder. The adapter handles timm's feature list internally.

### Preprocessing

Use `transform(image.convert("RGB"))` to obtain a floating-point tensor `[3,224,224]`. The transform uses the model's configuration for aspect-ratio-preserving resizing, center cropping, and normalization. This differs from the original script's direct square resizing.

Use consistent preprocessing for training, validation, and inference. Do not normalize twice. Regenerate cached features produced with the old transform. The initial transform is deterministic, without random augmentation.

### Verified Environment

Integration checks used Python 3.14, torch 2.10.0, torchvision 0.25.0, and timm 1.0.31. Checks covered global/spatial shapes, real training-image preprocessing, gradients, frozen encoder parameters and BatchNorm statistics, and checkpoint save/load consistency.

These checks did not load pretrained weights or measure caption quality. Enable `pretrained=True` for actual initial training.

## Package Files

| File | Purpose |
| --- | --- |
| `model.py` | Model assembly, feature-layout adaptation, linear bridge, greedy generation |
| `encoder.py` | EfficientNet adapter, preprocessing, model factory |
| `training.py` | Training/validation-loss epochs and checkpoint utilities |
| `demo.py` | Temporary encoder, example LSTM decoder, basic checks |
| `encoder_demo.py` | EfficientNet integration checks with the temporary decoder |
| `requirements.txt` | Runtime dependencies |

## Basic Framework Example

```powershell
python -m caption_framework.demo
```

This example uses random images, artificial captions, and small temporary networks. It does not download weights, access the course test split, or produce submission artifacts. Generated captions are debugging output, not submission predictions.

It checks forward shapes, gradients, bridge updates, frozen encoder parameters and BatchNorm statistics, training/stepwise consistency, checkpoint restoration, resumed training, and three feature layouts. Temporary checkpoints are removed after execution.

## Module Interfaces

Symbols: `B` = batch size, `N` = visual positions, `C` = encoder channels, `D` = decoder visual dimension, `T` = sequence length, `V` = vocabulary size.

### Encoder

Provide `encoder(images) -> Tensor`, specifying preprocessing, output dimension, and layout:

- `global`: `[B,C]`, converted to `[B,1,C]`.
- `spatial`: `[B,C,H,W]`, converted to `[B,H*W,C]`.
- `tokens`: `[B,N,C]`, used directly.

Wrap dictionary/tuple outputs in an adapter selecting the intended feature tensor. Return continuous features, not argmax class IDs.

### Bridge

`FeatureBridge(encoder_dim=C, decoder_dim=D)` maps `[B,N,C]` to `[B,N,D]` using a linear layer. Dimensions are configurable, not fixed to the temporary example's 16 and 12. LSTM state initialization and attention belong inside the decoder.

### Decoder

Implement these methods; see `ExampleDecoder` in `demo.py`:

```python
forward(visual_features, input_tokens)  # Returns raw logits [B,T,V].
init_state(visual_features)            # Returns decoder-specific state.
decode_step(previous_token, visual_features, state)  # Returns ([B,V], new_state).
```

Visual features are `[B,N,D]`; input tokens are `[B,T]`; previous tokens are `[B]`. Token tensors must use `torch.long`. Return logits without softmax. Training and stepwise decoding must use compatible logic and shared parameters.

A basic LSTM can initialize state from image features. An attention decoder can reuse visual features at every step. These details are handled by the decoder, not the bridge.

## Training with Real Data

The DataLoader returns `(images, captions)`: preprocessed floating-point images and token sequences containing BOS/EOS, right-padded with PAD. Build the vocabulary only from course training captions and keep PAD/BOS/EOS/UNK IDs fixed. Use a `collate_fn` for variable-length captions.

Real CSV loading, vocabulary construction, tokenization, and DataLoader creation are not implemented. EfficientNet preprocessing is provided; assign remaining data tasks within the team.

```python
import torch
from caption_framework import CaptionModel, FeatureBridge
from caption_framework.training import run_epoch

model = CaptionModel(
    encoder=your_encoder,
    bridge=FeatureBridge(encoder_dim=C, decoder_dim=D),
    decoder=your_decoder,
    encoder_layout="global",  # Match the encoder's actual output.
    freeze_encoder=True,
).to(device)

optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3
)
for epoch in range(number_of_epochs):
    train_loss = run_epoch(model, train_loader, device, pad_id, optimizer)
    val_loss = run_epoch(model, validation_loader, device, pad_id)
```

This is a template: supply real modules, dimensions, device, loaders, token IDs, and epoch count. Inputs are `captions[:, :-1]`; targets are `captions[:, 1:]`. Cross-entropy ignores PAD. No scheduling, mixed precision, or other optimization techniques are added.

Passing an optimizer enables updates: only use training data in that case. Omitting it computes loss without updates. Frozen encoders stay in evaluation mode when the overall model enters training mode. Reconfigure the optimizer if encoder parameters are later unfrozen.

## Validation and Generation

### Preview the First 10 Training Images

```powershell
python -m caption_framework.preview_training_images --data-root "D:\111\Intro to AI\AIAA2205Project\AIAA2205Project"
```

This inference-only preview loads the pretrained encoder and uses a randomly initialized
bridge and temporary decoder. It does not update parameters or write a model checkpoint.
Generated text is expected to be meaningless until the bridge and decoder are trained.
The vocabulary is built only from Training_Labels.csv. The first 10 training images are
selected by filename order; references are displayed for comparison, not used as targets.

One new JSON file in temporary_results/ contains the vocabulary, configuration, image
paths, references, generated text/token IDs, and complete encoder/bridge feature arrays
with shapes and summary statistics. Use --output to specify a new output filename;
existing files are not overwritten. Use --count to change the number of images.
The --random-encoder flag is available for offline debugging and disables pretrained weights.

Teacher-forcing validation loss is diagnostic, not the course metric. Generate complete validation captions using `model.generate`, export a validation CSV keyed by `image_id`, and use the frozen MiniLM procedure in `Computing_Text_Similarity.ipynb`. Do not backpropagate through that metric.

Validation CSV export, MiniLM scoring, and automatic checkpoint selection are not implemented here.

`model.generate(images, bos_id, eos_id, pad_id, max_len=30)` returns token-ID lists without BOS/EOS/PAD. Convert them to words using the training vocabulary. Each sequence stops at EOS or the length limit. Generation temporarily enters evaluation mode and restores the previous mode afterward.

Immediate EOS can produce an empty list. Diagnose this during validation rather than hiding it with manual captions. Final submitted descriptions must be nonempty and generated by the submitted model.

## Checkpoints and Final Delivery

`save_checkpoint(path, model, config, vocabulary, epoch, optimizer)` saves all model weights and metadata. Include the optimizer to resume training; omit it from the final inference checkpoint.

`load_checkpoint(path, model, device, optimizer)` loads an already constructed model of the same architecture. To reconstruct from configuration, first read with `torch.load(path, map_location="cpu", weights_only=True)`, inspect the saved configuration, construct matching modules, and load their states.

For a complete EfficientNet checkpoint, build with `pretrained=False`: encoder weights are already saved, so another download is unnecessary. See `encoder_demo.py` for the save/rebuild/load sequence.

Store configuration and vocabulary as ordinary dictionaries with primitive values. Record architecture names, dimensions, layout, frozen status, token IDs, and preprocessing. The factory configuration covers encoder/bridge dimensions and preprocessing; add the final decoder type and all constructor arguments before saving.

Frozen status is not in `state_dict` and must be restored from configuration. Resuming requires matching optimizer parameter groups. The example does not save every random-number-generator state needed for exact stochastic replay.

The final checkpoint must include all encoder, bridge, and decoder weights. The byte count returned by `save_checkpoint` must be strictly below **200,000,000**. Exclude datasets, the MiniLM evaluator, and cached image features. The framework does not automatically enforce the final submission directory structure.

Use the course `Test/` split only for final inference after selecting the model and generation settings using training and validation data.
