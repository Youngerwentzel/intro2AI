#!/usr/bin/env python3
"""Evaluation harness for the AIAA 2205 image-captioning project.

Scores generated captions against reference captions with the frozen
``sentence-transformers/all-MiniLM-L6-v2`` model stored in
``Text_Embedding_Model/``.  The score is the mean cosine similarity over
images, computed exactly as in ``Computing_Text_Similarity.ipynb``:

1. Run every description through ``AutoModel`` and take ``last_hidden_state``.
2. Mean-pool the token vectors, weighting by ``attention_mask``.
3. L2-normalise the resulting 384-dimensional sentence vectors.
4. Cosine similarity = dot product of the two normalised vectors.
5. Average the per-image similarities (arithmetic mean).

The demo at the bottom scores ``Validation_Predictions_Example.csv`` against
``Validation_Labels.csv``.  It follows the notebook step for step, so it
returns the same value as re-running ``Computing_Text_Similarity.ipynb`` on
the same files.

Usage
-----
    python eval.py

Dependencies: torch, transformers, pandas, safetensors.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 hidden size
DEFAULT_BATCH_SIZE = 64       # descriptions per forward pass (memory friendly)
DEFAULT_MAX_LENGTH = 64       # same truncation length as the official notebook

# Value saved in Computing_Text_Similarity.ipynb.  It is informational only:
# Validation_Labels.csv was modified after that output was recorded (Nov 9 vs
# Nov 4), so re-running the notebook's own code on the current files gives a
# different number.  The demo therefore reports a match or an explanation
# instead of treating this old value as ground truth.
NOTEBOOK_SAVED_SCORE = 0.583020
SCORE_TOLERANCE = 5e-4

# Loaded (tokenizer, model) pairs, keyed by (resolved model dir, device).
# Caching avoids reloading the ~90 MB MiniLM checkpoint when several prediction
# files (e.g. checkpoints) are evaluated in the same Python process.
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


# ---------------------------------------------------------------------------
# Device / model loading
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Return the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    # ``torch.backends.mps`` does not exist on builds without Metal support,
    # hence the getattr guard.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Normalise a user-supplied device, auto-detecting when it is None."""
    if device is None:
        return get_device()
    return torch.device(device)


def _load_encoder(model_dir: str | Path, device: torch.device) -> tuple:
    """Load the local tokenizer/model once and keep them on ``device``.

    ``local_files_only=True`` guarantees the frozen local checkpoint is used
    and nothing is downloaded from the Hugging Face Hub.
    """
    model_path = Path(model_dir).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Text embedding model directory not found: {model_path}"
        )

    cache_key = (str(model_path.resolve()), str(device))
    if cache_key not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)

        # The embedding model is frozen: evaluation only, never trained.
        model.to(device)
        model.eval()
        model.requires_grad_(False)

        _MODEL_CACHE[cache_key] = (tokenizer, model)

    return _MODEL_CACHE[cache_key]


def clear_model_cache() -> None:
    """Drop cached models (useful to release memory in long notebooks)."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
def _validate_texts(texts: list[str], name: str) -> list[str]:
    """Ensure we have a non-empty list of non-blank strings; strip whitespace."""
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"{name} must be a list of strings, got {type(texts).__name__}")

    cleaned: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(
                f"{name}[{index}] must be a string, got {type(text).__name__}"
            )
        text = text.strip()
        if not text:
            raise ValueError(f"{name}[{index}] is empty; captions must be non-empty.")
        cleaned.append(text)

    if not cleaned:
        raise ValueError(f"{name} is empty; at least one caption is required.")
    return cleaned


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    """Encode texts into L2-normalised sentence embeddings.

    Returns a float32 CPU tensor of shape ``(len(texts), 384)``.  Working on
    CPU after each batch keeps GPU/MPS memory bounded by ``batch_size``.
    """
    batches: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {name: tensor.to(device) for name, tensor in tokens.items()}

        # (B, T, 384): contextual embedding for every token.
        last_hidden_state = model(**tokens).last_hidden_state

        # ---------------------------------------------------------------
        # Attention-mask-weighted mean pooling:
        #   sum(hidden * mask) / clamp(sum(mask), min=1e-9)
        # The mask is cast to the hidden-state dtype so the multiplication
        # and division stay in float32 on every backend (MPS included).
        # Note: the denominator must sum the original 2-D attention_mask
        # (B, T), not the unsqueezed 3-D copy, otherwise keepdim produces a
        # (B, 1, 1) denominator that broadcasts into a (B, B, H) tensor.
        # Padding positions contribute zero; clamp protects empty or fully
        # padded sequences from division by zero.
        # ---------------------------------------------------------------
        attention_mask = tokens["attention_mask"]
        pooled = torch.sum(
            last_hidden_state
            * attention_mask.unsqueeze(-1).to(last_hidden_state.dtype),
            dim=1,
        ) / torch.clamp(
            attention_mask.sum(dim=1, keepdim=True).to(last_hidden_state.dtype),
            min=1e-9,
        )

        # L2 normalisation; float() is a no-op for fp32 but guards against
        # checkpoints loaded in half precision.
        normalised = F.normalize(pooled.float(), p=2, dim=1)

        batches.append(normalised.cpu())

    embeddings = torch.cat(batches, dim=0)

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIM}-dimensional embeddings from "
            f"{model.name_or_path!r}, got shape {tuple(embeddings.shape)}. "
            "Check that Text_Embedding_Model/ is all-MiniLM-L6-v2."
        )

    return embeddings


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------
def mean_cosine(
    preds: list[str],
    refs: list[str],
    model_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: str | torch.device | None = None,
) -> float:
    """Mean cosine similarity between paired prediction/reference captions.

    Args:
        preds: Generated captions, already aligned with ``refs``.
        refs: Ground-truth captions, same length and order as ``preds``.
        model_dir: Local directory containing the frozen MiniLM model.
        batch_size: Number of texts per forward pass (default 64).
        max_length: Tokenizer truncation length (default 64, matching the
            official notebook).
        device: Optional ``"cuda"``, ``"mps"``, ``"cpu"``; auto-detected if None.

    Returns:
        The arithmetic mean of the per-pair cosine similarities.
    """
    if len(preds) != len(refs):
        raise ValueError(
            f"preds and refs must have the same length, got "
            f"{len(preds)} and {len(refs)}."
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}.")

    preds = _validate_texts(preds, "preds")
    refs = _validate_texts(refs, "refs")

    resolved_device = _resolve_device(device)
    tokenizer, model = _load_encoder(model_dir, resolved_device)

    # Both sides use identical tokenisation, pooling, and normalisation.
    pred_embeddings = _encode_texts(
        preds, tokenizer, model, resolved_device, batch_size, max_length
    )
    ref_embeddings = _encode_texts(
        refs, tokenizer, model, resolved_device, batch_size, max_length
    )

    # Vectors are unit length, so the dot product *is* the cosine similarity.
    per_image = torch.sum(pred_embeddings * ref_embeddings, dim=1)
    return float(per_image.mean().item())


# ---------------------------------------------------------------------------
# CSV loading / alignment
# ---------------------------------------------------------------------------
def _read_csv(path: str | Path, kind: str) -> pd.DataFrame:
    """Read a UTF-8 CSV (tolerating a BOM) and normalise column names."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(f"{kind} CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _clean_description(value) -> str:
    if pd.isna(value):
        raise ValueError("Found a missing (NaN) description.")
    text = str(value).strip()
    if not text:
        raise ValueError("Found an empty description.")
    return text


def _normalise_image_id(value) -> str:
    """Convert ``validation_00001.jpg`` / ``dir/validation_00001.jpg`` to
    ``validation_00001`` so labels and predictions are joinable.

    ``Path.stem`` strips the extension and any directory component; the
    backslash replacement also handles paths written on Windows.
    """
    if pd.isna(value):
        raise ValueError("Found a missing (NaN) image_id.")
    text = str(value).strip().replace("\\", "/")
    if not text:
        raise ValueError("Found an empty image_id.")
    return Path(text).stem


def _prepare_table(
    frame: pd.DataFrame,
    id_column: str,
    kind: str,
) -> pd.DataFrame:
    """Return a clean ``image_id`` / ``description`` table with unique IDs."""
    if id_column not in frame.columns:
        raise ValueError(
            f"{kind} CSV is missing the {id_column!r} column; "
            f"found columns: {list(frame.columns)}"
        )
    if "description" not in frame.columns:
        raise ValueError(
            f"{kind} CSV is missing the 'description' column; "
            f"found columns: {list(frame.columns)}"
        )

    prepared = pd.DataFrame(
        {
            "image_id": frame[id_column].map(_normalise_image_id),
            "description": frame["description"].map(_clean_description),
        }
    )

    duplicates = prepared["image_id"][prepared["image_id"].duplicated()]
    if not duplicates.empty:
        examples = sorted(duplicates.unique().tolist())[:5]
        raise ValueError(
            f"{kind} CSV contains duplicate image_ids, e.g. {examples}. "
            "Each image must appear exactly once."
        )

    return prepared


def _format_id_examples(ids: set[str]) -> str:
    examples = sorted(ids)[:5]
    suffix = ", ..." if len(ids) > len(examples) else ""
    return "{" + ", ".join(examples) + suffix + "}"


def evaluate_csv(
    pred_csv_path: str,
    ref_csv_path: str,
    model_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    strict: bool = True,
    verbose: bool = True,
    device: str | torch.device | None = None,
) -> float:
    """Score a prediction CSV against a reference CSV, matching by ``image_id``.

    Rows are aligned with an inner join on ``image_id``, never by row order.
    The reference file may use either ``image_filename`` (as supplied in
    ``Validation_Labels.csv``) or ``image_id``; filenames are reduced to their
    stem so ``validation_00001.jpg`` matches ``validation_00001``.

    Args:
        pred_csv_path: CSV with ``image_id,description`` columns.
        ref_csv_path: CSV with ``image_filename,description`` or
            ``image_id,description`` columns.
        model_dir: Local frozen MiniLM directory.
        batch_size: Number of texts per forward pass.
        max_length: Tokenizer truncation length.
        strict: If True (default), raise when the two files do not contain
            exactly the same image_ids. If False, warn and score only the
            intersection.
        verbose: Print how many rows were aligned.
        device: Optional device override; auto-detected if None.

    Returns:
        Mean cosine similarity over the aligned images.
    """
    predictions = _prepare_table(
        _read_csv(pred_csv_path, "Prediction"), "image_id", "Prediction"
    )

    references_raw = _read_csv(ref_csv_path, "Reference")
    # Validation_Labels.csv uses image_filename; also accept an already-stemmed
    # image_id column so the helper works with either format.
    if "image_filename" in references_raw.columns:
        reference_id_column = "image_filename"
    elif "image_id" in references_raw.columns:
        reference_id_column = "image_id"
    else:
        raise ValueError(
            "Reference CSV must contain 'image_filename' or 'image_id'; "
            f"found columns: {list(references_raw.columns)}"
        )

    references = _prepare_table(
        references_raw, reference_id_column, "Reference"
    )

    # Inner join: only image_ids present in both files are scored. Sorting by
    # image_id makes the result independent of the input row order.
    merged = references.rename(
        columns={"description": "reference_description"}
    ).merge(
        predictions.rename(columns={"description": "predicted_description"}),
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.sort_values("image_id", kind="stable").reset_index(drop=True)

    reference_ids = set(references["image_id"])
    prediction_ids = set(predictions["image_id"])
    missing = reference_ids - prediction_ids      # labels without predictions
    extra = prediction_ids - reference_ids        # predictions without labels

    if missing or extra:
        problems = []
        if missing:
            problems.append(
                f"{len(missing)} reference image_id(s) have no prediction, "
                f"e.g. {_format_id_examples(missing)}"
            )
        if extra:
            problems.append(
                f"{len(extra)} predicted image_id(s) are not in the labels, "
                f"e.g. {_format_id_examples(extra)}"
            )
        message = "Prediction and reference image_ids do not match: " + "; ".join(
            problems
        )

        if strict:
            raise ValueError(
                message
                + f". Matched {len(merged)} of {len(references)} reference "
                f"and {len(predictions)} prediction rows. Pass strict=False "
                "to score the intersection instead."
            )
        warnings.warn(
            message + f". Scoring the {len(merged)} matched image(s) only.",
            stacklevel=2,
        )

    if merged.empty:
        raise ValueError(
            "No image_ids matched between the prediction and reference CSVs; "
            "nothing to score."
        )

    if verbose:
        print(
            f"Aligned {len(merged)} image(s) by image_id "
            f"({len(predictions)} predictions, {len(references)} references)."
        )

    return mean_cosine(
        merged["predicted_description"].tolist(),
        merged["reference_description"].tolist(),
        model_dir,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )


# ---------------------------------------------------------------------------
# Demo / self-check
# ---------------------------------------------------------------------------
def _find_data_root() -> Path:
    """Locate the folder holding the course data.

    Supports running from the project root, from its parent (as the notebook
    does), or from anywhere if eval.py sits next to the data.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd(),
        Path.cwd() / "AIAA2205Project",
        here,
        here / "AIAA2205Project",
    ]

    for candidate in dict.fromkeys(candidates):  # de-duplicate, keep order
        if (
            (candidate / "Validation_Labels.csv").is_file()
            and (candidate / "Validation_Predictions_Example.csv").is_file()
            and (candidate / "Text_Embedding_Model").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find Validation_Labels.csv, "
        "Validation_Predictions_Example.csv, and Text_Embedding_Model/. "
        "Run eval.py from the AIAA2205Project folder (or its parent)."
    )


if __name__ == "__main__":
    data_root = _find_data_root()

    labels_path = data_root / "Validation_Labels.csv"
    example_predictions_path = data_root / "Validation_Predictions_Example.csv"
    model_path = data_root / "Text_Embedding_Model"

    print(f"Data root : {data_root}")
    print(f"Device    : {get_device()}")

    score = evaluate_csv(
        str(example_predictions_path),
        str(labels_path),
        str(model_path),
    )

    print(f"Mean cosine similarity: {score:.6f}")

    # Comparing with the notebook's saved output is only a sanity note.  The
    # verification that matters is that the notebook's own code produces the
    # same value on the current files; a mismatch here reflects the data
    # revision, not a different metric.
    delta = abs(score - NOTEBOOK_SAVED_SCORE)
    if delta <= SCORE_TOLERANCE:
        print(
            f"[check] matches the notebook's saved score "
            f"{NOTEBOOK_SAVED_SCORE:.6f} (|delta| = {delta:.1e})."
        )
    else:
        print(
            f"[note] Computing_Text_Similarity.ipynb saved "
            f"{NOTEBOOK_SAVED_SCORE:.6f}, but its own code also produces "
            f"{score:.6f} on the current Validation_Labels.csv.  The labels "
            "file was modified after that saved output, so compare against a "
            "fresh run of the notebook rather than the stored number."
        )

    print(
        "\nTo score your own captions, run:\n"
        '  evaluate_csv("Validation_Predictions.csv", '
        '"Validation_Labels.csv", "Text_Embedding_Model")'
    )
