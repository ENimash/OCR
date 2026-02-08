from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
)
from transformers import (
    pipeline as hf_pipeline,
)
from transformers.pipelines.token_classification import AggregationStrategy

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё-]+")
PATRONYMIC_SUFFIXES = (
    "вич",
    "вна",
    "ична",
    "овна",
    "евна",
    "ич",
    "инична",
    "оглы",
    "кызы",
)
MAX_OCR_SIDE = 4000


@dataclass(frozen=True)
class FioData:
    name: str | None
    surname: str | None
    patronymic: str | None

    def is_empty(self) -> bool:
        return not (self.name or self.surname or self.patronymic)


@dataclass(frozen=True)
class ExtractResult:
    ru: FioData
    ocr_result: list[str]


@dataclass(frozen=True)
class PipelineConfig:
    ner_model_name: str
    ner_device: str
    paddleocr_gpu: bool

    @staticmethod
    def from_env() -> PipelineConfig:
        ner_model_name = os.getenv("NER_MODEL_NAME", "Gherman/bert-base-NER-Russian")
        ner_device = os.getenv("NER_DEVICE", "cpu").lower()
        paddleocr_gpu = _env_to_bool("PADDLEOCR_GPU", default=False)
        return PipelineConfig(
            ner_model_name=ner_model_name,
            ner_device=ner_device,
            paddleocr_gpu=paddleocr_gpu,
        )


class OcrPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        if config is None:
            config = PipelineConfig.from_env()
        self._ocr = PaddleOCR(lang="ru", use_textline_orientation=True)
        self._ner = _build_ner_pipeline(config.ner_model_name, config.ner_device)

    def process(self, image_bytes: bytes) -> ExtractResult:
        image = _preprocess_image(image_bytes)
        lines = _run_ocr(self._ocr, image)
        ner_tokens = _extract_ner_tokens(self._ner, lines)
        ru_tokens = ner_tokens
        if len(ru_tokens) < 2:
            ru_tokens = _select_tokens_from_lines(lines, _is_cyrillic)

        ru_fio = _assign_ru(ru_tokens)
        return ExtractResult(ru=ru_fio, ocr_result=lines)


def _env_to_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _build_ner_pipeline(model_name: str, device_name: str) -> Pipeline:
    device = 0 if device_name == "cuda" else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return hf_pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy=AggregationStrategy.SIMPLE,
        device=device,
    )


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image = _resize_image(image, MAX_OCR_SIDE)
    array = np.array(image)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _resize_image(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    if max_dim <= max_side:
        return image
    scale = max_side / max_dim
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def _run_ocr(ocr: PaddleOCR, image: np.ndarray) -> list[str]:
    result = ocr.ocr(image)
    lines = _extract_paddle_texts(result)
    return [line.strip() for line in lines if line and line.strip()]


def _extract_paddle_texts(result: object) -> list[str]:
    if not isinstance(result, list):
        return []

    info = result[0]

    if "rec_texts" not in info:
        return []

    return info["rec_texts"]


def _extract_ner_tokens(ner: Pipeline, lines: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    text = " ".join(lines)
    entities = ner(text)
    print(entities)
    for entity in entities:
        label = entity.get("entity_group")
        if label in {"LAST_NAME", "FIRST_NAME", "MIDDLE_NAME"}:
            word = entity.get("word", "").replace("##", "")
            tokens.extend(_tokenize(word))
    cyrillic_tokens = [token for token in tokens if _is_cyrillic(token)]
    return _normalize_tokens(cyrillic_tokens)

def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text) if len(token) > 1]


def _normalize_tokens(tokens: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip("-")
        if not token:
            continue
        token = _titlecase_token(token)
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(token)
    return normalized


def _titlecase_token(token: str) -> str:
    parts = token.split("-")
    titled = [part[:1].upper() + part[1:].lower() if part else "" for part in parts]
    return "-".join(titled)


def _select_tokens_from_lines(lines: Iterable[str], predicate: Callable[[str], bool]) -> list[str]:
    # Берем строку с максимальным числом токенов, если NER ничего не дал.
    best: list[str] = []
    for line in lines:
        tokens = [token for token in _tokenize(line) if predicate(token)]
        if len(tokens) > len(best):
            best = tokens
    return _normalize_tokens(best)


def _assign_ru(tokens: list[str]) -> FioData:
    if not tokens:
        return FioData(name=None, surname=None, patronymic=None)

    # Эвристика: отчество по суффиксу, далее фамилия/имя по порядку.
    patronymic = _pick_patronymic(tokens)
    remaining = [token for token in tokens if token != patronymic] if patronymic else tokens

    surname = remaining[0] if len(remaining) > 0 else None
    name = remaining[1] if len(remaining) > 1 else None
    if patronymic is None and len(remaining) > 2:
        patronymic = remaining[2]

    return FioData(name=name, surname=surname, patronymic=patronymic)


def _pick_patronymic(tokens: list[str]) -> str | None:
    for token in tokens:
        if _is_patronymic(token):
            return token
    return None


def _is_patronymic(token: str) -> bool:
    lower = token.lower()
    return any(lower.endswith(suffix) for suffix in PATRONYMIC_SUFFIXES)


def _is_cyrillic(token: str) -> bool:
    return bool(CYRILLIC_RE.search(token)) and not bool(LATIN_RE.search(token))
