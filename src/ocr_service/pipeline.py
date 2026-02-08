from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import Final

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

WHITESPACE_RE: Final = re.compile(r"\s+")

TRAILING_ADMIN_WORDS_RE: Final = re.compile(
    r"\b(республика|край|субъект|область|российской|федерации)\b.*$",
    flags=re.IGNORECASE,
)
NUM_TOKEN_RE: Final = re.compile(r"\b\d+[\/\.\-]?\d*\b")
SINGLE_LETTER_RE: Final = re.compile(r"\b[а-яa-z]\b", flags=re.IGNORECASE)
NON_WORD_RE: Final = re.compile(r"[^\w\sа-я]", flags=re.IGNORECASE)
LATIN_WORD_2PLUS_RE: Final = re.compile(r"\b[a-z]{2,}\b", flags=re.IGNORECASE)
ALNUM_TOKEN_RE: Final = re.compile(r"\b\S*\d\S*\b")
LAT2RU: Final = str.maketrans(
    {
        "a": "а",
        "b": "в",
        "c": "с",
        "e": "е",
        "h": "н",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)
DOC_STOPWORDS: Final = {"ру"}
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
        text = " ".join(lines)
        clean_text = _clean_for_ner(text)
        ner_tokens = _extract_ner_tokens(self._ner, clean_text)
        ru_fio = _assign_ru(ner_tokens)
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
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((960, 1280), resample=Image.Resampling.LANCZOS)
    array = np.array(image)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


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


def _clean_for_ner(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text).lower().strip()
    text = TRAILING_ADMIN_WORDS_RE.sub("", text).strip()
    text = NUM_TOKEN_RE.sub(" ", text)
    text = SINGLE_LETTER_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    text = LATIN_WORD_2PLUS_RE.sub(" ", text)
    text = ALNUM_TOKEN_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = text.translate(LAT2RU)

    text = " ".join(w for w in text.split() if w not in DOC_STOPWORDS)

    return text


def _extract_ner_tokens(ner: Pipeline, text: str) -> list[str]:
    tokens: list[str] = []
    entities = ner(text)

    for entity in entities:
        label = entity.get("entity_group")
        if label in {"LAST_NAME", "FIRST_NAME", "MIDDLE_NAME"}:
            word = entity.get("word", "").replace("##", "")
            tokens.append(word)
    return _normalize_tokens(tokens)


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
