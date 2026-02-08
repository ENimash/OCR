from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from easyocr import Reader
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
    en: FioData | None


@dataclass(frozen=True)
class PipelineConfig:
    ocr_langs: tuple[str, ...]
    ner_model_name: str
    ner_device: str
    easyocr_gpu: bool

    @staticmethod
    def from_env() -> PipelineConfig:
        ocr_langs = tuple(
            lang.strip() for lang in os.getenv("OCR_LANGS", "ru,en").split(",") if lang.strip()
        )
        ner_model_name = os.getenv("NER_MODEL_NAME", "zaalbar/rubert-finetuned-ner")
        ner_device = os.getenv("NER_DEVICE", "cpu").lower()
        easyocr_gpu = _env_to_bool("EASYOCR_GPU", default=False)
        return PipelineConfig(
            ocr_langs=ocr_langs,
            ner_model_name=ner_model_name,
            ner_device=ner_device,
            easyocr_gpu=easyocr_gpu,
        )


class OcrPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        if config is None:
            config = PipelineConfig.from_env()
        self._reader = Reader(list(config.ocr_langs), gpu=config.easyocr_gpu)
        self._ner = _build_ner_pipeline(config.ner_model_name, config.ner_device)

    def process(self, image_bytes: bytes) -> ExtractResult:
        image = _preprocess_image(image_bytes)
        lines = _run_ocr(self._reader, image)
        ner_tokens = _extract_ner_tokens(self._ner, lines)
        ru_tokens = [token for token in ner_tokens if _is_cyrillic(token)]
        en_tokens = [token for token in ner_tokens if _is_latin(token)]

        if len(ru_tokens) < 2:
            ru_tokens = _select_tokens_from_lines(lines, _is_cyrillic)
        if len(en_tokens) < 2:
            en_tokens = _select_tokens_from_lines(lines, _is_latin)

        ru_fio = _assign_ru(ru_tokens)
        en_fio = _assign_en(en_tokens) if en_tokens else None
        return ExtractResult(ru=ru_fio, en=en_fio)


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
    image = image.convert("L")
    return np.array(image)


def _run_ocr(reader: Reader, image: np.ndarray) -> list[str]:
    lines = reader.readtext(image, detail=0, paragraph=False)
    return [line.strip() for line in lines if line and line.strip()]


def _extract_ner_tokens(ner: Pipeline, lines: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        entities = ner(line)
        for entity in entities:
            label = entity.get("entity_group") or entity.get("entity", "")
            if label in {"PER", "PERSON"} or label.endswith("PER"):
                word = entity.get("word", "").replace("##", "")
                tokens.extend(_tokenize(word))
    return _normalize_tokens(tokens)


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


def _assign_en(tokens: list[str]) -> FioData:
    if not tokens:
        return FioData(name=None, surname=None, patronymic=None)

    surname = tokens[0] if len(tokens) > 0 else None
    name = tokens[1] if len(tokens) > 1 else None
    patronymic = tokens[2] if len(tokens) > 2 else None

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


def _is_latin(token: str) -> bool:
    return bool(LATIN_RE.search(token)) and not bool(CYRILLIC_RE.search(token))
