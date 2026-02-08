from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from ocr_service import pipeline as pl

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_pipeline_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NER_MODEL_NAME", "model-x")
    monkeypatch.setenv("NER_DEVICE", "CUDA")
    monkeypatch.setenv("PADDLEOCR_GPU", "yes")

    config = pl.PipelineConfig.from_env()

    assert config.ner_model_name == "model-x"
    assert config.ner_device == "cuda"
    assert config.paddleocr_gpu is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("y", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("", False),
    ],
)
def test_env_to_bool(value: str, expected: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAG", value)
    assert pl._env_to_bool("FLAG", default=False) is expected


def test_env_to_bool_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLAG", raising=False)
    assert pl._env_to_bool("FLAG", default=True) is True


@pytest.mark.parametrize(
    "device_name,expected_device",
    [
        ("cuda", 0),
        ("cpu", -1),
    ],
)
def test_build_ner_pipeline_device_mapping(
    device_name: str,
    expected_device: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(pl.AutoTokenizer, "from_pretrained", lambda name: f"tok:{name}")
    monkeypatch.setattr(
        pl.AutoModelForTokenClassification,
        "from_pretrained",
        lambda name: f"model:{name}",
    )

    def fake_pipeline(
        task: str, model: object, tokenizer: object, aggregation_strategy: object, device: int
    ) -> str:
        calls["task"] = task
        calls["model"] = model
        calls["tokenizer"] = tokenizer
        calls["aggregation_strategy"] = aggregation_strategy
        calls["device"] = device
        return "ner"

    monkeypatch.setattr(pl, "hf_pipeline", fake_pipeline)

    result = pl._build_ner_pipeline("model-x", device_name)

    assert result == "ner"
    assert calls["task"] == "token-classification"
    assert calls["model"] == "model:model-x"
    assert calls["tokenizer"] == "tok:model-x"
    assert calls["aggregation_strategy"] == pl.AggregationStrategy.SIMPLE
    assert calls["device"] == expected_device


def test_preprocess_image_outputs_color_array() -> None:
    image_path = next(path for path in DATA_DIR.iterdir() if path.is_file())
    array = pl._preprocess_image(image_path.read_bytes())

    assert isinstance(array, np.ndarray)
    assert array.ndim == 3
    assert array.shape[2] == 3
    assert array.size > 0


def test_run_ocr_filters_empty_lines() -> None:
    class OcrStub:
        def ocr(self, image: np.ndarray) -> list[dict[str, list[str]]]:
            return [{"rec_texts": ["Foo", "Бар"]}]

    reader = OcrStub()
    image = np.zeros((4, 4), dtype=np.uint8)

    assert pl._run_ocr(reader, image) == ["Foo", "Бар"]


def test_extract_ner_tokens_filters_and_normalizes() -> None:
    def fake_ner(text: str) -> list[dict[str, str]]:
        if "Иванов" in text:
            return [
                {"entity_group": "LAST_NAME", "word": "Иванов"},
                {"entity_group": "MIDDLE_NAME", "word": "Иван##ович"},
                {"entity_group": "ORG", "word": "ACME"},
            ]
        if "John" in text:
            return [
                {"entity": "LAST_NAME", "word": "John"},
                {"entity": "FIRST_NAME", "word": "Doe"},
                {"entity": "LOC", "word": "Paris"},
            ]
        return []

    lines = ["Иванов Иванович", "John Doe", " "]

    ner = cast(pl.Pipeline, fake_ner)
    assert pl._extract_ner_tokens(ner, lines) == ["Иванов", "Иванович"]


def test_tokenize_filters_short_and_symbols() -> None:
    assert pl._tokenize("A Li Jean-Pierre Иван-Иванович Q") == [
        "Li",
        "Jean-Pierre",
        "Иван-Иванович",
    ]


def test_normalize_tokens_titlecases_and_dedups() -> None:
    tokens = ["ivan", "Ivan", "IVAN", "jean-pierre", "Jean-Pierre", "-"]
    assert pl._normalize_tokens(tokens) == ["Ivan", "Jean-Pierre"]


def test_titlecase_token_handles_hyphens() -> None:
    assert pl._titlecase_token("iVaN-ivanov") == "Ivan-Ivanov"


def test_select_tokens_from_lines_prefers_max_tokens() -> None:
    lines = ["ABC", "Иван Иванов", "Петр"]
    assert pl._select_tokens_from_lines(lines, pl._is_cyrillic) == ["Иван", "Иванов"]


def test_assign_ru_with_patronymic_suffix() -> None:
    fio = pl._assign_ru(["Иванов", "Иван", "Петрович"])
    assert fio.surname == "Иванов"
    assert fio.name == "Иван"
    assert fio.patronymic == "Петрович"


def test_assign_ru_fallback_patronymic() -> None:
    fio = pl._assign_ru(["Иванов", "Иван", "Сергеев"])
    assert fio.patronymic == "Сергеев"


def test_assign_empty_returns_empty_fio() -> None:
    empty = pl.FioData(name=None, surname=None, patronymic=None)
    assert pl._assign_ru([]) == empty


def test_pick_patronymic_and_is_patronymic() -> None:
    tokens = ["Иван", "Петрович", "Сидоров"]
    assert pl._pick_patronymic(tokens) == "Петрович"
    assert pl._is_patronymic("Кызы") is True
    assert pl._is_patronymic("Smith") is False


def test_is_cyrillic_predicate() -> None:
    assert pl._is_cyrillic("Иван") is True
    assert pl._is_cyrillic("John") is False
    assert pl._is_cyrillic("ИванJohn") is False
