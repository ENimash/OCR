from __future__ import annotations

import os


def _set_safe_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def preload_hf(model_name: str) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(model_name)
    AutoModelForTokenClassification.from_pretrained(model_name)


def preload_paddle() -> None:
    from paddleocr import PaddleOCR

    # Инициализация скачивает необходимые модели в PADDLEX_HOME.
    PaddleOCR(lang="ru", use_textline_orientation=True)


def main() -> None:
    _set_safe_env()
    model_name = os.getenv("NER_MODEL_NAME", "Gherman/bert-base-NER-Russian")
    preload_hf(model_name)
    preload_paddle()


if __name__ == "__main__":
    main()
