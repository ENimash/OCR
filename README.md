# OCR
Тестовое задание в Московский Метрополитен. Сервис извлекает ФИО из изображений СТС: OCR (PaddleOCR) + NER (ruBERT).

**Стек:** FastAPI, PaddleOCR, Transformers, PyTorch, OpenCV.

**Функциональность:**
- `POST /extract` — принимает изображение и возвращает `{"name","surname","patronymic"}`.
- `GET /health` — healthcheck.

## Требования
- Python `>=3.11,<3.12`
- Менеджер зависимостей: `poetry`
- Для Docker — установленный Docker

## Быстрый старт (локально)
```bash
poetry install
make run
```

Сервис будет доступен на `http://127.0.0.1:8000`.

## Пример запроса
```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/sample.jpg"
```

Ответ:
```json
{"name":"Иван","surname":"Иванов","patronymic":"Иванович"}
```

## Docker
Сборка образа:
```bash
docker build -t ocr-service .
```

Запуск контейнера:
```bash
docker run --rm -p 8000:8000 ocr-service
```

Образы собираются с предзагрузкой моделей (см. `scripts/preload_models.py`), поэтому при первом запуске контейнера не требуется интернет.

## Переменные окружения
- `NER_MODEL_NAME` — модель NER для `transformers` (по умолчанию `Gherman/bert-base-NER-Russian`).
- `NER_DEVICE` — `cpu` или `cuda` (по умолчанию `cpu`).
- `PADDLEOCR_GPU` — `true/false` (по умолчанию `false`).

## Архитектура пайплайна
1. Предобработка изображения (resize + бинаризация).
2. OCR через PaddleOCR.
3. Очистка текста для NER.
4. NER на русском (ruBERT).
5. Эвристика для назначения ФИО (по суффиксам отчества и порядку).

## Команды разработки
```bash
make lint
make format
make typecheck
make test
```