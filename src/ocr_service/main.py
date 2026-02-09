from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Annotated

import anyio
from fastapi import FastAPI, File, HTTPException, UploadFile

from ocr_service.pipeline import OcrPipeline
from ocr_service.schemas import FioResponse

app = FastAPI(title="OCR FIO", version="0.1.0")


@lru_cache
def get_pipeline() -> OcrPipeline:
    return OcrPipeline()


@app.post("/extract", response_model=FioResponse)
async def extract(file: Annotated[UploadFile, File(...)]) -> FioResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Поддерживаются только изображения")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Файл пустой")

    pipeline = get_pipeline()
    result = await anyio.to_thread.run_sync(pipeline.process, payload)

    return FioResponse(**asdict(result.ru))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
