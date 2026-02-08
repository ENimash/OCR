from __future__ import annotations

from pydantic import BaseModel


class Fio(BaseModel):
    name: str | None = None
    surname: str | None = None
    patronymic: str | None = None

    def is_empty(self) -> bool:
        return not (self.name or self.surname or self.patronymic)


class MultiFio(BaseModel):
    ru: Fio
    en: Fio


class FioWithOcr(Fio):
    ocr_result: list[str]


class MultiFioWithOcr(BaseModel):
    ru: Fio
    en: Fio
    ocr_result: list[str]


FioResponse = FioWithOcr | MultiFioWithOcr
