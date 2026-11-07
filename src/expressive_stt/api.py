"""Local HTTP API. Run with `estt serve`, then POST an audio file to /transcribe."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .formats import render
from .pipeline import Options, Pipeline

app = FastAPI(title="expressive-stt", version="0.2.0")


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    return Pipeline(Options(language=os.environ.get("ESTT_LANGUAGE") or None))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    format: Literal["json", "text", "llm"] = Query("json"),
    pipe: Pipeline = Depends(get_pipeline),
):
    data = file.file.read()  # sync endpoint: FastAPI runs it in a thread pool
    if not data:
        raise HTTPException(400, "empty file")
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "upload" + suffix)
        with open(path, "wb") as f:
            f.write(data)
        try:
            result = pipe.run(path)
        except Exception as e:  # decoding errors etc.
            raise HTTPException(422, f"could not process audio: {e}") from e
    if format == "json":
        return JSONResponse(result.to_dict())
    return PlainTextResponse(render(result, format))
