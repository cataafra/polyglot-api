import logging
import os
from io import BytesIO
from typing import Optional

import colorlog
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from .semantic_memory import build_semantic_memory
from .translation_pipeline import TranslationPipeline, TranslationRequest
from .translator import SeamlessTranslator


def setup_logger():
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s:     %(message)s%(reset)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "yellow",
                "WARNING": "orange",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    gunicorn_logger = logging.getLogger("gunicorn.error")
    api_logger = colorlog.getLogger(__name__)
    api_logger.handlers = gunicorn_logger.handlers or [handler]
    api_logger.setLevel(gunicorn_logger.level or logging.INFO)
    return api_logger


logger = setup_logger()
app = FastAPI()

translator = SeamlessTranslator.load(logger)
semantic_memory = build_semantic_memory()
pipeline = TranslationPipeline(translator=translator, semantic_memory=semantic_memory)


@app.get("/")
async def root():
    return {"message": "Welcome to Polyglot API!"}


@app.get("/health")
async def health():
    translator_health = translator.health()
    file_system_accessible = os.access(".", os.W_OK)
    semantic_health = semantic_memory.health()

    return {
        "status": all(
            [
                translator_health["model_loaded"],
                translator_health["processor_loaded"],
                file_system_accessible,
            ]
        ),
        "model_loaded": translator_health["model_loaded"],
        "processor_loaded": translator_health["processor_loaded"],
        "device": translator_health.get("device", "unknown"),
        "semantic_memory": semantic_health,
        "details": {
            **translator_health,
            "file_system_accessible": file_system_accessible,
            "semantic_memory": semantic_health,
        },
    }


@app.post("/process")
def process(
    file: UploadFile = File(...),
    language: str = Form(...),
    speaker_id: int = Form(...),
    background_tasks: BackgroundTasks = None,
    session_id: Optional[str] = Form(None),
    source_language: Optional[str] = Form(None),
    domain: Optional[str] = Form(None),
    privacy_level: Optional[str] = Form(None),
    use_semantic_cache: Optional[str] = Form(None),
    cache_strategy: Optional[str] = Form(None),
    use_transcript_memory: Optional[str] = Form(None),
):
    output_path = None
    try:
        result = _process_upload(
            file=file,
            language=language,
            speaker_id=speaker_id,
            session_id=session_id,
            source_language=source_language,
            domain=domain,
            privacy_level=privacy_level,
            use_semantic_cache=use_semantic_cache,
            cache_strategy=cache_strategy,
            use_transcript_memory=use_transcript_memory,
        )

        output_path = "processed_" + os.path.basename(file.filename)
        with open(output_path, "wb") as output_file:
            output_file.write(result.audio_bytes)
            output_file.flush()
            os.fsync(output_file.fileno())

        if background_tasks is None:
            background_tasks = BackgroundTasks()
        background_tasks.add_task(os.remove, output_path)

        return FileResponse(
            output_path,
            filename=output_path,
            headers=result.headers,
            background=background_tasks,
        )
    except Exception as exc:
        logger.error("There was an error processing the file: ", exc_info=True)
        return {"message": "There was an error processing the file: " + str(exc)}
    finally:
        file.file.close()


@app.post("/process_memory")
def process_memory(
    file: UploadFile = File(...),
    language: str = Form(...),
    speaker_id: int = Form(...),
    session_id: Optional[str] = Form(None),
    source_language: Optional[str] = Form(None),
    domain: Optional[str] = Form(None),
    privacy_level: Optional[str] = Form(None),
    use_semantic_cache: Optional[str] = Form(None),
    cache_strategy: Optional[str] = Form(None),
    use_transcript_memory: Optional[str] = Form(None),
):
    try:
        result = _process_upload(
            file=file,
            language=language,
            speaker_id=speaker_id,
            session_id=session_id,
            source_language=source_language,
            domain=domain,
            privacy_level=privacy_level,
            use_semantic_cache=use_semantic_cache,
            cache_strategy=cache_strategy,
            use_transcript_memory=use_transcript_memory,
        )
        return StreamingResponse(
            BytesIO(result.audio_bytes),
            media_type="audio/wav",
            headers=result.headers,
        )
    except Exception as exc:
        logger.error("There was an error processing the file: ", exc_info=True)
        return {"message": "There was an error processing the file: " + str(exc)}
    finally:
        file.file.close()


@app.post("/memory/expire")
def expire_memory():
    try:
        return {"deleted_sessions": semantic_memory.expire_old_records()}
    except Exception as exc:
        logger.error("There was an error expiring semantic memory: ", exc_info=True)
        return {"message": "There was an error expiring semantic memory: " + str(exc)}


def _process_upload(
    file: UploadFile,
    language: str,
    speaker_id: int,
    session_id: Optional[str],
    source_language: Optional[str],
    domain: Optional[str],
    privacy_level: Optional[str],
    use_semantic_cache: Optional[str],
    cache_strategy: Optional[str],
    use_transcript_memory: Optional[str],
):
    return pipeline.process(
        contents=file.file.read(),
        request=TranslationRequest(
            target_language=language,
            speaker_id=speaker_id if speaker_id is not None else 1,
            session_id=session_id,
            source_language=source_language,
            domain=domain,
            privacy_level=privacy_level,
            use_semantic_cache=use_semantic_cache,
            cache_strategy=cache_strategy,
            use_transcript_memory=use_transcript_memory,
        ),
    )
