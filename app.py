"""
Polyglot main FastAPI application.

This file contains the FastAPI application that will be used to serve the model and perform inference.
"""

import os
import time
from io import BytesIO
import colorlog as colorlog
import soundfile as sf
import torch
import logging
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.logger import logger
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.background import BackgroundTasks
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech
from io import BytesIO


# Create the FastAPI app
app = FastAPI()

# Set up colorlog
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:     %(message)s%(reset)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'yellow',
        'WARNING': 'orange',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    reset=True,
    style='%'
))

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers
if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.DEBUG)

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)

# Set the device to use for inference
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using {torch_device}")
torch.set_grad_enabled(False)

# Get the model and processor from the local path if it exists, otherwise download from Hugging Face
(path, path_name) = ("./model", "local path") if os.path.exists("./model") else (
    "facebook/seamless-m4t-v2-large", "Hugging Face")
logger.info(f"Loading model and processor from {path_name}...")

# Load the model and processor
try:
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(path).to(torch_device)
    processor = AutoProcessor.from_pretrained(path)
    logger.info(f"Model and processor loaded successfully from {path_name}")
except Exception as e:
    logger.error(f"Error loading model and processor: {e}")


@app.get("/")
async def root():
    return {"message": "Welcome to Polyglot API!"}


@app.get("/health")
async def health():
    """
    Check the health of the application.
    :return: dict containing the health status and details
    """
    # Check if the model and processor are loaded
    model_loaded = model is not None
    processor_loaded = processor is not None

    # Check if the application can access the file system
    file_system_accessible = os.access(".", os.W_OK)

    # Return the health status
    return {
        "status": all([model_loaded, processor_loaded, file_system_accessible]),
        "details": {
            "model_loaded": model_loaded,
            "processor_loaded": processor_loaded,
            "file_system_accessible": file_system_accessible,
        }
    }


@app.post("/process")
def process(file: UploadFile = File(...), language: str = Form(...), speaker_id: int = Form(...),
            background_tasks: BackgroundTasks = None):
    """
    Process the uploaded audio file and return the translated audio file.
    :param speaker_id: int, the speaker ID to use for the audio
    :param background_tasks: BackgroundTasks, used to schedule file deletion after response
    :param file: file-like object containing the audio data
    :param language: str, the target language to translate the audio to
    :return: FileResponse containing the processed audio file
    """
    try:
        start = time.time()
        contents = file.file.read()
        temp_path = "temp_" + file.filename
        with open(temp_path, 'wb') as f:
            f.write(contents)
            f.flush()  # Flush Python's internal buffer
            os.fsync(f.fileno())  # Ensure all data is written to disk

        # Read the audio file
        audio_data, samplerate = sf.read(temp_path)

        # Check for speaker_id
        speaker_id = speaker_id if speaker_id else 1

        # Process the audio file
        processed_audio = processor(audios=audio_data, sampling_rate=samplerate, return_tensors="pt").to(torch_device)
        audio_array_from_wav = model.generate(**processed_audio, speaker_id=speaker_id, tgt_lang=language)[
            0].cpu().numpy().squeeze()

        output_path = "processed_" + file.filename
        with open(output_path, 'wb') as f:
            sf.write(f, audio_array_from_wav, samplerate)  # Directly write using soundfile to file
            f.flush()
            os.fsync(f.fileno())  # Ensure the output file is also flushed to disk

        response = FileResponse(output_path, filename=output_path)  # Prepare response
        background_tasks.add_task(os.remove, output_path)  # Schedule file deletion after response
        logger.info(f"Processing took {time.time() - start:.2f} seconds")
        return response
    except Exception as err:
        logger.error("There was an error processing the file: ", exc_info=True)
        return {"message": "There was an error processing the file: " + str(err)}
    finally:
        file.file.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Ensure temporary file is removed after processing


@app.post("/process_memory")
def process_memory(file: UploadFile = File(...), language: str = Form(...), speaker_id: int = Form(...)):
    """
    Process the uploaded audio file in memory and return the translated audio file.
    Similar to the process endpoint, but reads the audio file into memory and processes it without writing to disk.
    :param speaker_id: int, the speaker ID to use for the audio
    :param file: file-like object containing the audio data
    :param language: str, the target language to translate the audio to
    :return: StreamingResponse containing the processed audio file
    """
    try:
        start = time.time()
        # Read the audio file
        contents = file.file.read()
        file_io = BytesIO(contents)

        # Read the audio file from memory
        audio_data, samplerate = sf.read(file_io)

        # Check for speaker_id
        speaker_id = speaker_id if speaker_id else 1

        # Process the audio file
        processed_audio = processor(audios=audio_data, sampling_rate=samplerate, return_tensors="pt").to(torch_device)
        audio_array_from_wav = model.generate(**processed_audio, speaker_id=speaker_id, tgt_lang=language)[
            0].cpu().numpy().squeeze()

        # Write the processed audio to memory
        output_io = BytesIO()
        sf.write(output_io, audio_array_from_wav, samplerate, format='WAV')
        output_io.seek(0)  # Rewind the buffer to the beginning

        # Log the processing time
        logger.info(f"Processing took {time.time() - start:.2f} seconds")

        # Prepare the response
        return StreamingResponse(output_io, media_type="audio/wav")
    except Exception as err:
        logger.error("There was an error processing the file: ", exc_info=True)
        return {"message": "There was an error processing the file: " + str(err)}
