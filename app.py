"""
Polyglot main FastAPI application.

This file contains the FastAPI application that will be used to serve the model and perform inference.
"""

import os

import colorlog as colorlog
import soundfile as sf
import torch
import logging
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech


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

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
except Exception as e:
    logger.error(f"Error loading model and processor: {e}")


@app.get("/")
async def root():
    return {"message": "Welcome to Polyglot API!"}


@app.post("/process")
def process(file: UploadFile = File(...), language: str = Form(...)):
    """
    Process the uploaded audio file and return the translated audio file.
    :param file: file-like object containing the audio data
    :param language: str, the target language to translate the audio to
    :return: FileResponse containing the processed audio file
    """
    try:
        contents = file.file.read()
        temp_path = "temp_" + file.filename
        with open(temp_path, 'wb') as f:
            f.write(contents)
            f.flush()  # Flush Python's internal buffer
            os.fsync(f.fileno())  # Ensure all data is written to disk

        # Read the audio file
        audio_data, samplerate = sf.read(temp_path)
        processed_audio = processor(audios=audio_data, sampling_rate=samplerate, return_tensors="pt").to(torch_device)
        audio_array_from_wav = model.generate(**processed_audio, tgt_lang=language)[0].cpu().numpy().squeeze()

        output_path = "processed_" + file.filename
        with open(output_path, 'wb') as f:
            sf.write(f, audio_array_from_wav, samplerate)  # Directly write using soundfile to file
            f.flush()
            os.fsync(f.fileno())  # Ensure the output file is also flushed to disk

        return FileResponse(output_path)
    except Exception as err:
        logger.error("There was an error processing the file: ", exc_info=True)
        return {"message": "There was an error processing the file: " + str(err)}
    finally:
        file.file.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Ensure temporary file is removed after processing
        if os.path.exists(output_path):
            os.remove(output_path)
