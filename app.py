import os
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
import uvicorn
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {torch_device}")
torch.set_grad_enabled(False)

# Model and processor paths
path = "./model" if os.path.exists("./model") else "facebook/seamless-m4t-v2-large"
print(f"Loading model and processor from {path}")
if not os.path.exists(path):
    path = "facebook/seamless-m4t-v2-large"

model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(path).to(torch_device)
processor = AutoProcessor.from_pretrained(path)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/process")
def process(file: UploadFile = File(...), language: str = Form(...)):
    try:
        contents = file.file.read()
        temp_path = "temp_" + file.filename
        with open(temp_path, 'wb') as f:
            f.write(contents)

        audio_data, samplerate = sf.read(temp_path)
        processed_audio = processor(audios=audio_data, sampling_rate=samplerate, return_tensors="pt").to(torch_device)
        audio_array_from_wav = model.generate(**processed_audio, tgt_lang=language)[0].cpu().numpy().squeeze()

        output_path = "processed_" + file.filename
        sf.write(output_path, audio_array_from_wav, samplerate)

        return FileResponse(output_path)
    except Exception as e:
        return {"message": "There was an error uploading the file " + str(e)}
    finally:
        file.file.close()
