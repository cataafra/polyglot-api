import logging
import os
from io import BytesIO

import soundfile as sf
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech


class SeamlessTranslator:
    def __init__(self, model=None, processor=None, device="cpu"):
        self.model = model
        self.processor = processor
        self.device = device

    @classmethod
    def load(cls, logger: logging.Logger):
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        logger.info("Using %s", torch_device)

        path, path_name = ("./model", "local path") if os.path.exists("./model") else (
            "facebook/seamless-m4t-v2-large",
            "Hugging Face",
        )
        logger.info("Loading model and processor from %s...", path_name)

        try:
            model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(path).to(torch_device)
            processor = AutoProcessor.from_pretrained(path)
            logger.info("Model and processor loaded successfully from %s", path_name)
            return cls(model=model, processor=processor, device=torch_device)
        except Exception as exc:
            logger.error("Error loading model and processor: %s", exc)
            return cls(device=torch_device)

    @property
    def ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def health(self):
        return {
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "device": self.device,
        }

    def translate(self, audio_data, sample_rate: int, target_language: str, speaker_id: int) -> bytes:
        if not self.ready:
            raise RuntimeError("Translation model or processor is not loaded")

        processed_audio = self.processor(
            audio=audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)
        audio_array = self.model.generate(
            **processed_audio,
            speaker_id=speaker_id,
            tgt_lang=target_language,
        )[0].cpu().numpy().squeeze()

        output_io = BytesIO()
        sf.write(output_io, audio_array, sample_rate, format="WAV")
        output_io.seek(0)
        return output_io.read()
