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

    def transcribe(self, audio_data, sample_rate: int, source_language: str) -> str:
        if not self.ready:
            raise RuntimeError("Translation model or processor is not loaded")
        if not source_language or source_language.lower() == "auto":
            return ""
        source_language = source_language.replace("__", "")

        processed_audio = self.processor(
            audio=audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(self.device)

        generation_kwargs = dict(processed_audio)
        input_features = generation_kwargs.pop("input_features", None)
        if input_features is None:
            raise RuntimeError("Processor did not return input_features for transcription")

        language_id = self.model.generation_config.text_decoder_lang_to_code_id.get(source_language)
        if language_id is None:
            raise ValueError(f"source_language={source_language} is not supported for transcription")

        batch_size = len(input_features)
        generation_kwargs["decoder_input_ids"] = torch.tensor([[language_id]] * batch_size, device=self.device)
        generation_kwargs["return_dict_in_generate"] = True
        generation_kwargs.setdefault("max_new_tokens", int(os.getenv("POLYGLOT_TRANSCRIPT_MAX_NEW_TOKENS", "64")))

        with torch.inference_mode():
            output_tokens = super(SeamlessM4Tv2ForSpeechToSpeech, self.model).generate(
                input_features,
                **generation_kwargs,
            )
        return self._decode_text_tokens(output_tokens)

    def _decode_text_tokens(self, output_tokens) -> str:
        tokens = output_tokens
        if hasattr(tokens, "sequences"):
            tokens = tokens.sequences
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().cpu()
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()

        while isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        if not tokens:
            return ""
        return self.processor.decode(tokens, skip_special_tokens=True).strip()
