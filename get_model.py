"""
Simple script to download the model and processor
from the Hugging Face model hub and save them to the local directory.

The script must be run before running the app.py script.
"""


from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

print("Loading model and processor from HF...")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")

print("Saving model and processor to local path...")
processor.save_pretrained("./model")
model.save_pretrained("./model")
