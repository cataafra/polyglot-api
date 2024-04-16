from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

print("Loading model and processor")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")

print("Saving model and processor")
processor.save_pretrained("./model")
model.save_pretrained("./model")
