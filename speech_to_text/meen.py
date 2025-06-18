from transformers import pipeline

cls = pipeline("automatic-speech-recognition")
res = cls("texttospeech.mp3")

print(res)