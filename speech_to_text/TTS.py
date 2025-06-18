# from gtts import gTTS

# language = "en"

# text = "hey hello hi how are u byeee"

# speech = gTTS(text=text, lang=language,slow=False, tld = "com.au",)
# speech.save("texttospeech4.mp3")

import pyttsx3

def init_tts_engine():
    """Initialize and return the TTS engine with custom settings"""
    engine = pyttsx3.init()

    # Set speech rate (default ~200)
    engine.setProperty('rate', 150)

    # Set volume (0.0 to 1.0)
    engine.setProperty('volume', 1.0)

    # Optional: Change voice (index 0 for male, 1 for female — varies by OS)
    voices = engine.getProperty('voices')
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name} - {voice.languages}")
    engine.setProperty('voice', voices[1].id)  # Change index as needed

    return engine

def speak_text(text, engine):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    engine = init_tts_engine()
    while True:
        text = input("📝 Enter text (or 'exit' to quit): ")
        if text.lower() == 'exit':
            print("👋 Goodbye!")
            break
        speak_text(text, engine)
