import speech_recognition as sr
import pyttsx3

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

def speak(text):
    """Converts text to speech"""
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_command():
    """Listens to the microphone and returns the recognized text"""
    with sr.Microphone() as source:
        print("\n[Listening...]")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio)
            return command.lower()
        except sr.WaitTimeoutError:
            print("⏱️ Timeout: No speech detected.")
        except sr.UnknownValueError:
            print("🤷 Could not understand the audio.")
        except sr.RequestError as e:
            print(f"❌ Could not request results: {e}")
    return ""

# Main loop
print("🎤 Say something... (say 'exit' to quit)")

while True:
    text = listen_command()
    if text:
        print(f"✅ You said: {text}")
        speak(f"You said {text}")
        if "exit" in text or "quit" in text:
            speak("Goodbye!")
            print("👋 Exiting...")
            break

