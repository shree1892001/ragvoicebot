import speech_recognition as sr
import pyttsx3


def audio_to_text(timeout: int = 5) -> str:
    """Convert audio from microphone to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
            print("🧠 Transcribing...")
            text = recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
        except sr.WaitTimeoutError:
            print("⏳ Listening timed out")
            return ""
        except sr.UnknownValueError:
            print("🤷 Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"🌐 Could not request results; {e}")
            return ""


def text_to_audio(text: str):
    """Convert text to audio using pyttsx3."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)  # speed
        engine.setProperty("volume", 1)  # max volume
        print("🔊 Speaking response...")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ Error in TTS: {e}")
