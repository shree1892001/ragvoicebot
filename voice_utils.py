import speech_recognition as sr
import pyttsx3
import os
from datetime import datetime


def audio_to_text(timeout: int = 5) -> str:
    """Convert audio from microphone to text using SpeechRecognition. Also saves the transcription to a text file."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        try:
            audio = recognizer.listen(source, timeout=timeout)
            print("🧠 Transcribing...")
            text = recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            # Save transcription
            transcripts_dir = "transcripts"
            os.makedirs(transcripts_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_path = os.path.join(transcripts_dir, f"transcript_{timestamp}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(text.strip() + "\n")
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
    """Convert text to audio using pyttsx3, using an Indian English voice if available."""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)  # speed
        engine.setProperty("volume", 1)  # max volume
        # Try to select an Indian English voice
        selected_voice = None
        for voice in engine.getProperty('voices'):
            if ('en-in' in voice.id.lower() or 'hindi' in voice.id.lower() or 'india' in voice.name.lower()):
                selected_voice = voice
                break
        if selected_voice:
            engine.setProperty('voice', selected_voice.id)
            print(f"🔊 Using Indian voice: {selected_voice.name}")
        else:
            print("🔊 Indian voice not found, using default voice.")
        print("🔊 Speaking response...")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ Error in TTS: {e}")
