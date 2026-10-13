"""Speech-to-text (Google) and text-to-speech (piper)."""
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import speech_recognition as sr

from chatbot import config

recognizer = sr.Recognizer()


def transcribe(audio: bytes | sr.AudioData) -> str | None:
    """Transcribe WAV bytes or recorded audio; return None if speech is unintelligible."""
    if isinstance(audio, bytes):
        with sr.AudioFile(io.BytesIO(audio)) as source:
            audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language=config.LANGUAGE)
    except sr.UnknownValueError:
        return None


def tts_available() -> bool:
    return bool(config.PIPER_MODEL and shutil.which("piper"))


def synthesize(text: str) -> bytes:
    """Return WAV bytes of `text` spoken by piper."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.wav"
        cmd = ["piper", "--model", config.PIPER_MODEL, "--output_file", str(out)]
        if config.PIPER_CONFIG:
            cmd += ["--config", config.PIPER_CONFIG]
        subprocess.run(cmd, input=text, text=True, check=True, capture_output=True)
        return out.read_bytes()
