"""Terminal voice chat using the microphone: `python cli.py`. Say "thoát" to quit."""
import speech_recognition as sr

from chatbot import rag
from chatbot.speech import recognizer, transcribe


def main() -> None:
    messages = []
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while True:
            print("\nListening...")
            text = transcribe(recognizer.listen(source))
            if not text:
                continue
            print(f"Bạn: {text}")
            if "thoát" in text.lower():
                break
            messages.append({"role": "user", "content": text})
            reply = rag.answer(messages)
            messages.append({"role": "assistant", "content": reply})
            print(f"Bot: {reply}")


if __name__ == "__main__":
    main()
