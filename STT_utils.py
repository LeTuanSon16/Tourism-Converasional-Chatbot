import openai
import speech_recognition as sr
import subprocess
from pydub import AudioSegment
from pathlib import Path
import requests
import os
from dotenv import load_dotenv,find_dotenv

import wave



load_dotenv(find_dotenv())

def listen():
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=0)
    r.dynamic_energy_threshold = True
    r.energy_threshold = 600

    with mic as source:
        print('\nListening...')
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    return audio


def recognize(audio):
    r = sr.Recognizer()
    try:
        user_input = r.recognize_google(audio, language='vi-VN')
        print(user_input)
        # Uncomment the following lines if you want to use the whisper function and check_quit function
        # if check_quit(user_input):
        #     return True
    except sr.UnknownValueError:
        return None

import speech_recognition as sr

def recognize_wav(file_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = r.record(source)
            user_input = r.recognize_google(audio_data, language='vi-VN')
            return user_input
            # Uncomment the following lines if you want to use the whisper function and check_quit function
            # if check_quit(user_input):
            #     return True
    except sr.UnknownValueError:
        return None


import openai

# Define the OpenAI API key
openai.api_key = "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"

def generate_response(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        temperature=0.8
    )

    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    print(f"\n{response}\n")
    return messages





def generate_audio_from_text_file(text_file, model_path, output_file, config_file):
    model_path = "/Users/phuongminh/Downloads/Sơn/Project/model.onnx"
    output_file = "/Users/phuongminh/Downloads/Sơn/Project/Voice_Test/Test.wav"
    config_file = "/Users/phuongminh/Downloads/Sơn/Project/config.json"
    with open(text_file, 'r') as file:
        text = file.read().strip()
    command = f"echo '{text}' | piper -m {model_path} --output_file {output_file} --config {config_file}"
    subprocess.run(command, shell=True)


def text_to_speech2(text):
    model_path = "/Users/phuongminh/Downloads/Sơn/Project/model.onnx"
    output_file = "/Users/phuongminh/Downloads/Sơn/Project/Voice_Test/Test.wav"
    config_path = "/Users/phuongminh/Downloads/Sơn/Project/config.json"
    webm_file_path = "temp_audio_play.mp3"
    command = f"echo '{text}' | piper -m {model_path} --output_file {output_file} --config {config_path}"
    subprocess.run(command, shell=True)

        # Load the WAV audio file
    audio = AudioSegment.from_wav(output_file)

        # Export the audio to MP3 format
    audio.export(webm_file_path, format="mp3")

    return webm_file_path









