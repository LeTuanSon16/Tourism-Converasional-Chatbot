import openai
import speech_recognition as sr

key = "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"
openai.api_key = key

personality = "p.txt"

with open(personality, "r") as file:
    mode = file.read()

messages = [
    {"role": "system", "content": f"{mode}"}
]

r = sr.Recognizer()
mic = sr.Microphone(device_index=0)
r.dynamic_energy_threshold = True
r.energy_threshold = 600

# def whisper(audio):
#     with open('speech.wav', 'wb') as f:
#         f.write(audio.get_wav_data())
#     speech = open('speech.wav', 'rb')
#     wcompletion = openai.Audio.transcribe(
#         model="whisper-1",
#         file=speech
#     )
#     user_input = wcompletion['text']
#     print(user_input)
#     return user_input


# def check_quit(user_input: str):
#     if "thoát" in user_input.lower().split():
#         return True
#     return False


while True:
    with mic as source:
        print('\nListening...')
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)
        try:
            user_input = r.recognize_google(audio, language='vi-VN')
            print(user_input)

        except:
            continue

    messages.append({"role": "user", "content": user_input})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=messages,
        temperature=0.8
    )

    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    print(f"\n{response}\n")

    # if check_quit(user_input):
    #     break