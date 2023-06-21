import speech_recognition as s
from gtts import gTTS
import os
import numpy as np
import transformers
import time

nlp = transformers.pipeline("conversational",
                            model="microsoft/DialoGPT-medium")


class ChatBot():
    def __init__(self, name):
        print("---starting up", name, "---")
        self.name = name

    def speech_to_text(self):
        sr = s.Recognizer()
        with s.Microphone() as m:
            sr.adjust_for_ambient_noise(m, duration=1)
            print("listening...")
            audio = sr.listen(m)
            self.text = ''
            try:
                self.text = sr.recognize_google(audio)
                print(self.text)

            except s.UnknownValueError:
                print(" Error")

            except s.RequestError as e:
                print("Request Error")

    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        os.system("start res.mp3")
        statbuf = os.stat("res.mp3")
        mbytes = statbuf.st_size / 1024
        duration = mbytes / 200
        time.sleep(int(50 * duration))
        os.remove("res.mp3")


if __name__ == "__main__":
    ai = ChatBot(name="Joy")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    st = "Hello I am Joy the AI, what can I do for you?"
    ai.text_to_speech(st)
    while True:

        ai.speech_to_text()

        if any(i in ai.text for i in ["thank", "thanks"]):
            res = np.random.choice(
                ["you're welcome!", "anytime!",
                 "no problem!", "cool!",
                 "I'm here if you need me!", "peace out!"])

        ## conversation
        else:
            chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256, padding_side)
            res = str(chat)
            res = res[res.find("bot >> ") + 6:].strip()

        ai.text_to_speech(res)