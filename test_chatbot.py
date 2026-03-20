import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
import random
import json
import pickle
import os
import threading

from tensorflow.keras.models import load_model
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

import customtkinter as ctk

# -----------------------------
# LOAD MODEL & FILES
# -----------------------------
model = load_model('best_chatbot_model.h5')
intents = json.loads(open('new.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# -----------------------------
# NLP FUNCTIONS
# -----------------------------

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        return classes[results[0][0]]
    else:
        return "fallback"


def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return intent['responses'][0]

    return "Sorry, I didn't understand."


def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "voice.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)


# -----------------------------
# VOICE LISTENING
# -----------------------------

def start_listening():
    threading.Thread(target=listen_voice).start()


def listen_voice():

    recognizer = sr.Recognizer()

    add_bot_message("Listening...")

    with sr.Microphone() as source:

        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:

        user_input = recognizer.recognize_google(audio)

        add_user_message(user_input)

        tag = predict_class(user_input)

        response = get_response(tag)

        add_bot_message(response)

        speak(response)

    except:

        add_bot_message("Sorry, I didn't catch that.")
        speak("Sorry, I didn't catch that.")


# -----------------------------
# CHAT MESSAGE DESIGN
# -----------------------------

def add_user_message(text):

    frame = ctk.CTkFrame(chat_box, fg_color="#2B7A78", corner_radius=10)

    label = ctk.CTkLabel(
        frame,
        text=text,
        wraplength=900,
        justify="left",
        text_color="white"
    )

    label.pack(padx=10, pady=5)
    frame.pack(anchor="e", pady=5, padx=10)


def add_bot_message(text):

    frame = ctk.CTkFrame(chat_box, fg_color="#3A3A3A", corner_radius=10)

    label = ctk.CTkLabel(
        frame,
        text=text,
        wraplength=900,
        justify="left"
    )

    label.pack(padx=10, pady=5)
    frame.pack(anchor="w", pady=5, padx=10)


# -----------------------------
# APP WINDOW
# -----------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()

app.title("AI Voice Companion")
app.geometry("700x650")

# -----------------------------
# TITLE
# -----------------------------

title = ctk.CTkLabel(
    app,
    text="🤖 AI Voice Companion",
    font=ctk.CTkFont(size=28, weight="bold")
)

title.pack(pady=20)

# -----------------------------
# CHAT AREA
# -----------------------------

chat_frame = ctk.CTkFrame(app)

chat_frame.pack(padx=20, pady=10, fill="both", expand=True)

chat_box = ctk.CTkScrollableFrame(chat_frame)

chat_box.pack(fill="both", expand=True, padx=10, pady=10)

# -----------------------------
# BUTTON AREA
# -----------------------------

button_frame = ctk.CTkFrame(app)

button_frame.pack(pady=15)

listen_btn = ctk.CTkButton(
    button_frame,
    text="🎤 Speak",
    command=start_listening,
    width=140,
    height=40,
    font=ctk.CTkFont(size=15, weight="bold")
)

listen_btn.grid(row=0, column=0, padx=15)

exit_btn = ctk.CTkButton(
    button_frame,
    text="Exit",
    fg_color="red",
    hover_color="#8B0000",
    command=app.destroy,
    width=140,
    height=40,
    font=ctk.CTkFont(size=15, weight="bold")
)

exit_btn.grid(row=0, column=1, padx=15)

# -----------------------------
# START MESSAGE
# -----------------------------

add_bot_message("Hello! Press the Speak button and talk with me.")

app.mainloop()
