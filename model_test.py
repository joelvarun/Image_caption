import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import random
from urllib.request import urlopen
import requests
from io import BytesIO
import pandas as pd
import cloudpickle as cp
import streamlit as st


#load data



model = load_model('https://drive.google.com/file/d/1dFXQlFvfmKN1YjG3_k8U3ZpIZTkfba_g/view?usp=sharing')

features = pickle.open(urlopen("https://drive.google.com/file/d/1F8FGdOdCOYiNRcFxZdhKzaXcJSFUOaW1/view?usp=sharing"))
words_to_index = cp.load(urlopen("https://github.com/ok1341/Image-Captioning-files/blob/eee94e6288157c94532f50208ca89af5be6cf31c/words.pkl"))

images = ""
max_length = 33
index_to_words = pickle.load(urlopen("https://github.com/ok1341/Image-Captioning-files/blob/eee94e6288157c94532f50208ca89af5be6cf31c/words1.pkl"))


#generate captions
def image_captioning(picture):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([picture,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def org_caption(id):
    with open("captions.txt", 'r') as f:
        caps = [line.rstrip() for line in f]
        original = caps[id]
    return original


#show picture an caption
def show():
    z = random.randint(0,20)
    pic = list(features.keys())[z]
    image = features[pic].reshape((1,2048))
    x = plt.imread(images + pic)
    st.image(x)
    st.write("Generated caption:", image_captioning(image))
    st.write("Original caption:", org_caption(z + 1))
    
if st.button("Generate caption"):
    show()
 