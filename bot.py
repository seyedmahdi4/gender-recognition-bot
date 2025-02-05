import time
from os import remove, environ
from telebot import TeleBot
from requests import exceptions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import librosa
import numpy as np


def create_model(vector_length=128):
    """5 hidden dense layers from 256 units to 64, not the best model, but not bad."""

    model = Sequential()
    model.add(Input(shape=(vector_length,)))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))

    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    
    return model

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)

    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    
    return result


# construct the model
model = create_model()

# load the saved/trained weights
model.load_weights("model.h5")

def gender_recognition(filename: str):
    "predict the gender!"

    # extract features and reshape it
    features = extract_feature(filename, mel=True).reshape(1, -1)
    male_prob = model.predict(features, verbose=0)[0][0]
    female_prob = 1 - male_prob

    return male_prob, female_prob


# Get it from @BotFather
API_TOKEN = environ.get('API_TOKEN')

bot = TeleBot(API_TOKEN)

@bot.message_handler(content_types=['voice', 'audio'])
def handle_audio(message):
    voice_msg = message.voice if message.voice else message.audio
    
    if voice_msg.file_size < 10 * (1024 ** 2): # limit: 10MB
        try:
            file_path = bot.get_file(voice_msg.file_id).file_path
            downloaded_file = bot.download_file(file_path)

            voice_path = f"{time.strftime(r'%Y-%m-%d-%H:%M:%S', time.localtime())}-{file_path.replace('/', '-')}"
            with open(voice_path, 'wb') as voice_file:
                voice_file.write(downloaded_file)
                voice_file.close()

            male_prob, female_prob = gender_recognition(voice_path)
            remove(voice_path)

            bot.reply_to(message, "Result:  " + ("male 🙎🏻" if male_prob > female_prob else "female 🙎🏻‍♀️") + f"\n\nProbabilities:\n + Male: {male_prob*100:.2f}%\n + Female: {female_prob*100:.2f}%")
        except:
            bot.reply_to(message, "Failed.")
    else:
        bot.reply_to(message, "Upload Limit: 10MB.")

@bot.message_handler(func=lambda message: True)
def handle_non_audio(message):
    bot.reply_to(message, "Please send a voice 🎙")

try:
    bot.polling(none_stop=True)
except exceptions.ConnectTimeout:
    print("\n  [*] Turn on VPN !")
except exceptions.ReadTimeout:
    print("\n  [*] Turn on VPN !")
except exceptions.ConnectionError:
    print("\n  [*] You're offline :/")
