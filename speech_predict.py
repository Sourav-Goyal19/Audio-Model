import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model/emotion_model.h5');

def extract_mfcc(audio_path, max_length=162):
    audio, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=0)
    if mfcc_mean.shape[0] > max_length:
        mfcc_mean = mfcc_mean[:max_length]
    else:
        mfcc_mean = np.pad(mfcc_mean, (0, max_length - mfcc_mean.shape[0]), mode='constant')
    mfcc_mean = np.expand_dims(mfcc_mean, axis=0)
    mfcc_mean = np.expand_dims(mfcc_mean, axis=-1)
    return mfcc_mean

def audio_predict(audio_path: str) -> str:
    test_mfcc = extract_mfcc(audio_path)
    predictions = model.predict(test_mfcc)
    emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    predicted_emotion = emotion_labels[np.argmax(predictions[0])]
    return predicted_emotion

