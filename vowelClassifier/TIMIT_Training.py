from utils import *

TIMIT_PATH = "TIMIT"
AUDIO_NUM = 500     # total num of audio
size = 100          # number of test audio

# load dataset
# TIMIT: https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech
TIMIT = TIMIT(TIMIT_PATH)
TIMIT.getMetadata()
TIMIT.load_audio(AUDIO_NUM)

# build classifier
phonClassifier = Phoncode_Classifier(TIMIT.audio_MFCC[:size], TIMIT.audio_t_labels_binary[:size])
phonClassifier.train(classifier = "SVM")

# predict and get accuracy
pred = phonClassifier.predict(TIMIT.audio_MFCC[size:])
phonClassifier.get_ACC(TIMIT.audio_t_labels_binary[size:])
