import pickle
from sklearn import svm

class vowelClassifier():
    def __init__(self):
        self.model = None
        with open('Python.txt', 'rb') as f:
            self.model = pickle.load(f)

        '''
        to do prediction:
        pred = vowelClassifier().model.predict(data)

        data: (N, 100) MFCC
        pred: list of 1 and 0, 1 for vowel frames
        '''
        