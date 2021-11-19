import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import numpy as np
from scipy.io import wavfile
import pandas as pd
from sklearn import svm
import pickle

n_fft = 2048
n_mfcc = 100

def mfcc_from_amp(amp, sr, n_mfcc):
    power = np.abs(amp) ** 2
    mel_spec = librosa.feature.melspectrogram(S=power, sr=sr)
    return librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)


def mfcc(filename):
    fs, audio = wavfile.read(filename)
    f, t, zxx = stft(audio, fs, nperseg=n_fft)
    amp = np.abs(zxx)
    return fs, t, mfcc_from_amp(amp, fs, n_mfcc)



class TIMIT():
    def __init__(self, TIMIT_path):
        # path of CVS files
        self.dirPath = TIMIT_path + "/"
        self.metadataCSVpath = self.dirPath + "train_data.csv"
        # CVS dataframes
        self.metadata = None
        self.metadataWave = None

        self.audio_MFCC = []        # audio frames in MFCC
        self.audio_Phoncode = []    # audio frames's corresponding phoncodes
        self.audio_t = []           # time stamp for each audio frame
        self.audio_t_labels = []    # each audio frames' phoncodes converted to numbers
        self.audio_t_labels_binary = []    # each audio frames' phoncodes converted to numbers based on vowel/non-vowel
        self.audio_t_labels_type = []    # each audio frames' phoncodes converted to numbers based on type
        self.phonDict = None        # key: phoncode, value: phoncode types
        self.phon2num = None        # key: phonecode, value: integers
        self.type2num = None        # key: type, value: type integers
        self.sr = None              # sampling rate

        def get_phonDict(file):
            '''
            input
                file: phoncode dictionary file variable
            output
                phonDict: key: phoncode, value: phoncode types
                phon2num_dict: key: phonecode, value: integers
            '''
            lines = file.readlines()
            phonDict = {}
            phon2num_dict = {}
            type2num_dict = {}
            i = 0
            j = 0
            for line in lines:
                if line[-1:] == "\n":
                    line = line[:-1]
                
                phon, type = line.split(" ")
                phonDict[phon] = type
                if type not in type2num_dict:
                    type2num_dict[type] = j 
                    j += 1
                if phon not in phon2num_dict:
                    phon2num_dict[phon] = i
                    i += 1
            return phonDict, phon2num_dict, type2num_dict

        # get phonDict and phon2num
        with open("phoncode_dict.txt") as file:
            self.phonDict, self.phon2num, self.type2num = get_phonDict(file)


    def getMetadata(self, doshuffle=True):
        ''' 
        input
            doshuffle: shuffle the csv
        '''
        self.metadata = pd.read_csv(self.metadataCSVpath)

        # if doshuffle:
        #     self.metadata = shuffle(self.metadata)

        self.metadata = self.metadata.query("is_converted_audio == True or is_phonetic_file == True")
        self.metadataWave = self.metadata.query("is_converted_audio == True")
        self.metadata.reset_index(drop=True, inplace=True)
        self.metadataWave.reset_index(drop=True, inplace=True)
        
    
    def load_audio(self, n=10):
        ''' 
        load audio into MFCC and get phoncode labels 
        '''
        def get_label(file):
            ''' 
            input 
                file: phonocde label file of an audio
            out
                label: list of (start_time, end_time, phoncode) (time in unit of wave frames)
            '''
            lines = file.readlines()
            label = []
            for line in lines:
                if line[-1:] == "\n":
                    line = line[:-1]
                
                s, e, l  = line.split(" ")
                label.append([float(s),float(e), l])
            return np.array(label)
        
        def get_phoncode(t, label):
            ''' 
            input
                t: time stamps for MFCC
                label: phoncode labeling of the audio cooresp to the t
            output
                phoncode: phoncode labeling for each time stamp (-1 if not found)
            '''
            phoncode = []
            phoncodeType = []
            phoncodeBinary = []
            for i, x in enumerate(t):
                flag = True
                for s,e,p in label:
                    if x >= float(s) and x < float(e):
                        phoncode.append(self.phon2num[p])
                        if self.phonDict[p] == "Vowels":
                            phoncodeBinary.append(1)
                        else:
                            phoncodeBinary.append(0)
                        phoncodeType.append(self.type2num[self.phonDict[p]])
                        flag = False
                if flag:
                    phoncode.append(-1)
                    phoncodeBinary.append(0)
                    phoncodeType.append(-1)
            return phoncode, phoncodeBinary, phoncodeType
        
        i = 0   # number of audios counter
        while i != n:
            audio_fn = self.dirPath + "data/" + self.metadataWave["path_from_data_dir"][i]
            Phn_fn = audio_fn[:-8] + ".PHN"
            with open(Phn_fn) as file:
                label = get_label(file)
                fs, t, MFCC = mfcc(audio_fn)
                # fs, t, MFCC = STFT(audio_fn)
                self.sr = fs
                self.audio_MFCC.append(MFCC.T)
                self.audio_t.append(t*self.sr)
                self.audio_Phoncode.append(label)
                phonNum, binaryNum, typeNum = get_phoncode(t*self.sr, label)
                self.audio_t_labels.append(phonNum)
                self.audio_t_labels_type.append(typeNum)
                self.audio_t_labels_binary.append(binaryNum)
                i += 1
            


class Phoncode_Classifier():
    def __init__(self, data, labels):
        self.X = []     # audio in whatever from
        self.y = []     # phoncode labels
        
        # each audio will have different size, concate them together
        for i, x in enumerate(data):
            self.X += list(x)
            self.y += list(labels[i])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        self.model = None           # classification model
        self.prediction = None      # prediction

        self.PCA = -1

        print("X", np.shape(self.X))
        print("y", np.shape(self.y))

    
    def train(self, classifier="SVM", PCAdim=-1):
        X = self.X
        y = self.y
        # if PCAdim != -1:
        #     self.PCA = PCAdim
        #     X = PCA(n_components=PCAdim).fit(self.X.T).components_
        #     X = X.T
        # if classifier == "MLP":
        #     self.model = MLPClassifier().fit(X, y)
        # if classifier == "NB":
        #     self.model = GaussianNB().fit(X, y)
        # else:    
        #     self.model = svm.SVC().fit(X, y)
        self.model = svm.SVC().fit(X, y)

        # save model to file
        file = open('vowelClassifier.txt', 'wb')
        pickle.dump(self.model, file)
        file.close()

        

    def predict(self, X_test):
        test_set = []
        for i,x in enumerate(X_test):
            test_set += list(x)
        # test_set = np.array(test_set) 
        # if self.PCA != -1:
        #     test_set = PCA(n_components=self.PCA).fit(test_set.T).components_
        #     test_set = test_set.T
        self.prediction = self.model.predict(test_set)
        return self.prediction

    def get_ACC(self, y_label):
        label = []
        for i, y in enumerate(y_label):
            label += list(y)
        
        acc = len(np.where((self.prediction - label) == 0)[0])/len(self.prediction)
        print("Accuracy:", acc)
