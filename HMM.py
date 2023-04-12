import os

import numpy as np
from hmmlearn import hmm
import librosa
from librosa.feature import mfcc

import warnings

warnings.filterwarnings("ignore")  # to ignore n_init warnings from HMM model as warning wasn't considered significant

train_folder = os.path.join('audio', 'train')
test_folder = os.path.join('audio', 'test')

print(os.listdir(train_folder))


class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                                         covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)


# To fit the input into the HMM, we have to concatenate all the input matrices with the same label
# For smooth concatenation, we will trim the number of columns so the number of columns for each 2D matrix will be same

hmm_models = []
for label in os.listdir(train_folder):
    subfolder = os.path.join(train_folder, label)
    X = np.array([])
    y_words = []
    for filename in os.listdir(subfolder):
        filepath = os.path.join(subfolder, filename)

        try:
            audio, sampling_freq = librosa.load(filepath)
        except:
            print("Problem with " + filename)
        mfcc_features = mfcc(y=audio, sr=sampling_freq)
        if len(X) == 0:
            X = mfcc_features[:, :15]
        else:
            X = np.append(X, mfcc_features[:, :15], axis=0)
        y_words.append(label)

    hmm_trainer = HMMTrainer()
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, label))
    hmm_trainer = None

test_files = []

for file in os.listdir(test_folder):
    test_files.append(os.path.join(test_folder, file))

for input_file in test_files:
    audio, sampling_freq = librosa.load(input_file)

    # Extract MFCC features
    mfcc_features = mfcc(y=audio, sr=sampling_freq)
    mfcc_features = mfcc_features[:, :15]

    scores = []
    for item in hmm_models:
        hmm_model, label = item

        score = hmm_model.get_score(mfcc_features)
        scores.append(score)
    index = np.array(scores).argmax()
    print("\nFile name:", os.path.basename(input_file))
    print("Predicted:", hmm_models[index][1])
