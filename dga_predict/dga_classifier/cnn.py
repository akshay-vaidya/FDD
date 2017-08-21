import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Dense
import sklearn
from sklearn.cross_validation import train_test_split


def build_model(max_features, maxlen):
    """Build CNN model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=75))
    model.add(Convolution1D(64, 3, border_mode='same'))
    model.add(Convolution1D(32, 3, border_mode='same'))
    model.add(Convolution1D(16, 3, border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def run(max_epoch=25, nfolds=10, batch_size=128):
    """Run train/test on logistic regression model"""
    dataset = pd.read_csv('dataset_all.csv', sep = ',')

    # Extract data and labels
    chars = dataset['domain'].tolist()
    chars = ''.join(chars)
    chars = list(set(chars))
    labels = dataset['class'].tolist()
    max_features = len(chars) + 1
    maxlen = np.max([len(x) for x in dataset['domain']])

    # Translating
    char_to_ix= {'a' : 0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,'1':26,'2':27,'3':27,'4':28,'5':29,'6':30,'7':31,'8':32,'9':33,'0':34,'_':35,'-':36,'.':37}
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    NUM_VOCAB = len(chars)
    NUM_CHARS = 75

    N = len(dataset.index)
    X = np.zeros((N, NUM_CHARS)).astype('int32')
    for i, r in dataset.iterrows():
        inputs = [char_to_ix[ch] for ch in r['domain']]
        length = len(inputs)
        X[i,:length] = np.array(inputs)

    # Convert labels to 0-1
    y = [0 if x == 'legit' else 1 for x in labels]

    final_data = []

    for fold in range(nfolds):
        print "fold %u/%u" % (fold+1, nfolds)
        X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels,
                                                                           test_size=0.2)

        print 'Build model...'
        model = build_model(max_features, maxlen)

        print "Train..."
        X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.05)
        best_iter = -1
        best_auc = 0.0
        out_data = {}

        for ep in range(max_epoch):
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)

            t_probs = model.predict_proba(X_holdout)
            t_auc = sklearn.metrics.roc_auc_score(y_holdout, t_probs)

            print 'Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc)

            if t_auc > best_auc:
                best_auc = t_auc
                best_iter = ep

                probs = model.predict_proba(X_test)

                out_data = {'y':y_test, 'labels': label_test, 'probs':probs, 'epochs': ep,
                            'confusion_matrix': sklearn.metrics.confusion_matrix(y_test, probs > .5)}

                print sklearn.metrics.confusion_matrix(y_test, probs > .5)
            else:
                # No longer improving...break and calc statistics
                if (ep-best_iter) > 2:
                    break

        final_data.append(out_data)
    model.save('cnn_model_1.h5')
    return final_data
                       
