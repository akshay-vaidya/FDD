"""Train and test LSTM classifier"""
import dga_classifier.data as data
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.cross_validation import train_test_split


def build_model(max_features, maxlen):
    """Build LSTM model"""
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')

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
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    NUM_VOCAB = len(chars)
    NUM_CHARS = maxlen

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
        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.05)
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
    print("testing the real data..")
    reader = csv.reader(open("/home/akshayaa/Documents/from-20170101.txt", "rt"))
    real_data=[]
    for row in reader: 
        real_data.append(row[0][9:])
    i=0
    X_real = np.zeros((N, NUM_CHARS)).astype('int32')
    for r in real_data:
    	inputs_real = [char_to_ix[ch] for ch in r]
    	length_real = len(inputs_real)
    	X_real[i,:length_real] = np.array(inputs_real)
        i+=1
    probs_real=[]
    probs_real.append(model.predict_proba(X_real))
    thefile = open('test.txt', 'w')
    for item in probs_real:
        thefile.write("%s\n" % item)

    return final_data
