import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, LSTM, Bidirectional
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import sys, os, time, json, datetime, glob

from logzero import logger
import warnings
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
K.set_session(K.tf.compat.v1.Session(config=K.tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8,
                                                   inter_op_parallelism_threads=8)))

import numpy as np
import pandas as pd
from pathlib import Path

from art.attacks.evasion import FastGradientMethod, SaliencyMapMethod, CarliniL0Method, CarliniL2Method, CarliniLInfMethod
from art.estimators.classification import KerasClassifier


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

warnings.simplefilter("ignore", category=FutureWarning)


if __name__ == '__main__':
    model = Sequential()
    model.add(Dense(64, input_dim = 19, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(41, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
    	metrics=["accuracy"])

    y = []
    packets = [] 
    i = 0
    
    devs = ['appkettle','blink-security-hub','bosiwo-camera-wifi','lefun-cam-wired','insteon-hub',
            'echoplus','meross-dooropener','smartlife-bulb','xiaomi-ricecooker','ubell-doorbell',
            'appletv','tplink-bulb','google-home','icsee-doorbell','t-wemo-plug','echospot',
            'nest-tstat','sousvide','smartlife-remote','netatmo-weather-station','lgtv-wifi',
            'wansview-cam-wired','xiaomi-plug','xiaomi-hub','lightify-hub','bosiwo-camera-wired',
            'tplink-plug2','allure-speaker','honeywell-thermostat','smarter-coffee-mach','roku-tv',
            'yi-camera','firetv','echodot','smartthings-hub','reolink-cam-wired','t-philips-hub',
            'switchbot-hub','ring-doorbell','blink-camera','samsungtv-wired']

    features = ['srcPort', 'destPort',
       'bytes_out', 'num_pkts_out', 'bytes_in', 'num_pkts_in', 'f_ipt_mean',
       'f_ipt_std', 'f_ipt_var', 'f_ipt_skew', 'f_ipt_kurtosis', 'f_b_mean',
       'f_b_std', 'f_b_var', 'f_b_skew', 'f_b_kurtosis', 'duration', 'pr',
       'domainId']
    label = 'deviceId'


    #weeksToProcess = list(range(44,53)) + list(range(1,19))
    weeks = [list(range(44,53)), list(range(1,10)), list(range(10,19))]
    weeksToProcess = weeks[0] + weeks[1] + weeks[2]
    #weeksToProcess = weeks[0]
    test = True
    #test = True

    modelName = "model_mar-apr/model.50-0.07.h5"

    df = pd.read_csv(sys.argv[1], parse_dates = ['time_start'], low_memory=False)
    df.fillna(0, inplace=True)
    

    trainCon = df['time_start'].dt.week.isin(weeks[int(sys.argv[2])])

    testCons = pd.date_range('2019-11-01', '2020-05-01', freq = '1Y').tolist()
    lb = LabelBinarizer()
    lb.fit(range(1,42))

    if test:
        model = keras.models.load_model(sys.argv[3])
        advClassifier = KerasClassifier(model=model)
        print("model loaded")
        for date in testCons:
            testCon = df['time_start'].dt.week == pd.Timestamp(date).week
            #testCon = df['time'].dt.date == pd.Timestamp(date).date
            #testDF = df[testCon]
            testDF = df

            X = testDF[features]
            y = testDF[label]

            if len(X) == 0:
                continue

            scaler = StandardScaler().fit(df[trainCon][features])
            X = scaler.transform(X)
            
            labels = lb.transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels)

            

        
            predictions = advClassifier.predict(x=X_test, batch_size=256)
        
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

            
        
            print("Accuracy on benign test examples: {}%".format(accuracy * 100))
            
            print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))#, labels=devs))
            print("Week", pd.Timestamp(date).week,
                  "Accuracy: ", metrics.accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)), 
                  "F1: ", metrics.f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='micro'))

            print("FGSM TESTS BEGIN")


            attackFGSM = FastGradientMethod(estimator=advClassifier, eps=0.1)
            x_test_adv_FGSM = attackFGSM.generate(x=X_test)
            print(x_test_adv_FGSM)
            np.savetxt("fgsm.csv", x_test_adv_FGSM, delimiter=",")

            predictions = advClassifier.predict(x_test_adv_FGSM, batch_size=256)

            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on adversarial FGSM test examples: {}%".format(accuracy * 100))
            print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))#, labels=devs))
            print("Week", pd.Timestamp(date).week,
                  "Accuracy: ", metrics.accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)), 
                  "F1: ", metrics.f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='micro'))

            print("JSMA TESTS BEGIN")
            attackJSMA = SaliencyMapMethod(classifier=advClassifier)
            x_test_adv_JSMA = attackJSMA.generate(x=X_test)
            print(x_test_adv_JSMA)
            np.savetxt(f"jsma-{pd.Timestamp(date).week}.csv", x_test_adv_JSMA, delimiter=",")

            predictions = advClassifier.predict(x_test_adv_JSMA, batch_size=256)

            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on adversarial JSMA test examples: {}%".format(accuracy * 100))
            print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))#, labels=devs))
            print("Week", pd.Timestamp(date).week,
                  "Accuracy: ", metrics.accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)), 
                  "F1: ", metrics.f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='micro'))
            
            print("C and W 0 TESTS BEGIN")
            attackCW0 = CarliniL0Method(classifier=advClassifier)
            x_test_adv_CW0 = attackCW0.generate(x=X_test)
            print(x_test_adv_CW0)
            np.savetxt(f"cw0-{pd.Timestamp(date).week}.csv", x_test_adv_CW0, delimiter=",")


            predictions = advClassifier.predict(x_test_adv_CW0, batch_size=256)

            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on adversarial C&W0 test examples: {}%".format(accuracy * 100))
            print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))#, labels=devs))
            print("Week", pd.Timestamp(date).week,
                  "Accuracy: ", metrics.accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)), 
                  "F1: ", metrics.f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='micro'))

            print("C and W 2 TESTS BEGIN")
            attackCW2 = CarliniL2Method(classifier=advClassifier)
            x_test_adv_CW2 = attackCW2.generate(x=X_test)
            print(x_test_adv_CW2)
            np.savetxt(f"CW2-{pd.Timestamp(date).week}.csv", x_test_adv_CW2, delimiter=",")

            predictions = advClassifier.predict(x_test_adv_CW2, batch_size=256)

            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on adversarial C&W2 test examples: {}%".format(accuracy * 100))
            print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))#, labels=devs))
            print("Week", pd.Timestamp(date).week,
                  "Accuracy: ", metrics.accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)), 
                  "F1: ", metrics.f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='micro'))

            print("C and W INF TESTS BEGIN")
            np.savetxt(f"CWINF-{pd.Timestamp(date).week}.csv", x_test_adv_CWINF, delimiter=",")
            attackCWINF =  CarliniLInfMethod(classifier=advClassifier)
            x_test_adv_CWINF = attackCWINF.generate(x=X_test)
            print(x_test_adv_CWINF)

            predictions = advClassifier.predict(x_test_adv_CWINF, batch_size=256)

            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on adversarial C&WINF test examples: {}%".format(accuracy * 100))
            print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))#, labels=devs))
            print("Week", pd.Timestamp(date).week,
                  "Accuracy: ", metrics.accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)), 
                  "F1: ", metrics.f1_score(y_test.argmax(axis=1), predictions.argmax(axis=1), average='micro'))



    
            

    else:
        X = df[trainCon][features]
        y = df[trainCon][label]
    
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        #labels = lb.transform(y)
        labels = y - 1#lb.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels)

        my_callbacks = [
                #tf.keras.callbacks.EarlyStopping(patience=2),
                keras.callbacks.ModelCheckpoint(filepath='advmodel.{epoch:02d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1),
                #tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            ]
        advClassifier = KerasClassifier(model=model)

        advClassifier.fit(X_train, y_train, batch_size=128, nb_epochs=8, callbacks=my_callbacks)
       
        predictions = advClassifier.predict(x=X_test)
       
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
       
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

        attack = FastGradientMethod(estimator=advClassifier, eps=0.2)
        x_test_adv = attack.generate(x=X_test)

        predictions = advClassifier.predict(x_test_adv)

        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))



        #H = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=15, verbose=1, callbacks=my_callbacks) 


