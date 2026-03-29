import sklearn
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import pywt
import sklearn.svm
import joblib
import antropy
import math
import matplotlib
import scipy,scipy.signal
from matplotlib import pyplot

rawdata = pd.read_csv("formatted.csv",index_col=0)
label = rawdata["label"]
label[label!=5]=0
label[label==5]=1

splitEEG = []
splitlabel = []
rawwave = rawdata.drop("label",axis="columns")
row,column = rawwave.shape
feature = []

def notch_filter(data,rate,freq,quality):
    return scipy.signal.filtfilt(*scipy.signal.iirnotch(freq / (rate / 2), quality), data)

for x in range(row):
    for y in range(4):
        splitEEG.append(np.array(rawdata.iloc[x,1024*y:1024*(y+1)-1]))
        splitlabel.append(label.iloc[x])


for x in range(len(splitEEG)):
    features = []
    filtered = notch_filter(splitEEG[x],173.61,50,30)
    cAf,cDf = pywt.dwt(splitEEG[x],"db4")
    cA2,cD2 = pywt.dwt(cDf,"db4")
    cA3,cD3 = pywt.dwt(cD2,"db4")
    #features.append(antropy.spectral_entropy(cAf,sf=173.61))
    features.append(np.std(splitEEG[x]))
    features.append(max(cAf)-min(cAf))
    concatdwt = np.concatenate([cAf,cA2,cA3,cD3])
    features.append(antropy.katz_fd(concatdwt))
    features.append(np.mean(splitEEG[x]))
    #features.append(np.var(splitEEG[x]))
    #features.append(antropy.spectral_entropy(concatdwt,sf=173.61))
    #features.append(np.var(splitEEG[x]))
    feature.append(features)

accuracyarray = []
precisionarray = []
recallarray = []
f1array = []

for x in range(10):
    testindex = list(range(40*x,40*(x+1)))+list(range(400+40*x,400+40*(x+1)))+list(range(800+40*x,800+40*(x+1)))+list(range(1200+40*x,1200+40*(x+1)))+list(range(1600+40*x,1600+40*(x+1)))
    trainindex = list(range(2000))
    for y in testindex:
        trainindex.remove(y)
    trainfeature = [feature[a] for a in trainindex]
    trainlabel = [splitlabel[a] for a in trainindex]
    testfeature = [feature[a] for a in testindex]
    testlabel = [splitlabel[a] for a in testindex]
    svm = sklearn.svm.SVC(C=1,decision_function_shape="ovo",kernel="linear")
    svm.fit(trainfeature,trainlabel)
    predicted = svm.predict(testfeature)
    accuracy = sklearn.metrics.accuracy_score(testlabel,predicted)
    precision = sklearn.metrics.precision_score(testlabel,predicted)
    recall = sklearn.metrics.recall_score(testlabel,predicted)
    f1 = sklearn.metrics.f1_score(testlabel,predicted)
    accuracyarray.append(accuracy)
    precisionarray.append(precision)
    recallarray.append(recall)
    f1array.append(f1)
    print(f"The accuracy of {x+1}-th iteration is {accuracy}")
    print(f"The precision of {x+1}-th iteration is {precision}")
    print(f"The recall of {x+1}-th iteration is {recall}")
    print(f"The f1 of {x+1}-th iteration is {f1}")
    print("---------------------")


print(np.average(accuracyarray))
print(np.average(precisionarray))
print(np.average(recallarray))
print(np.average(f1array))
