import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

x, y = make_multilabel_classification(n_samples=10000, n_features=20,
                                      n_classes=5, random_state=88)

for i in range(5): 
    print(x[i]," =====> ", y[i])
    
xtrain, xtest, ytrain, ytest=train_test_split(x, y, train_size=0.8, random_state=88)
print(len(xtest)) 


classifier = MultiOutputClassifier(XGBClassifier())
clf = Pipeline([('classify', classifier)
               ])
print (clf)


clf.fit(xtrain, ytrain)
print(clf.score(xtrain, ytrain))

yhat = clf.predict(xtest)

auc_y1 = roc_auc_score(ytest[:,0],yhat[:,0])
auc_y2 = roc_auc_score(ytest[:,1],yhat[:,1])
auc_y3 = roc_auc_score(ytest[:,2],yhat[:,2])
auc_y4 = roc_auc_score(ytest[:,3],yhat[:,3])
auc_y5 = roc_auc_score(ytest[:,4],yhat[:,4])
print("ROC AUC y1: %.4f, y2: %.4f, y3: %.4f, y4: %.4f, y5: %.4f" % (auc_y1, auc_y2, auc_y3, auc_y4, auc_y5))


cm_y1 = confusion_matrix(ytest[:,0],yhat[:,0])
cm_y2 = confusion_matrix(ytest[:,1],yhat[:,1])
cm_y3 = confusion_matrix(ytest[:,2],yhat[:,2])
cm_y4 = confusion_matrix(ytest[:,3],yhat[:,3])
cm_y5 = confusion_matrix(ytest[:,4],yhat[:,4])
print(cm_y1)


cr_y1 = classification_report(ytest[:,0],yhat[:,0])
cr_y2 = classification_report(ytest[:,1],yhat[:,1])
cr_y3 = classification_report(ytest[:,2],yhat[:,2])
cr_y4 = classification_report(ytest[:,3],yhat[:,3])
cr_y5 = classification_report(ytest[:,4],yhat[:,4])
print (cr_y1)




