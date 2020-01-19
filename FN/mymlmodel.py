import pandas as pd
import numpy as np
from sklearn.externals import joblib

def maketextpred(textfeed):
    model = joblib.load('newsmodel.pkl') 
    tfidf_vect = joblib.load('vectorizer.pickle')
    mylist = []
    mylist.append(textfeed)
    mylist = list(mylist)
    df2 = pd.DataFrame(mylist, columns = ['textinput'])
    myX = df2.textinput
    mytest = tfidf_vect.transform(myX)
    #return mytest
    return (model.predict(mytest)[0])

