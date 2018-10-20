from scipy import sparse
from sklearn.linear_model import LogisticRegression
import numpy as np

class NBSVM_clf():

    def __init__(self,C=0.1,dual=True):
        self.C = C
        self.dual = dual
        self._clf = LogisticRegression(C=self.C,dual=self.dual)
    
    def log_count(self,x,y):
        p = (x[y==1].sum(0) +1) / ((y==1).sum() +1)
        q = (x[y==0].sum(0)+1) / ((y==0).sum() +1) 
        r = np.log(p/q)
        return sparse.csr_matrix(r)


    def predict_proba(self,features):
        x_nb = features.multiply(self.r)
        preds = self._clf.predict_proba(x_nb)
        return preds 

    def fit(self,features,labels):
        self.r = self.log_count(features,labels)
        x_nb = features.multiply(self.r)
        self._clf.fit(x_nb,labels)
        return self
