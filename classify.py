import message_cleaner
import models
import argparse
import os
import glob
import numpy as np
import re, string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def interpret(userinput,model,vectorizer,names):
    features = vectorizer.transform(np.array([userinput]))
    prob = float(model.predict_proba(features)[0,1])

    switcher = {
        prob<=0.25: 'Definitely sounds like {}, confidence: {:.3f}'.format(names[0],1-prob),
        0.25<prob<=0.45: 'Kinda sounds like {}, confidence: {:.3f}'.format(names[0],1-prob),
        0.45<prob<=0.55: 'Really could be either of you',
        0.55<prob<=0.75: 'Kinda sounds like {}, confidence: {:.3f}'.format(names[1],prob),
        0.75<prob: 'Definitely sounds like {}, confidence: {:.3f}'.format(names[1],prob),
    }
    return switcher[True]

def train(data,names,C=4.0,dual=True):

    X = np.array(data['Text'])
    Y = np.array(list(map(lambda x:0 if x==names[0] else 1,data['Label'])))

    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
    def tokenize(s): return re_tok.sub(r' \1 ', s).split()

    vectorizer = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
    
    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.25)
    trainvec = vectorizer.fit_transform(xtrain)
    testvec = vectorizer.transform(xtest)

    oof_preds = np.zeros(len(ytrain))
    test_preds = np.zeros(len(ytest))
    kfold = KFold(n_splits=5)
    model = models.NBSVM_clf(C=C,dual=dual)

    for train_i, val_i in kfold.split(trainvec,ytrain):
        train_x,train_y = trainvec[train_i],ytrain[train_i]
        valid_x = trainvec[val_i]

        model.fit(train_x,train_y)
        oof_preds[val_i] = model.predict_proba(valid_x)[:,1]
        test_preds += model.predict_proba(testvec)[:,1] / kfold.n_splits 

    model.fit(trainvec,ytrain)    
    test_preds = model.predict_proba(testvec)[:,1]

    val_score = roc_auc_score(ytrain,oof_preds)
    test_score = roc_auc_score(ytest,test_preds)
    print('Validation score: {:.3f}'.format(val_score))
    print('Test score: {:.3f}'.format(test_score))

    return model, vectorizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify your whatsapp texts')
    parser.add_argument('datafolder',type=str,help='folder name containing the data')
    args = parser.parse_args()

    folderpath = os.path.join(os.getcwd(),args.datafolder)
    files = glob.glob(folderpath+r'\*.txt')

    if len(files) < 1:
        raise Exception('No text files found in the directory: {}'.format(folderpath)) 
    
    if len(files)>1:
        print('{} files found for processing. Select one: (enter number)'.format(len(files))) 
        for i,f in enumerate(files):
            print('{}: {}'.format(f,i))
        
        selection = int(input())
        datafile = files[selection]
    else:
        datafile = files[0]

    cleaner = message_cleaner.whatsapp_cleaner()
    data,names = cleaner.clean_texts(datafile)
    model,vectorizer = train(data,names)
    
    while(True):
        print('Enter message: ')
        userinput = input()
        prediction = interpret(userinput,model,vectorizer,names)
        print(prediction)
    


