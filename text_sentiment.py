import glob
import numpy as np
import pandas as pd
import sklearn
from paths import train_path, test_path

df = pd.DataFrame()
for f in glob.glob(train_path):
    df = df.append(pd.read_csv(f,sep=';'),ignore_index=True)

def create_data_df(df_name,data_path):
    
    for f in glob.glob(data_path):
        df_name = df_name.append(pd.read_csv(f,sep=';'),ignore_index=True)
    
    if 'Speech' not in df_name.columns:
        df_name['Speech'] = ''    
    if 'speech' in df_name.columns:
        df_name['Speech'] = df_name[['Speech','speech']].fillna('').sum(axis=1)   
    if 'transcription' in df_name.columns:
        df_name['Speech'] = df_name[['Speech','transcription']].fillna('').sum(axis=1)
    
    if 'sentimentAnnotation' not in df_name.columns:
        df_name['sentimentAnnotation'] = 0    
    if 'sentimentAnnotations' in df_name.columns:
        df_name['sentimentAnnotation'] = df_name[['sentimentAnnotation','sentimentAnnotations']].fillna(0).sum(axis=1)
    if 'sentimentannotations' in df_name.columns:
        df_name['sentimentAnnotation'] = df_name[['sentimentAnnotation','sentimentannotations']].fillna(0).sum(axis=1)
    
    df_name = df_name.query('sentimentAnnotation != 0')
    
    df_name = df_name[['Speech','sentimentAnnotation']].reset_index(drop=True)  
    return df_name


df = pd.DataFrame() #training data frame
df_t = pd.DataFrame() #testing data frame
df = create_data_df(df,train_path)
df_t = create_data_df(df_t,test_path)


X_trn, y_trn = df[['Speech']],df[['sentimentAnnotation']]

# countVectorizer initialization
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None, stop_words = None,lowercase = True,max_features = 5000) 

# create bag of words vector for the training set using countVectorizer
train_data_features = vectorizer.fit_transform(X_trn['Speech'].values)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(train_data_features)


X_tst, y_tst = df_t[['Speech']],df_t[['sentimentAnnotation']]

# transformation of test data
test_data_features = vectorizer.transform(X_tst['Speech'].values)
X_test_tfidf = tfidf_transformer.transform(test_data_features)

from sklearn import svm
model_tf = svm.SVC(kernel='linear', C=1, gamma=1).fit(X_train_tfidf,y_trn['sentimentAnnotation'].values)

# generate predictions
predicted_tf = model_tf.predict(X_test_tfidf)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_tst['sentimentAnnotation'].values, predicted_tf))

#create df to show results
disp = X_tst.join(y_tst).reset_index(drop=True).join(pd.DataFrame(predicted_tf,columns=['Prediction']))
disp = disp.join(pd.DataFrame(disp['sentimentAnnotation']==disp['Prediction'],columns=['Right/Wrong']))

scores = model_tf.score(X_test_tfidf,y_tst['sentimentAnnotation'].values)
print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
print("Mean sentiment: {!r}.".format('Positive' if disp['sentimentAnnotation'].mean()>=0 else 'Negative'))
                            
print("Predicted mean sentiment: {!r}.".format('Positive' if disp['Prediction'].mean()>=0 else 'Negative'))