import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from paths import model_path,data_path

clf = joblib.load(model_path)
df = pd.DataFrame()
data = pd.DataFrame()
df = pd.read_csv(data_path,skiprows=2)
data = data.append(df)
data = data.drop(["File"],axis=1)
data = data.drop(["Neutral (v3)"],axis=1)
data = data.fillna(0)

#remove cols with no var
#data = data.loc[:,data.var() > 0.0001]

#scaling
scaler = MinMaxScaler()
scaler.fit(data)
data[list(data)] = scaler.transform(data[list(data)])
x_test = data.iloc[ : , 0 : 75]
y_test = data.iloc[ : , 75: ]
predictions = clf.predict(x_test)



#score = clf.score(x_test,y_test)
emotions = ['Anger','Contempt','Disgust','Fear','Joy','Sad','Surprise']

print predictions.mean(axis = 0) #mean of each emotion across various frames in a single video
index = np.argmax(predictions.mean(axis=0))
print "Overall emotion : " + emotions[index]
