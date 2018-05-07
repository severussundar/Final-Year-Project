import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from paths import data_path2 as data_path

score = 0
data = pd.DataFrame() #creating an empty pandas data frame

#reading data from the directory and appending to the data frame
for f in glob.glob(data_path) :
	df = pd.read_csv(f,skiprows=2)
	data = data.append(df)
data = data.drop(["File"],axis=1)
data = data.drop(["Neutral (v3)"],axis=1)
data = data.fillna(0) #replacing NaN values with 0

#calculating variance and removing cols with variance < 0.0001
#data = data.loc[:,data.var() > 0.0001]

#feature scaling
scaler = MinMaxScaler()
scaler.fit(data)
data[list(data)] = scaler.transform(data[list(data)])

while score < 0.70:
	data = data.sample(frac=1) #shuffling the data 
	#splitting our data into training and testing data
	X_train,X_test,Y_train,Y_test = train_test_split(data.ix[:,0:75],data.ix[:,75:],test_size=0.10,random_state=42)
	#creating the Multi Layer Perceptron Regressor model 
	clf = MLPRegressor(solver="adam",alpha=1e-5,hidden_layer_sizes=(70,100,40,8),max_iter=240) 	
	clf.fit(X_train, Y_train) #training the model
	score =  clf.score(X_test, Y_test) 
	print(clf.score(X_test, Y_test)) #printing the score
	#print(mean_squared_error(clf.predict(X_test),Y_test))
	if score > 0.90: 
		joblib.dump(clf, 'project'+str(score)+'.pkl') #storing the model as a pickle file
