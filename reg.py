

import pandas as pd 

data= pd.read_csv('Social_Network_Ads.csv')
def readData():
    return data
x = data.iloc[:,2:4].values
y = data.iloc[:,4].values 


#separate train and test data

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25, random_state = 0)


# variables scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
xTrain = sc_X.fit_transform(xTrain)
xTest = sc_X.transform(xTest)


#Create a Model for the regression such as 
from sklearn.naive_bayes import GaussianNB
naiveBayesModel = GaussianNB()
naiveBayesModel.fit(xTrain,yTrain)



#Data prediction
yPred = naiveBayesModel.predict(xTest)


def newPrediction(data):
    data = sc_X.transform(data)
    yPred = naiveBayesModel.predict(data)
    return yPred



