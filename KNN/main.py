import KNN
from KNN import KNN 
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    # collecting data

    data = pd.read_csv("car.data")
    #print(data.head())
    
    le = preprocessing.LabelEncoder()
    
    buying = le.fit_transform(list(data['buying']))
    maint  = le.fit_transform(list(data['maint']))
    door   = le.fit_transform(list(data['door']))
    persons= le.fit_transform(list(data['persons']))
    lug_boot= le.fit_transform(list(data['lug_boot']))
    safety  = le.fit_transform(list(data['safety']))
    cls = le.fit_transform(list(data['class']))
    
    print(type(buying[0])) 

    raw_x = list(zip(buying, maint, door, persons, lug_boot, safety)) 
    raw_y = list(cls)

    raw_x = np.array(raw_x, dtype = np.float64)
    raw_y = np.array(raw_y)

    x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, random_state= 1, test_size = 0.2)

    # apply KNN
    
    accuracy = {'method' : 'mahattan_distance'}
    for i in range(1,101):
        model = KNN(i)

        model.fit(x_train, y_train, accuracy['method'])
    
    
        predictions = model.predict(x_test)
     
        
        #print("Model evaluation ")
        acc, con_matrix = model.evaluate(y_test, predictions)
        #print("Accuracy: {}".format(acc))
        #print("Confusion matrix")
        #print(con_matrix)
        accuracy.update({i : acc})
import json 
with open("norm1.json", 'w') as file:
    json.dump(accuracy, file)

