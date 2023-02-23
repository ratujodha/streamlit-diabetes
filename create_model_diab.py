import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle


#2 load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.head()
tes = diabetes_dataset['Outcome'].value_counts()
#print(tes)

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
#print(X)
#print(Y)

#3 standarisasi data
scaler = StandardScaler()
scaler.fit(X)

standarized_data = scaler.transform(X)
#print(standarized_data)

X = standarized_data
Y = diabetes_dataset['Outcome']

#print(X)
#print(Y)

#4 memisahkan data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)


#5 membuat data latih SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

#6 akurasi
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#print("Data Training data : ",training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
#print("Data Testing data : ",test_data_accuracy)

 #7 membuat model prediksi

input_data = (6, 148, 72, 35, 0, 33.6 ,0.627, 50)
input_data_as_numpy_array = np.array(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshape)
#print(std_data)

prediction = classifier.predict(std_data)
#print(prediction)

if(prediction==0):
    print('Tidak')
else:
    print('Ya')

filename = 'diabetes_model.sav'
pickle.dump(classifier, open(filename,'wb'))
   