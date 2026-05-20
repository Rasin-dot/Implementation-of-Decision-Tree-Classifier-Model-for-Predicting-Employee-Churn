# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
   
2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score 

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RASINDHAN R
RegisterNumber: 212224230222

```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:


<img width="1626" height="274" alt="image" src="https://github.com/user-attachments/assets/113ea26a-5ae6-46e3-ac78-05b145cf29eb" />


<img width="1760" height="404" alt="image" src="https://github.com/user-attachments/assets/91513074-caf6-462d-a0aa-ca1cae79a9bb" />


<img width="1739" height="509" alt="image" src="https://github.com/user-attachments/assets/e5cd92c7-ca97-4067-ab63-45bab00e194a" />


<img width="1757" height="239" alt="image" src="https://github.com/user-attachments/assets/a52f1c4f-38c4-40c6-bcb9-cbc14335f1e8" />


<img width="1622" height="272" alt="image" src="https://github.com/user-attachments/assets/b73e286b-db4d-4125-9144-91579cd4d6af" />


<img width="1436" height="267" alt="image" src="https://github.com/user-attachments/assets/b7d187cd-e35e-4c8b-b233-dfc81684c9ef" />


<img width="1705" height="55" alt="image" src="https://github.com/user-attachments/assets/583faedc-dad5-410d-be23-e2e2b6abf432" />


<img width="1774" height="110" alt="image" src="https://github.com/user-attachments/assets/d8c39f2b-3a7a-4bc0-8656-513c354bf058" />


<img width="1750" height="622" alt="image" src="https://github.com/user-attachments/assets/5d1506c9-3d13-48be-9793-622e739448fa" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
