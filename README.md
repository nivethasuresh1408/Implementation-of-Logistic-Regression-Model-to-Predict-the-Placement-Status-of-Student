# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Input the dataset

Load the student placement dataset containing academic records, work experience, and placement status.

Step 2: Preprocess the data

Handle missing values (if any).

Encode categorical attributes (gender, ssc_b, hsc_b, hsc_s, degree_t, workex, specialisation).

Normalize/scale features if required.

Step 3: Define features and target

Features (X): [gender, ssc_p, ssc_b, hsc_p, hsc_b, hsc_s, degree_p, degree_t, workex, etest_p, specialisation, mba_p]

Target (y): status (Placed / Not Placed)

Step 4: Split dataset

Divide data into training set (80%) and testing set (20%).

Step 5: Train Logistic Regression model

Initialize logistic regression classifier.

Fit the model on training data (X_train, y_train).

Step 6: Predict output

Use the trained model to predict placement status on test data (X_test).

Step 7: Evaluate model performance

Compute Accuracy, Confusion Matrix, Precision, Recall, and F1-score.

Step 8: Predict for new input

Accept a new student’s attributes.

Preprocess the input in the same way as training data.

Use the trained model to predict whether the student will be placed.

Step 9: Output result

Display prediction: “Placed” or “Not Placed”.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NIVETHA S
RegisterNumber:  212223040137

```

```python
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

```
<img width="1038" height="167" alt="Screenshot 2025-09-03 211931" src="https://github.com/user-attachments/assets/e2d9066b-faa9-4825-8d98-7b87ccb7823b" />

```python

data1=data.copy()
data1=data1.drop(["sl_no","salary"], axis=1)
data1.head()

```
<img width="988" height="177" alt="Screenshot 2025-09-03 212138" src="https://github.com/user-attachments/assets/16eeb84d-3c8c-499f-8323-2570ae4d10dd" />

```python
data1.isnull().sum()

```
<img width="789" height="246" alt="Screenshot 2025-09-03 212211" src="https://github.com/user-attachments/assets/84c2252c-bc4c-4695-8ed0-16c5fd347baa" />

```python
data1.duplicated().sum()

```
<img width="292" height="27" alt="Screenshot 2025-09-03 212245" src="https://github.com/user-attachments/assets/7ff6f8cb-ff44-4cf8-8358-cafafdc8168d" />

```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1 ["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

```
<img width="1010" height="356" alt="Screenshot 2025-09-03 212623" src="https://github.com/user-attachments/assets/bd67998b-e957-4852-a847-8024ddb50679" />

```python
x=data1.iloc[:,:-1]
x

```
<img width="941" height="355" alt="Screenshot 2025-09-03 212713" src="https://github.com/user-attachments/assets/7da99d2d-283d-460a-948d-c192efb36eb6" />

```python
y=data1["status"]
y

```
<img width="560" height="215" alt="Screenshot 2025-09-03 212748" src="https://github.com/user-attachments/assets/fe78fc59-accb-4efd-ab4a-03bf24419627" />

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression (solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

```
<img width="755" height="49" alt="Screenshot 2025-09-03 213235" src="https://github.com/user-attachments/assets/635642c2-d83c-4301-903c-7ea81912ffcc" />

```python
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred) 
accuracy

```
<img width="220" height="14" alt="Screenshot 2025-09-03 213510" src="https://github.com/user-attachments/assets/7fc14cf1-9454-4e16-a843-880c67ffcc56" />

```python
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

```
<img width="364" height="45" alt="Screenshot 2025-09-03 213615" src="https://github.com/user-attachments/assets/7898f9e3-f4f3-4e0d-a73f-f89bf9c751da" />

```python
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

```
<img width="684" height="162" alt="Screenshot 2025-09-03 213817" src="https://github.com/user-attachments/assets/ac2959f0-958c-48c3-9e30-6620a9608822" />

```python
x_new=pd.DataFrame([[1,80,1,90,1,1,90,1,0,85,1,85]],columns=['gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'hsc_s','degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p'])
print('Name: NIVETHA S')
print('Reg No: 212223040137')
lr.predict(x_new)

```
## Output:
<img width="1084" height="179" alt="Screenshot 2025-09-04 095055" src="https://github.com/user-attachments/assets/50120271-575f-4e53-84f5-78cc8b691d3b" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
