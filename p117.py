import pandas as pd

df = pd.read_csv("BankNote_Authentication.csv")

print(df.head())

#Predicting using Logistic Regression for Binary classification 
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix


LR = LogisticRegression()
LR.fit(X_train,y_train) #fitting the model 

y_prediction = LR.predict(X_test) #prediction

#creating the lists of the data
predicted_values = []
for i in y_prediction:
  if i == 0:
    predicted_values.append("Authorized")
  else:
    predicted_values.append("Forged")

actual_values = []
for i in y_test:
  if i == 0:
    actual_values.append("Authorized")
  else:
    actual_values.append("Forged")
    
labels = ["Forged", "Authorized"]

cm = confusion_matrix(actual_values, predicted_values, labels)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)