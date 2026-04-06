#========================== IMPORT PACKAGES ============================

import pandas as pd
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from sklearn import metrics
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import pickle

#=========================== DATA SELECTION ============================

dataframe=pd.read_csv("Dataset.csv")
print("----------------------------------------------------")
print("                     Data Selection                 ")
print("----------------------------------------------------")
print()
print(dataframe.head(15))


#========================== PRE PROCESSING ================================

#====== CHECKING MISSING VALUES ========

print("----------------------------------------------------")
print("             Handling Missing values                ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())

res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



# --------- Encode string/object columns ------------

print("----------------------------------------------------")
print("         Encoding Categorical/String Columns        ")
print("----------------------------------------------------")
print()

# Find object type columns
object_cols = dataframe.select_dtypes(include=['object']).columns.tolist()

if len(object_cols) == 0:
    print("No categorical/string columns detected for encoding.")
else:
    print("Columns before encoding (showing first 5 rows):")
    print(dataframe[object_cols].head())

    # Save each categorical column before encoding separately to pickle
    for col in object_cols:
        col_data = dataframe[col]
        filename = f"{col}_before_encoding.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(col_data, f)
        print(f"Saved '{col}' column to '{filename}'")

    le = preprocessing.LabelEncoder()

    for col in object_cols:
        dataframe[col] = le.fit_transform(dataframe[col].astype(str))

    print()
    print("Columns after encoding (showing first 5 rows):")
    print(dataframe[object_cols].head())


#========================== DATA SPLITTING ===========================


X=dataframe.drop(['Diagnosis'],axis=1)
y=dataframe['Diagnosis']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("--------------------------------------------")
print("             Data Splitting                 ")
print("--------------------------------------------")
print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])


#====================== CLASSIFICATION ===============================

#-------------------------------------------------
# Random Forest Classifier
#-------------------------------------------------


from sklearn.ensemble import RandomForestClassifier

#=== initialize the model ===
rf = RandomForestClassifier()

#=== fitting the model ===
rf.fit(X_train, y_train)

#=== predict the model ===
y_pred_rf = rf.predict(X_test)


acc_rf=metrics.accuracy_score(y_test,y_pred_rf)*100


cm1 = metrics.confusion_matrix(y_test,y_pred_rf)

print("--------------------------------------------")
print("      Performance Analysis (Random forest)  ")
print("--------------------------------------------")
print()
print("1. Accuracy :",acc_rf )
print()
print("2.Confusion matrix :\n",cm1)
print()
print("3. Classification Report :" )
print()
print(metrics.classification_report(y_test,y_pred_rf))
print()
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm1, annot=True)
plt.title("RF Classifier")
plt.show()


# === 


import pickle

with open('model.pickle', 'wb') as f:
  pickle.dump(rf, f)





#-------------------------------------------------
# MLP Classifier
#-------------------------------------------------


from sklearn.neighbors import KNeighborsClassifier

#=== initialize the model ===
knn = KNeighborsClassifier()

#=== fitting the model ===
knn.fit(X_train, y_train)

#=== predict the model ===
y_pred_knn= knn.predict(X_test)


acc_knn = metrics.accuracy_score(y_test,y_pred_knn)*100


cm1 = metrics.confusion_matrix(y_test,y_pred_knn)

print("--------------------------------------------")
print("     Performance Analysis (KNN Classifier)  ")
print("--------------------------------------------")
print()
print("1. Accuracy :",acc_knn )
print()
print("2.Confusion matrix :\n",cm1)
print()
print("3. Classification Report :" )
print()
print(metrics.classification_report(y_test,y_pred_knn))
print()

sns.heatmap(cm1, annot=True)
plt.title("KNN Classifier")
plt.show()





import seaborn as sns
import matplotlib.pyplot as plt

#pie graph
plt.figure(figsize = (6,6))
counts = y.value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Patients: {}'.format(dataframe.shape[0]))
plt.title('Diagnosis Analysis', fontsize = 14);
plt.show()

