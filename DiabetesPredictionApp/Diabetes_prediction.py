# Importing libraries

import pickle
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')
df.head()
print(df.describe().head(5))
df.isnull().head(10)
df.isnull().sum()
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)


df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

color_wheel = {1: "#0392cf", 2: "#7bc043"}
colors = df["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(df.Outcome.value_counts())

cols = df.columns
cols = cols.drop('Outcome')
plt.subplot(121)
sns.distplot(df['Insulin'])
plt.subplot(122)
df['Insulin'].plot.box(figsize=(16, 5))

sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(df_copy.drop(['Outcome'], axis=1)),

                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                          'DiabetesPedigreeFunction', 'Age'])
X.head()
y = df_copy.Outcome
y.head()

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

model_final_random = RandomForestClassifier(n_estimators=200)
model_final_random.fit(X_train, y_train)

predictions = model_final_random.predict(X_test)

model_final_random.feature_importances_

saved_model = pickle.dumps(model_final_random)
model_final_random_from_pickle = pickle.loads(saved_model)
model_final_random_from_pickle.predict(X_test)

ipt_values1 = [10, 101, 76, 48, 180, 32.9, 0.171, 63]
ipt_values_np_array1 = np.asarray(ipt_values1)
ipt_values_reshape1 = ipt_values_np_array1.reshape(1,-1)
predict1 = model_final_random.predict(ipt_values_reshape1)
print(predict1)
if(predict1[0]==0):
    print("You are not Diabetic")
else :
    print("You are Diabetic")

with open('model_rf.pkl','wb') as file:
    pickle.dump(model_final_random,file)