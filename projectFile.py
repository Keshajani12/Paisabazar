import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('dataset-2.csv')

#print(df)
#print(df.info)
#print(df.shape)
#print(df.describe())

#duplicates = df[df.duplicated()]
#print(duplicates)

#Count Total Missing Values in the DataFrame
#missingValue= df.isnull().sum().sum()
#print(missingValue)

#print(df.head())
#print(df.tail())

#print(max(df.Age))
#print(min(df.Age))

minor=(df[df.Age<18])
#print(minor)

adult=(df[df.Age>=18])
#print(adult)

midAge=(df[df.Age>=40])
#print(midAge)

seniorcitizen=(df[df.Age>60])
#print(seniorcitizen)

occ=(df[df.Occupation=='Engineer'])
#print(occ)

occupation=df['Occupation'].value_counts()
#print(occupation)

#sns.lineplot(data=midAge,x='Occupation',y='Annual_Income', errorbar=None)
plt.show()

#sns.barplot(data=minor,x='Num_Bank_Accounts',y='Num_of_Loan')
plt.show()

#sns.lineplot(data=df, x="Month", y="Monthly_Balance", hue="Credit_Score")
plt.show()

#pd.crosstab(df['Occupation'], df['Credit_Score']).plot(kind="bar", figsize=(12,6))
plt.show()

#sns.pairplot(df[['Age','Annual_Income','Monthly_Balance','Credit_Score']], hue="Credit_Score")
plt.show()

#sns.violinplot(data=df, x="Occupation", y="Monthly_Balance")
plt.show()

le = LabelEncoder()
df['Credit_Score_encoded'] = le.fit_transform(df['Credit_Score'])
y = (df['Credit_Score_encoded'])

X = df[['Age','Monthly_Balance','Num_Bank_Accounts']]
#print(X_train.head())
#print(X_test.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
#print(y_test.head())
#print(y_pred_lr[:5])
#print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
#print(y_test.head())
#print(y_pred_dt[:5])
#print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#print(y_test.head())
#print(y_pred_rf[:5])
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

#(using Random Forest predictions here)
submission = pd.DataFrame({
    "ID": X_test.index,
    "Predicted_Credit_Score": le.inverse_transform(y_pred_rf)
})
submission.to_csv("ML_Submission.csv", index=False)
print("Submission file saved as ML_Submission.csv")