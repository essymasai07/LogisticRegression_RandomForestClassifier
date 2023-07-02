import pandas as pd
df=pd.read_csv('fitness_class_2212.csv')
missing_values = df['weight'].isnull().sum()
print(missing_values)
overall_average = df['weight'].mean()
print(overall_average)
print(df['weight'].fillna(overall_average, inplace=True))
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(df['months_as_member'], bins=10, edgecolor='black')
ax.set_xlabel('Number of Months as Member')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Number of Months as Member')
plt.show()
grouped_data = df.groupby('attended')['months_as_member'].mean()
fig, ax = plt.subplots()
ax.bar(grouped_data.index, grouped_data.values)
ax.set_xlabel('Attendance')
ax.set_ylabel('Average Number of Months as Member')
ax.set_title('Relationship between Attendance and Number of Months as Member')
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df[['months_as_member']]
y = df['attended'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
logistic_accuracy = accuracy_score(y_test, y_pred)
print("Baseline Model Accuracy:", logistic_accuracy)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df[['months_as_member']]
y = df['attended']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
random_forest_accuracy = accuracy_score(y_test, y_pred)
print("Comparison Model Accuracy:", random_forest_accuracy)
print("Logistic Regression Model Accuracy:", logistic_accuracy)
print("Random Forest Classifier Model Accuracy:", random_forest_accuracy) 