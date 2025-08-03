import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("PV_Data.csv")
numerical_summaries = df.describe()
print(df.head())
numerical_summaries

plt.figure(figsize=(10, 8))
sns.boxplot(x = 'Fault Label', y = 'Irradiance', data=df)
plt.title('Irradiance by Fault Label')
plt.show()


corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()


plt.figure(figsize=(12, 8))
df[['Current - String 1', 'Current - String 2']].hist(bins=20)
plt.suptitle('Currents in String 1 and String 2', y=1)
plt.show()


plt.figure(figsize=(15, 10))
sns.pairplot(df, hue='Fault Label', palette='viridis')
plt.suptitle('Numerical Variables Colored by Fault Label', y=1)
plt.show()


sns.scatterplot(x='Voltage - String 1', y='Voltage - String 2', hue='Fault Label', data=df)
plt.title('Scatterplot of Voltage Strings with Fault Types')
plt.show()

classifiers = {
'Decision Tree': DecisionTreeClassifier(random_state=42),
'k-NN': KNeighborsClassifier(n_neighbors=5),
'Naive Bayes': GaussianNB(),
'SVM Polynomial': SVC(kernel='poly', random_state=42),
'SVM RBF': SVC(kernel='rbf', random_state=42),
'Neural Network': MLPClassifier(random_state=42)
}

# Normalizations
normalizations = {
'Z-Score': StandardScaler(),
'Min-Max': MinMaxScaler()
}

# Loop through normalization techniques
for norm_name, normalization in normalizations.items():
print(f"\nNormalization Technique: {norm_name}")

# Loop through data splits
for random_state in [1, 20, 40]:
print(f"\nRandom State: {random_state}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
df.drop('Fault Label', axis=1), df['Fault Label'],
test_size=0.3, random_state=random_state)

# Normalize feature variables
X_train_normalized = normalization.fit_transform(X_train)
X_test_normalized = normalization.transform(X_test)\

# Loop through classifiers
results = {'Classifier': [], 'Accuracy': []}
for clf_name, classifier in classifiers.items():

# Train and predict
classifier.fit(X_train_normalized, y_train)
y_pred = classifier.predict(X_test_normalized)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

# Store results
results['Classifier'].append(clf_name)
results['Accuracy'].append(acc)

print(f"{clf_name} Accuracy: {acc:.4f}")

# Display average accuracy for data split
avg_acc = sum(results['Accuracy']) / len(results['Accuracy'])
print(f"\nAverage Accuracy for Data Split {random_state}:
{avg_acc:.4f}")
