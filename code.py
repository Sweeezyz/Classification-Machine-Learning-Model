import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '/content/PV_Data (1).xlsx'
df = pd.read_excel('/content/PV_Data (1).xlsx')

# Numerical summary
numerical_summary = df.describe()
print("Numerical Summary:")
print(numerical_summary)

# Pairplot
print(sampled_data.columns)
sns.pairplot(sampled_data, hue='Fault Label', diag_kind='kde')
plt.show()

# Heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot
sns.boxplot(x='Fault Label', y='Irradiance', data=df)
plt.title('Boxplot of Irradiation by Fault Type')
plt.show()

# Scatterplot
sns.scatterplot(x='Voltage - String 1', y='Voltage - String 2', hue='Fault Label', data=df)
plt.title('Scatterplot of Voltage Strings with Fault Types')
plt.show()

# Classifiers
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
            df.drop('Fault Label', axis=1), df['Fault Label'], test_size=0.3, random_state=random_state
        )

        # Normalize feature variables
        X_train_normalized = normalization.fit_transform(X_train)
        X_test_normalized = normalization.transform(X_test)

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
        print(f"\nAverage Accuracy for Data Split {random_state}: {avg_acc:.4f}")
