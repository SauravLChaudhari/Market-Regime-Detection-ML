# classification.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Create labels for regime shifts (for demonstration purposes)
data['Regime'] = data['Cluster']

# Split data into training and testing sets
X = data[['Return', 'Volatility']]
y = data['Regime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Implement SVM classification
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Evaluate the model
print('SVM Classification Report:')
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
