# classification.py
from sklearn.ensemble import RandomForestClassifier

# Implement Random Forest classification
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate the model
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
