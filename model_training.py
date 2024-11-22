import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn. tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

patient = pd.read_csv(r"Final_PCOS_Dataset.csv")
x = patient.iloc[:, :-1]
y = patient.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

classifiers = {
    'Logistic Regression' : LogisticRegression(max_iter = 7000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(max_depth=5),
    'Support Vector Machine': SVC(),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbours': KNeighborsClassifier()
}

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    X_test = np.array(X_test)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix for",name,"\n", cm)
    accuracy = accuracy_score(y_test, y_pred)
    results[name]  = accuracy
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))
    print('--------------------------------------------------------------------')

#Finding the best classifier
best_classifier = max(results, key = results.get)
print(f'Best Classifier: {best_classifier} \nAccuracy = {results[best_classifier]:.4f}')

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color = 'magenta', lw = 2, label = f'ROC Curve (AUC Area = {roc_auc})')
plt.plot([0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.legend(loc = "lower right")
plt.grid(True)
plt.show()

#Optimising Logistic Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C' : [0.001, 0.01, 0.1, 1, 10],
    'penalty' : ['l1', 'l2'],
    'solver' : ['liblinear','saga']
}

logistic_regression = LogisticRegression(max_iter = 7000)

grid_search = GridSearchCV(estimator = logistic_regression, param_grid = param_grid, cv = 5, scoring = 'accuracy')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

y_pred = grid_search.predict(X_test)

print("Best Parameters: ", best_params)
print("Best score: ", best_score)

best_model = grid_search.best_estimator_
print("Best Model: ", best_model)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')
f1 = f1_score(y_test, y_pred, average = 'weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')
print("Confusion Matrix\n", conf_matrix)

"""#### Optimisng Decision Tree"""

param_grid = {
    'max_depth':[None, 3, 4, 5, 6, 7],
    'min_samples_split' : [2, 5, 7],
    'min_samples_leaf' : [1, 2, 4],
    'criterion' : ['gini', 'entropy']

}

dt = DecisionTreeClassifier(random_state = 42)
grid_search = GridSearchCV(dt, param_grid, scoring = 'accuracy', cv = 5, verbose = 1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print("Best Params: ", best_params)
print("Best Score: ", best_score)
print("Best Model: ", best_model)

import joblib
joblib.dump(best_model, 'decision_tree_model.pkl')

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')
f1 = f1_score(y_test, y_pred, average = 'weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')
print("Confusion Matrix\n", conf_matrix)

from sklearn.ensemble import AdaBoostClassifier
base_dt = DecisionTreeClassifier(max_depth = 5, random_state = 42)
adaboost_clf = AdaBoostClassifier(estimator = base_dt, n_estimators = 50, random_state = 42)

adaboost_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'weighted')
recall = recall_score(y_test, y_pred, average = 'weighted')
f1 = f1_score(y_test, y_pred, average = 'weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')
print("Confusion Matrix\n", conf_matrix)

numerical_selected_indices = f_selector.get_support(indices=True)
categorical_selected_indices = chi2_selector.get_support(indices=True)

# Get selected feature names for both numerical and categorical
selected_numerical_features = [numerical_features[i] for i in numerical_selected_indices]
selected_categorical_features = [categorical_features[i] for i in categorical_selected_indices]

# Combine them into one list
selected_features = selected_numerical_features + selected_categorical_features

print("Selected Features:")
print(selected_features)

feature_importances = adaboost_clf.feature_importances_

# Create a DataFrame to display feature names and their importances together
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
})

feature_importance_df

plt.figure(figsize=(10, 10))
plt.barh(selected_features, feature_importances, color = 'navy')
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.title("Feature Importance Plot - Adaboost")
plt.grid(True)
plt.show()

from sklearn.tree import plot_tree
plt.figure(figsize =  (50,50))
plot_tree(best_model, feature_names = selected_features, filled = True, class_names = ["No PCOS", "PCOS"])
plt.title("Decision Tree Visualization")
plt.show()
