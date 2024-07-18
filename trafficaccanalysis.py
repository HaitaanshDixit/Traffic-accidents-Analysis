import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

file = r'.vscode\Projects\traffic_accidents_india-2016-18.csv'
df = pd.read_csv(file)                     # DataFrame created from the file

print(df, '\n\n')

#Preprocessing

print(df.isnull().sum(), '\n')
# Columns "Pedestrian", "Bicycles" have 8 null values each

# Using SimpleImputer
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
df['Pedestrian'] = imp.fit_transform(df[['Pedestrian']])
df['Bicycles'] = imp.fit_transform(df[['Bicycles']])
print(df.isnull().sum(), '\n')

#Data Encoding
from sklearn.preprocessing import OneHotEncoder

print(df.columns, '\n')

categorical_features = ['City', 'Accident type']
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[categorical_features])


feature_names = encoder.get_feature_names_out(categorical_features)
print(feature_names, '\n')

# Made DataFrame for new encoded data
encoded_df = pd.DataFrame(encoded, columns=feature_names)

# Concatenated the encoded data with original data
df_encoded = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)


#Data Scaling
from sklearn.preprocessing import StandardScaler

num_cols = df.select_dtypes(include=['int32', 'int64']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Made DataFrame for new scaled data
scaled_df = pd.DataFrame(df[num_cols], columns=num_cols)

# Concatenate encoded and scaled features
df_encodednscaled = pd.concat([scaled_df, encoded_df], axis=1)

X = df_encodednscaled.iloc[:, 0:-1]
y = df_encodednscaled.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Visualize Before Classification

# Histograms for Distribution of Features
df.hist(figsize=(8, 8))
plt.suptitle("Feature Distributions", fontsize=10)
plt.show()

# Scatter Plots for Relationships
sns.pairplot(df, hue='City')    # City is our target variable
plt.suptitle("Pair Plot of Features", fontsize=4)
plt.show()

# Correlation Matrix
plt.figure(figsize=(8,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


from sklearn.ensemble import GradientBoostingClassifier         

model = GradientBoostingClassifier(random_state=42)

# HyperParameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.05],
    'max_depth': [3, 4, 5, 6, 7]
             }
# Perform GridSearch
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# cv is cross validation, in param grid, 5 values each are given so cv is 5 and njobs is for cpu usage, 1 uses 1 cpu, -1 uses all cpu, 2 and so on use that many cpu cores.

# Perform the grid search
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print('\n', f"Best parameters: {best_params}")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
print(y_pred)


# Visualize After Classification

from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print('\n\n',"Classification Report:\n", classification_report(y_test, y_pred), '\n')

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('\n', f"Accuracy Score = {acc}", '\n')




# Feature Importance
feature_names = X_train.columns.tolist()
importances = best_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(8, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()























