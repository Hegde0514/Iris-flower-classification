 1. Import Required Libraries
 import numpy as np
 import pandas as pd
 import seaborn as sns
 import matplotlib.pyplot as plt
 from sklearn.datasets import load_iris
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 2. Load the Dataset
 iris = load_iris()
 X = pd.DataFrame(iris.data, columns=iris.feature_names)
 y = pd.Series(iris.target, name="species")
 species_map = dict(zip(range(3), iris.target_names))
 y = y.map(species_map)
 3. Exploratory Data Analysis (EDA)
 # Combine features and target for visualization
 df = pd.concat([X, y], axis=1)
 # Pairplot
 sns.pairplot(df, hue="species")
 plt.show()
 # Correlation heatmap
 plt.figure(figsize=(8, 5))
 sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
 plt.show()
 4. Split the Dataset
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 scaler = StandardScaler()
 X_train_scaled = scaler.fit_transform(X_train)
 X_test_scaled = scaler.transform(X_test)
 5. Train Logistic Regression Model
 model = LogisticRegression()
 model.fit(X_train_scaled, y_train)
 y_pred = model.predict(X_test_scaled)
 6. Evaluate the Model
 print(confusion_matrix(y_test, y_pred))
 print(classification_report(y_test, y_pred))
 print("Accuracy Score:", accuracy_score(y_test, y_pred)
