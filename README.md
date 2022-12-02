
# Import libraries and dependencies
import numpy as np
import pandas as pd

# ------------------------------ Data Set Loading ------------------------------

# Read data set
df = pd.read_csv('Desktop/file.csv')

# Visualize data set
df.head()

# ------------------------------- Data Cleaning --------------------------------

# Remove null values
df.dropna(inplace = True)

# Specify the features columns
X = df.drop(columns = [df.columns[-1]])

# Specify the target column
y = df.iloc[:,-1]

# Transform non-numerical columns into binary-type columns
X = pd.get_dummies(X)

# ----------------------------- Data Preprocessing -----------------------------

# Import train_test_split class
from sklearn.model_selection import train_test_split

# Divide data set into traning and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

# Import data scaling technique class
from sklearn.preprocessing import StandardScaler

# Instantiate data scaler
scaler = StandardScaler()

# Fit the Scaler with the training data
X_scaler = scaler.fit(X_train)

# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# ------------------------------ Data Resampling ------------------------------

# Import data resampling class
from imblearn.under_sampling import RandomUnderSampler

# Instantiate data resampler technique
rus = RandomUnderSampler()

# Resample training sets
X_resampled, y_resampled = rus.fit_resample(X_train_scaled, y_train)

# ------------------------------- Model Building -------------------------------

# Import machine learning model class
from sklearn.ensemble import RandomForestClassifier

# Instatiate machine learning model
rfc = RandomForestClassifier()

# Fit the machine learning model with the training data
rfc.fit(X_resampled, y_resampled)

# Make predictions using the testing data
y_pred = rfc.predict(X_test_scaled)

# ------------------------------ Model Evaluation ------------------------------

# Calculate balanced accuracy scrore
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_test, y_pred)

# Display the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred)

# Display the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
