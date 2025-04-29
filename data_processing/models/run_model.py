# Import statements
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
#from xgboost import XGBRegressor
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# File name for the dataset
file_name = 'simple_test_04_29'

mlflow.set_experiment("big-g-haulin-oats")
mlflow.autolog()

with mlflow.start_run():
    train_data_file_path = f"../preprocessed_data/{file_name}_train.csv"
    test_data_file_path = f"../preprocessed_data/{file_name}_test.csv"

    mlflow.log_param("train_data_file_path", train_data_file_path)
    mlflow.log_param("test_data_file_path", test_data_file_path)

    mlflow.log_artifact(train_data_file_path, artifact_path="data")
    mlflow.log_artifact(test_data_file_path, artifact_path="data")

    train_df = pd.read_csv(train_data_file_path)
    test_df = pd.read_csv(test_data_file_path)

    train_df = train_df.dropna()
    test_df = test_df.dropna()
    # Try imputing (optionally by ffill)

    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    #simple_model = SVC(class_weight='balanced', probability=True, random_state=42)
    #simple_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    simple_model = DecisionTreeClassifier(max_depth=8, random_state=42)
    simple_model.fit(X_train, y_train)
    y_pred = simple_model.predict(X_test)

     # Calculate additional metrics on the test set
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Log custom metrics for the test set
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_accuracy", accuracy)

    print(f"Test precision: {precision}")
    print(f"Test recall: {recall}")
    print(f"Test F1 score: {f1}")
    print(f"Test accuracy: {accuracy}")
    print("Model training and evaluation completed successfully.")