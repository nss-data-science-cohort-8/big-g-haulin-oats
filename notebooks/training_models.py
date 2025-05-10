# A py file to run the model and log the results using MLflow.

# Import statements
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def split_X_y(df):
    """
    Splits the DataFrame into features (X) and target (y).
    Assumes the first column is the target variable.
    """
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    return X, y

def evaluate_model(y_test, y_pred):
    # Calculate additional metrics on the test set
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return precision, recall, f1, accuracy

def ensure_target_cols_removed(df):
    target_cols = ['EventTimeStamp', 'EquipmentID', 'FullDerate']
    # Remove target_cols from df
    df = df.drop(columns=[col for col in target_cols if col in df.columns])
    return df

def start_mlflow_run(train_data_file_path, test_data_file_path, predictions_filepath, model, dropna=True):
    """
    Starts an MLflow run and logs the parameters and artifacts.
    """
    mlflow.set_experiment("big-g-haulin-oats")
    mlflow.autolog()

    with mlflow.start_run():
        mlflow.log_param("train_data_file_path", train_data_file_path)
        mlflow.log_param("test_data_file_path", test_data_file_path)

        mlflow.log_artifact(train_data_file_path, artifact_path="data")
        mlflow.log_artifact(test_data_file_path, artifact_path="data")

        train_df = pd.read_csv(train_data_file_path)
        test_df = pd.read_csv(test_data_file_path)

        train_df = ensure_target_cols_removed(train_df)
        test_df = ensure_target_cols_removed(test_df)

        if dropna:
            train_df = train_df.dropna()
            test_df = test_df.dropna()
        
        X_train, y_train = split_X_y(train_df)
        X_test, y_test = split_X_y(test_df)

        # Fit the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics on the test set
        precision, recall, f1, accuracy = evaluate_model(y_test, y_pred)
        print(f"Test precision: {precision}")
        print(f"Test recall: {recall}")
        print(f"Test F1 score: {f1}")
        print(f"Test accuracy: {accuracy}")
        # Log custom metrics for the test set
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_accuracy", accuracy)

        print("Model training and evaluation completed successfully.")

        # Save predictions to a CSV file
        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        predictions_df.to_csv(predictions_filepath, index=False)
        mlflow.log_artifact(predictions_filepath, artifact_path="data")

    return predictions_df

def count_predictions(predictions_df, test_df, num_prev_hours=24):

    df = pd.merge(predictions_df, test_df.dropna().reset_index(), left_index=True, right_index=True)
    df['EventTimeStamp'] = pd.to_datetime(df['EventTimeStamp'])
    df = df.sort_values('EventTimeStamp').reset_index(drop=True)

    # If Predicted == 1 and FullDerate occurs at least 2 hours later and less than 24 hours later, True Positive count ++
    df['NextDerateTime'] = np.where(df['FullDerate']==1, df['EventTimeStamp'], pd.NaT)
    df['NextDerateTime'] = pd.to_datetime(df['NextDerateTime'])
    df['NextDerateTime'] = df['NextDerateTime'].bfill()

    df['HoursUntilNextDerate'] = (df['NextDerateTime'] - df['EventTimeStamp']).dt.total_seconds() / 3600.0
    df['TruePositive'] = np.where(((df['HoursUntilNextDerate'] > 2) & (df['HoursUntilNextDerate'] <= num_prev_hours) & (df['Actual']==1) & (df['Predicted']==1)), 1, 0)
    df['FalsePositive'] = np.where(((df['Actual']==0) & (df['Predicted']==1)), 1, 0)

    df['CountedTruePositive'] = 0
    df['CountedFalsePositive'] = 0

    full_derates = df[df['FullDerate'] == 1]['EventTimeStamp'].tolist()

    for derate_time in full_derates:
        window_start = derate_time - pd.Timedelta(hours=num_prev_hours)
        tp_mask = (
            (df['EventTimeStamp'] >= window_start) &
            (df['EventTimeStamp'] < derate_time) &
            (df['TruePositive'] == 1)
        )
        if tp_mask.any():
            first_tp_index = df[tp_mask].index[0]
            df.loc[first_tp_index, 'CountedTruePositive'] = 1

    last_counted_fp_time = pd.Timestamp('1900-01-01')

    for i, row in df.iterrows():
        if row['FalsePositive'] == 1:
            if row['EventTimeStamp'] - last_counted_fp_time >= pd.Timedelta(hours=num_prev_hours):
                df.at[i, 'CountedFalsePositive'] = 1
                last_counted_fp_time = row['EventTimeStamp']

    tp_count = df['CountedTruePositive'].sum()
    fp_count = df['CountedFalsePositive'].sum()

    print(f'True Positives: {tp_count}')
    print(f'False Positives: {fp_count}')

    return tp_count, fp_count

def revenue_calc(tp, fp):
    cost_per_derate_correct = 4000
    cost_per_false_positive = 500

    total_revenue = (tp * cost_per_derate_correct) - (fp * cost_per_false_positive)
    print(f'Total revenue: {total_revenue}')

    return total_revenue