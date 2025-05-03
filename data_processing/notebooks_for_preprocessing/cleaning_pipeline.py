# Import statements
import pandas as pd
import numpy as np

# Merge faults and diagnostics
def merge_faults_and_diagnostics(faults_filepath, diagnostics_filepath):
    faults = pd.read_csv(faults_filepath)
    diagnostics = pd.read_csv(diagnostics_filepath).pivot(index='FaultId', columns='Name', values='Value') 
    # Immediately pivot diagnostic data on FaultId to then be merged
    df = faults.merge(diagnostics, how='left', left_on='RecordID', right_on='FaultId')
    return df

def remove_service_locations(df, radius=.05):
    """
    Remove data points that are near service locations.
    """
    # Define service locations
    service_locations = {
        'Location1': (36.0666667, -86.4347222), 
        'Location2': (35.5883333, -86.4438888), 
        'Location3': (36.1950, -83.174722) 
    }

    # Calculate distance to each service location and filter out points within a certain radius
    for loc, coords in service_locations.items():
        df['DistanceTo' + loc] = np.sqrt((df['Latitude'] - coords[0])**2 + (df['Longitude'] - coords[1])**2)
        df = df[df['DistanceTo' + loc] > radius] 
        df = df.drop(columns='DistanceTo' + loc)

    return df

def remove_columns_with_nan_threshold(df, threshold):
    # Remove columns with more than the specified threshold of NaN values
    nan_threshold = int(threshold * len(df))
    cols_to_drop = df.columns[df.isna().sum() > nan_threshold]
    return df.drop(columns=cols_to_drop)

def remove_data(df, remove_service_locs=True, unnecessary_cols:list=None, nan_threshold=0.5):
    """
    Remove data points that are near service locations and drop unnecessary columns.
    """
    # Remove service locations if specified
    if remove_service_locs:
        # Remove data points that are near service locations
        df = remove_service_locations(df)
    # Remove unnecessary columns if specified
    if unnecessary_cols:
        df = df.drop(columns=unnecessary_cols)
    # Remove columns that are not needed for the analysis
    else:
        df = df.drop(columns=[
            'RecordID', 'ESS_Id', 'eventDescription', 'actionDescription', 'ecuSoftwareVersion', 'ecuSerialNumber', 'ecuModel', 'ecuMake', 'ecuSource', 'active', 'activeTransitionCount', 'faultValue', 'MCTNumber', 'Latitude', 'Longitude', 'LocationTimeStamp'
        ])
    def normalize_boolean_column(series):
        return (
            series
            .map(lambda x: True if str(x).strip().lower() == 'true'
                else False if str(x).strip().lower() == 'false'
                else pd.NA)
            .astype('boolean')
        )
    
    # Convert columns to appropriate dtypes
    for col in df.columns:
        if col in ['IgnStatus', 'CruiseControlActive', 'ParkingBrake']:
            df[col] = normalize_boolean_column(df[col])
        elif col in ['EventTimeStamp']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif col in ['EquipmentID', 'spn', 'fmi']:
            df[col] = df[col].astype('str')
        else:
            # Convert to numeric and replace inf values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Print the data and data types of the columns after conversion
    print("Data after removing service locations and unnecessary columns:")
    print(df.head())
    print("Data types after conversion:")
    print(df.dtypes)

    # Remove rows with NaN values in important columns
    important_cols = ['EventTimeStamp', 'spn', 'fmi', 'EquipmentID']
    df_dropped = df.dropna(subset=important_cols)
    print("Number of rows with NaNs in important columns removed from df:")
    print(df.shape[0] - df_dropped.shape[0])
    
    df = df_dropped
    # Remove columns with more than the specified threshold of NaN values
    df = remove_columns_with_nan_threshold(df, nan_threshold)
    # Note that this removes EngineTimeLtd

    # Sort columns by number of NaN values
    print("Columns sorted by number of NaN values:")
    print(df.isna().sum().sort_values(ascending=False))

    return df

def scale_and_ohe_data(train_df, test_df):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.preprocessing import LabelEncoder

    # Columns not to scale or OHE
    cols_to_exclude = ['EquipmentID', 'spn', 'FullDerate', 'NextDerateTime', 'HoursUntilNextDerate', 'DerateInNextTwoHours', 'DerateInNextTwentyFourHours']

    num_cols = [col for col in train_df.select_dtypes(include=np.number).columns if col not in cols_to_exclude]
    scaler = StandardScaler().fit(train_df[num_cols])
    train_df[num_cols] = scaler.transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    # Determine categorical columns to OHE or label encode
    cols_to_ohe = [col for col in train_df.select_dtypes(include='object').columns if col not in cols_to_exclude]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_df[cols_to_ohe])
    train_ohe_df = pd.DataFrame(ohe.transform(train_df[cols_to_ohe]), columns=ohe.get_feature_names_out(cols_to_ohe), index=train_df.index)
    test_ohe_df = pd.DataFrame(ohe.transform(test_df[cols_to_ohe]), columns=ohe.get_feature_names_out(cols_to_ohe), index=test_df.index)

    # Combine OHE results and return
    train_df = train_df.drop(columns=cols_to_ohe).join(train_ohe_df)
    test_df = test_df.drop(columns=cols_to_ohe).join(test_ohe_df)
    return train_df, test_df


def impute_missing_values(train_df, test_df, method='ffill'):
    """
    Impute missing values in the dataset.
    """
    def ffill_nans(train_df, test_df): # alternatively try interpolation, moving averages, KNeighbors
        """
        Forward fill values for each EquipmentID group, resetting after each FullDerate == 1.
        """
        print("Imputing missing values using forward fill")
        
        def fill_group(group):
            group = group.sort_values('EventTimeStamp')
            segment = group['FullDerate'].eq(1).cumsum() # Create segments based on FullDerate == 1
            return group.groupby(segment).ffill()
        
        train_df = train_df.groupby('EquipmentID', group_keys=False).apply(fill_group)
        test_df = test_df.groupby('EquipmentID', group_keys=False).apply(fill_group)

        return train_df, test_df
    
    def KNeighborsImputer(train_df, test_df, n_neighbors=5):
        from sklearn.impute import KNNImputer, SimpleImputer
        print("Imputing missing values using KNeighborsImputer")

        # Only numeric data
        cols_to_impute = train_df.select_dtypes(include=[np.number]).columns

        numeric_imputer = KNNImputer(n_neighbors=n_neighbors)

        # Fit the imputer on the training data
        numeric_imputer.fit(train_df[cols_to_impute])

        # Transform both the train and test data
        train_df[cols_to_impute] = numeric_imputer.transform(train_df[cols_to_impute])
        test_df[cols_to_impute] = numeric_imputer.transform(test_df[cols_to_impute])

        return train_df, test_df
    
    # Designate method for imputation
    if method == 'ffill':
        imputed_train_df, imputed_test_df = ffill_nans(train_df, test_df)
    elif method == 'KNeighbors':
        imputed_train_df, imputed_test_df = KNeighborsImputer(train_df, test_df)
    else:
        # If the method is not recognized, raise an error
        raise ValueError("Invalid imputation method. Choose 'ffill' or 'KNeighbors'.")

    return imputed_train_df, imputed_test_df

def create_target_cols(df):
    """
    Create target columns for the dataset.
    """
    # Create target column for Full Derate
    df['FullDerate'] = (df['spn'] == '5246').astype('int')
    # Order data by truck (EquipmentID) and time
    df = df.sort_values(['EquipmentID', 'EventTimeStamp'])

    # Create target column for Derate in next two hours
    df['NextDerateTime'] = df.where(df['FullDerate'] == 1)['EventTimeStamp']
    df['NextDerateTime'] = df.groupby('EquipmentID')['NextDerateTime'].transform('bfill')
    df['HoursUntilNextDerate'] = (df['NextDerateTime'] - df['EventTimeStamp']).dt.total_seconds() / 3600.0
    df['DerateInNextTwoHours'] = np.where(df['HoursUntilNextDerate'] <= 2, 1, 0)
    df['DerateInNextTwentyFourHours'] = np.where(df['HoursUntilNextDerate'] <= 24, 1, 0)

    return df

def remove_after_derate(df, time_limit=2):
    """
    Remove data points that are after a derate for a certain time limit.
    """
    df['PrevDerateTime'] = df.where(df['FullDerate'] == 1)['EventTimeStamp']
    df['PrevDerateTime'] = df.groupby('EquipmentID')['PrevDerateTime'].transform('ffill')

    # Calculate the time difference from the last derate
    df['TimeAfterDerate'] = (df['EventTimeStamp'] - df['PrevDerateTime']).dt.total_seconds() / 3600.0

    # Filter out rows where TimeAfterDerate is less than the time limit while keeping rows if no derate occurs for that truck
    df = df[df['TimeAfterDerate'].isna() | ((df['TimeAfterDerate'] > time_limit) | (df['FullDerate'] != 0))]

    return df.drop(columns=['PrevDerateTime', 'TimeAfterDerate'])

def split_data(df, split_date='2019-01-01'):
    """
    Split the data into training and testing sets based on a date.
    """
    # Raise ValueError if EventTimeStamp is not datetime
    if df['EventTimeStamp'].dtype != 'datetime64[ns]':
        raise ValueError("EventTimeStamp column must be of datetime type.")
    
    # Split the data into training and testing sets
    train_df = df[df['EventTimeStamp'] < split_date].copy()
    test_df = df[(df['EventTimeStamp'] >= split_date) & (df['EventTimeStamp']<='2024-01-01')].copy()

    return train_df, test_df

def clean_data(faults_filepath, diagnostics_filepath, split_date='2019-01-01', remove_service_locs=True, unnecessary_cols:list=None, nan_threshold=0.5, impute_method='ffill'):
    """
    Clean the data by merging faults and diagnostics, removing service locations, unnecessary columns, and imputing missing values.
    """
    # Merge faults and diagnostics
    print("Merging faults and diagnostics data...")
    df = merge_faults_and_diagnostics(faults_filepath, diagnostics_filepath)
    # Remove service locations and unnecessary columns
    print("Removing service locations and unnecessary columns...")
    df = remove_data(df, remove_service_locs=remove_service_locs, unnecessary_cols=unnecessary_cols, nan_threshold=nan_threshold)
    # Create target columns
    print("Creating target columns...")
    df = create_target_cols(df)
    # Remove data points that are after a derate for a certain time limit
    print("Removing data occuring right after a derate...")
    df = remove_after_derate(df)
    # Split the data into training and testing sets
    print("Splitting data...")
    train_df, test_df = split_data(df, split_date=split_date)
    # Scale and OHE the data
    print("Scaling and One-hot encoding the data...")
    train_df, test_df = scale_and_ohe_data(train_df, test_df)
    print(f"train_df: {train_df.head()}")
    # Impute missing values
    print(f"Imputing missing values using {impute_method} method...")
    train_df, test_df = impute_missing_values(train_df, test_df, method=impute_method)

    return train_df, test_df

def save_to_csv(train_df, test_df, file_name):
    """
    Save the train and test dataframes to CSV files.
    """
    train_file_path = f"{file_name}_train.csv"
    test_file_path = f"{file_name}_test.csv"
    
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print(f"Train and test dataframes saved to {train_file_path} and {test_file_path}.")

def clean_and_save(faults_filepath, diagnostics_filepath, split_date='2019-01-01', remove_service_locs=True, unnecessary_cols:list=None, nan_threshold=0.5, impute_method='ffill', filename='place_holder_filename'):
    print("Starting full cleaning and saving...")
    train_df, test_df = clean_data(faults_filepath, diagnostics_filepath, split_date, remove_service_locs, unnecessary_cols, nan_threshold, impute_method)
    print("Saving to csv...")
    save_to_csv(train_df, test_df, filename)

        
