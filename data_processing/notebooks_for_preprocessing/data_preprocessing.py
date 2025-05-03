# Import statements
import pandas as pd
import numpy as np

def remove_service_locations(faults_filepath, diagnostics_filepath, radius=.05):
    """
    Remove data points that are near service locations.
    """
    faults = pd.read_csv(faults_filepath)
    diagnostics = pd.read_csv(diagnostics_filepath).pivot(index='FaultId', columns='Name', values='Value') # Immediately pivot diagnostic data on FaultId to then be merged

    df = faults.merge(diagnostics, how='left', left_on='RecordID', right_on='FaultId')

    # Convert diagnostic columns to appropriate dtypes
    for col, dtype in {
        "AcceleratorPedal":"float16",
        "BarometricPressure":"float16",
        "CruiseControlActive":"bool",
        "CruiseControlSetSpeed":"float16",
        "DistanceLtd":"float16",
        "EngineCoolantTemperature":"float16",
        "EngineLoad":"float16",
        "EngineOilPressure":"float16",
        "EngineOilTemperature":"float16",
        "EngineRpm":"float16",
        "EngineTimeLtd":"float16",
        "FuelLevel":"float16",
        "FuelLtd":"float32",
        "FuelRate":"float16",
        "FuelTemperature":"float16",
        "IgnStatus":"bool",
        "IntakeManifoldTemperature":"float16",
        "ParkingBrake":"bool",
        "Speed":"float16",
        "SwitchedBatteryVoltage":"float16",
        "Throttle":"float16",
        "TurboBoostPressure":"float16",
        "eventDescription":"str",
        "EquipmentID":"str"
    }.items():
        if dtype == 'bool':
            df[col] = df[col].astype('bool')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

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

    return df

def create_target_cols(df, time_limit=2):
    """
    Create target columns for the dataset.
    """
    # Create target column for Full Derate
    df['FullDerate'] = (df['spn'] == 5246).astype('int')
    # Order data by truck (EquipmentID) and time
    df = df.sort_values(['EquipmentID', 'EventTimeStamp'])
    # Ensure time is datetime
    df['EventTimeStamp'] = pd.to_datetime(df['EventTimeStamp'])

    # Create target column for Derate in next two hours
    df['NextDerateTime'] = df.where(df['FullDerate'] == 1)['EventTimeStamp']
    df['NextDerateTime'] = df.groupby('EquipmentID')['NextDerateTime'].transform('bfill')
    df['HoursUntilNextDerate'] = (df['NextDerateTime'] - df['EventTimeStamp']).dt.total_seconds() / 3600.0
    df['DerateInNextTwoHours'] = np.where(df['HoursUntilNextDerate'] <= 2, 1, 0)
    df['DerateInNextTwentyFourHours'] = np.where(df['HoursUntilNextDerate'] <= 24, 1, 0)

    # throw out anything after a derate for a short amount of time
    def remove_after_derate(df, time_limit=time_limit):
        """
        Remove data points that are after a derate for a certain time limit.
        """
        df['PrevDerateTime'] = df.where(df['FullDerate'] == 1)['EventTimeStamp']
        df['PrevDerateTime'] = df.groupby('EquipmentID')['PrevDerateTime'].transform('ffill')

        # Calculate the time difference from the last derate
        df['TimeAfterDerate'] = (df['EventTimeStamp'] - df['PrevDerateTime']).dt.total_seconds() / 3600.0

        # Filter out rows where TimeAfterDerate is less than the time limit while keeping rows if no derate occurs for that truck
        df = df[df['TimeAfterDerate'].isna() | (df['TimeAfterDerate'] > time_limit)]

        return df.drop(columns=['PrevDerateTime', 'TimeAfterDerate'])

    return remove_after_derate(df)

def ffill_nans(df): # alternatively try interpolation, moving averages, KNeighbors
    """
    Forward fill values for each EquipmentID group, resetting after each FullDerate == 1.
    """
    def fill_group(group):
        group = group.sort_values('EventTimeStamp')
        segment = group['FullDerate'].eq(1).cumsum() # Create segments based on FullDerate == 1
        return group.groupby(segment).ffill()

    return df.groupby('EquipmentID', group_keys=False).apply(fill_group)

def KNeighborsImputer(df, n_neighbors=5):
    # Do not include target
    """
    KNeighbors imputer for missing values on numeric columns only.
    """
    from sklearn.impute import KNNImputer

    # OHE first

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns



    imputer = KNNImputer(n_neighbors=n_neighbors)
    numeric_imputed_df = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    df_imputed = pd.concat([numeric_imputed_df, df[non_numeric_cols]], axis=1)

    return df_imputed

# Try SMOTE for imputation
def SMOTEImputer(df, target_col, feature_cols, n_neighbors=5):
    """
    Apply SMOTE only on feature_cols and target_col.
    """
    from imblearn.over_sampling import SMOTE

    df = df[feature_cols + [target_col]].dropna()
    X = df[feature_cols]
    y = df[target_col]

    smote = SMOTE(sampling_strategy='auto', k_neighbors=n_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
    df_resampled[target_col] = y_resampled
    return df_resampled

# Separate training and testing data based on before and after 2019-01-01
def xy_train_test_split(faults_filepath, diagnostics_filepath, feature_cols, target_col, imputer='ffill', SMOTE=False):
    #X_train = KNeighborsImputer(faults_and_diagnostics_train[feature_cols])
    #X_test = KNeighborsImputer(faults_and_diagnostics_test[feature_cols])
    print("Now removing service locations...")
    faults_and_diagnostics = remove_service_locations(faults_filepath, diagnostics_filepath, radius=.05)

    df_train = faults_and_diagnostics[faults_and_diagnostics['EventTimeStamp']<'2019-01-01']
    df_test = faults_and_diagnostics[(faults_and_diagnostics['EventTimeStamp']>='2019-01-01') & (faults_and_diagnostics['EventTimeStamp']<='2024-01-01')]
    
    print(f"Train set size: {df_train.shape}")
    print("Now creating target columns...")
    df_target_train = create_target_cols(df_train)
    df_target_test = create_target_cols(df_test)
    
    if imputer == 'KNeighbors':
        print("Now imputing missing values with KNeighborsImputer...")
        from sklearn.impute import KNNImputer
        imputer = KNNImputer()
        imputer.fit(df_target_train[feature_cols])
        print(f"Now imputing on these columns: {feature_cols}")

        faults_and_diagnostics_train[feature_cols] = imputer.transform(df_target_train[feature_cols])
        faults_and_diagnostics_test[feature_cols] = imputer.transform(df_target_test[feature_cols])

    elif imputer == 'ffill':
        print("Now imputing missing values with ffill...")
        faults_and_diagnostics_train = ffill_nans(df_target_train[feature_cols])
        faults_and_diagnostics_test = ffill_nans(df_target_test[feature_cols])
    else:
        raise ValueError("Invalid imputer type. Choose 'KNeighbors' or 'ffill'.")
    
    if SMOTE:
        print("Now balancing train dataset with SMOTE...")
        faults_and_diagnostics_train = SMOTEImputer(faults_and_diagnostics_train, target_col, feature_cols)
    
    X_train = faults_and_diagnostics_train[feature_cols]
    X_test = faults_and_diagnostics_test[feature_cols]
    y_train = faults_and_diagnostics_train[target_col]
    y_test = faults_and_diagnostics_test[target_col]

    # create train and test dataframes
    train_df = pd.concat([y_train, X_train], axis=1).rename(columns={target_col: 'target'})
    test_df = pd.concat([y_test, X_test], axis=1).rename(columns={target_col: 'target'})

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