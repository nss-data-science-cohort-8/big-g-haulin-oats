# Import statements
import pandas as pd
import numpy as np

# Read in data
faults = pd.read_csv("../../data/J1939Faults.csv")
diagnostics = pd.read_csv("../../data/VehicleDiagnosticOnboardData.csv").pivot(index='FaultId', columns='Name', values='Value') # Immediately pivot diagnostic data on FaultId to then be merged
faults_and_diagnostics = faults.merge(diagnostics, how='left', left_on='RecordID', right_on='FaultId')

# Create FullDerate column indicating if the fault is a Full Derate
faults_and_diagnostics['FullDerate'] = faults_and_diagnostics['spn'].apply(lambda x: 1 if x == 5246 else 0)

# Order data by truck (EquipmentID) and time
faults_and_diagnostics = faults_and_diagnostics.sort_values(['EquipmentID', 'EventTimeStamp'])

# Ensure time is datetime
faults_and_diagnostics['EventTimeStamp'] = pd.to_datetime(faults_and_diagnostics['EventTimeStamp'])

# Mark Full Derate time and backfill for that truck specifically
faults_and_diagnostics['NextDerateTime'] = faults_and_diagnostics.where(faults_and_diagnostics['FullDerate']==1)['EventTimeStamp']
faults_and_diagnostics['NextDerateTime'] = faults_and_diagnostics.groupby('EquipmentID')['NextDerateTime'].fillna(method='bfill')

# Calculate hours until next derate and whether or not a derate is occuring in next two hours (target variables)
faults_and_diagnostics['HoursUntilNextDerate'] = (faults_and_diagnostics['NextDerateTime'] - faults_and_diagnostics['EventTimeStamp']).dt.total_seconds()/3600.0
faults_and_diagnostics['DerateInNextTwoHours'] = np.where(faults_and_diagnostics['HoursUntilNextDerate'] <= 2, 1, 0)

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
        faults_and_diagnostics[col] = faults_and_diagnostics[col].astype('bool')
    else:
        faults_and_diagnostics[col] = pd.to_numeric(faults_and_diagnostics[col], errors='coerce').astype(dtype) # Need to do this because certain numeric columns need to be coerced

# Separate training and testing data based on before and after 2019-01-01
faults_and_diagnostics_train = faults_and_diagnostics[faults_and_diagnostics['EventTimeStamp']<'2019-01-01']
faults_and_diagnostics_test = faults_and_diagnostics[(faults_and_diagnostics['EventTimeStamp']>='2019-01-01') & (faults_and_diagnostics['EventTimeStamp']<='2024-01-01') ]

def xy_train_test_split(feature_cols, target_col):
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
    train_file_path = f"../preprocessed_data/{file_name}_train.csv"
    test_file_path = f"../preprocessed_data/{file_name}_test.csv"
    
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print(f"Train and test dataframes saved to {train_file_path} and {test_file_path}.")
