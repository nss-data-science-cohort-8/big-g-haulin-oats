# Imports
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import mlflow

# Function to calculate money made/lost
def monetary_impact(y_true, y_pred, tp_value=4000, fp_cost=500):
    """
    - tp_value: Value gained from a true positive
    - fp_cost: Cost of a false positive
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    tp_revenue = tp * tp_value
    fp_cost_total = fp * fp_cost
    net_impact = tp_revenue - fp_cost_total
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "tp_revenue": tp_revenue,
        "fp_cost": fp_cost_total,
        "net_monetary_impact": net_impact
    }

def analyze_model_predictions(predictions_path, tp_value=4000, fp_cost=500):
    predictions_df = pd.read_csv(predictions_path)
    impact = monetary_impact(
        predictions_df['actual'], 
        predictions_df['predicted'],
        tp_value,
        fp_cost
    )
    results_df = pd.DataFrame({
        'Metric': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives',
                  'TP Revenue ($)', 'FP Cost ($)', 'Net Impact ($)'],
        'Value': [impact['true_positives'], impact['false_positives'], 
                 impact['true_negatives'], impact['false_negatives'],
                 impact['tp_revenue'], impact['fp_cost'], 
                 impact['net_monetary_impact']]
    })
    
    return results_df, impact

def run_monetary_analysis(file_name, tp_value=400, fp_cost=500):
    output_dir = Path("data_processing/model_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cost_analyses_dir = Path("data_processing/model_cost_analyses")
    cost_analyses_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_path = output_dir / f"{Path(file_name).name}.csv"
    results_path = cost_analyses_dir / f"monetary_analysis_{Path(file_name).name}.csv"
    
    results_df, impact = analyze_model_predictions(predictions_path, tp_value, fp_cost)
    results_df.to_csv(results_path, index=False)
    
    with mlflow.start_run():
        mlflow.log_metric("monetary_net_impact", impact['net_monetary_impact'])
        mlflow.log_metric("true_positives", impact['true_positives'])
        mlflow.log_metric("false_positives", impact['false_positives'])
        mlflow.log_metric("tp_revenue", impact['tp_revenue'])
        mlflow.log_metric("fp_cost", impact['fp_cost'])
    
    print(f"Monetary Analysis Summary for {file_name}:")
    print(f"True Positives: {impact['true_positives']} (Revenue: ${impact['tp_revenue']})")
    print(f"False Positives: {impact['false_positives']} (Cost: ${impact['fp_cost']})")
    print(f"Net Monetary Impact: ${impact['net_monetary_impact']}")
    print(f"Detailed results saved to {results_path}")
    
    return results_df


results = run_monetary_analysis("derate_in_next_twenty_four_hours_ffill", tp_value=4000, fp_cost=500)

 