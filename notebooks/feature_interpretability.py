
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score, mean_absolute_error

import shap
import xgboost

xgb = xgboost.XGBRegressor(tree_method="hist", enable_categorical = True).fit(X_train, y_train)

def plot_explainer(fitted_model, X_test):

    explainer = shap.TreeExplainer(fitted_model)
    explanation = explainer(X_test)



    def waterfall_plot(i):
        shap.plots.waterfall(explanation[i])

    def force_plot(i):
        shap.plots.initjs()
        shap.plots.force(explanation[i])

    def global_predictions_plot():
        shap.plots.scatter(explanation[:, "years_since_renovation"], color=explanation)

    def beeswarm_plot():
        shap.plots.beeswarm(explanation)

    def avg_bar_plot():
        shap.plots.bar(explanation)