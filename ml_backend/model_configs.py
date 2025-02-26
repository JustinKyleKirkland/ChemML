from dataclasses import dataclass
from typing import Any, Dict, Type

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""

    model_class: Type
    param_grid: Dict[str, Any]


# Common hyperparameter ranges
TREE_PARAMS = {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}

ENSEMBLE_PARAMS = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5]}

MODEL_CONFIGS = {
    "Linear Regression": ModelConfig(LinearRegression, {}),
    "Ridge Regression": ModelConfig(Ridge, {"alpha": [0.01, 0.1, 1.0, 10.0]}),
    "Lasso Regression": ModelConfig(Lasso, {"alpha": [0.01, 0.1, 1.0, 10.0]}),
    "Elastic Net Regression": ModelConfig(ElasticNet, {"alpha": [0.01, 0.1, 1.0, 10.0], "l1_ratio": [0.2, 0.5, 0.8]}),
    "Decision Tree Regression": ModelConfig(DecisionTreeRegressor, TREE_PARAMS),
    "Random Forest Regression": ModelConfig(RandomForestRegressor, {**TREE_PARAMS, "n_estimators": [50, 100, 200]}),
    "Gradient Boosting Regression": ModelConfig(GradientBoostingRegressor, {**ENSEMBLE_PARAMS, "max_depth": [3, 5, 7]}),
    "AdaBoost Regression": ModelConfig(AdaBoostRegressor, ENSEMBLE_PARAMS),
    "Support Vector Regression": ModelConfig(
        SVR, {"kernel": ["linear", "rbf"], "C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
    ),
    "Neural Network Regression": ModelConfig(
        MLPRegressor,
        {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
        },
    ),
    "K-Nearest Neighbors Regression": ModelConfig(
        KNeighborsRegressor, {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
    ),
    "Gaussian Process Regression": ModelConfig(
        GaussianProcessRegressor, {"alpha": [1e-10, 1e-5], "normalize_y": [True, False]}
    ),
}
