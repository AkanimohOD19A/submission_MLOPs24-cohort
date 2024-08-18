import logging

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class HyperparameterOptimization:
    """
    Class for doing hyperparameter optimization.
    """

    def __init__(
            self,
            X_train: pd.DataFrame, y_train: pd.Series,
            X_test: pd.DataFrame, y_test: pd.Series,
    ) -> None:
        """Initialize the class with the training and test data."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize_randomforest(self, trial: optuna.Trial) -> float:
        """
        Method for optimizing Random Forest.
        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        maX_depth = trial.suggest_int("maX_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            maX_depth=maX_depth,
            min_samples_split=min_samples_split,
        )
        reg.fit(self.X_train, self.y_train)
        val_accuracy = reg.score(self.X_test, self.y_test)
        return val_accuracy

    def optimize_lightgbm(self, trial: optuna.Trial) -> float:
        """
        Method for Optimizing LightGBM.
        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        maX_depth = trial.suggest_int("maX_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            maX_depth=maX_depth,
        )
        reg.fit(self.X_train, self.y_train)
        val_accuracy = reg.score(self.X_test, self.y_test)
        return val_accuracy

    def optimize_xgboost_regressor(self, trial: optuna.Trial) -> float:
        """
        Method for Optimizing Xgboost
        """
        param = {
            "maX_depth": trial.suggest_int("maX_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate", 1e-7, 10.0
            ),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        }
        reg = xgb.XGBRegressor(**param)
        reg.fit(self.X_train, self.y_train)
        val_accuracy = reg.score(self.X_test, self.y_test)
        return val_accuracy


class ModelTrainer:
    """
    Class for training models.
    """

    def __init__(
            self,
            X_train, y_train, X_test, y_test,
    ) -> None:
        """Initialize the class with the training and test data."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def linear_regression_trainer(self) -> RegressorMixin:
        """
        It trains the random forest model.
        Args:fine_tuning: None - Linear Regression are simple.
        """
        logging.info("Started training Linear Regression model.")
        try:
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
            return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def random_forest_trainer(
            self, fine_tuning: bool = True
    ) -> RegressorMixin:
        """
        It trains the random forest model.

        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used. Defaults to True (optional).

        """
        logging.info("Started training Random Forest model.")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimization(
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=10)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                maX_depth = trial.params["maX_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                reg = RandomForestRegressor(
                    n_estimators=n_estimators,
                    maX_depth=maX_depth,
                    min_samples_split=min_samples_split,
                )
                reg.fit(self.X_train, self.y_train)
                return reg
            else:
                model = RandomForestRegressor(
                    n_estimators=152, maX_depth=20, min_samples_split=17
                )
                model.fit(self.X_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def lightgbm_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """
        It trains the LightGBM model.

        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, Defaults to True (optional).
        """

        logging.info("Started training LightGBM model.")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimization(
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_lightgbm, n_trials=10)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                maX_depth = trial.params["maX_depth"]
                learning_rate = trial.params["learning_rate"]
                reg = LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    maX_depth=maX_depth,
                )
                reg.fit(self.X_train, self.y_train)
                return reg
            else:
                model = LGBMRegressor(
                    n_estimators=200, learning_rate=0.01, maX_depth=20
                )
                model.fit(self.X_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training LightGBM model.")
            logging.error(e)
            return None

    def xgboost_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """
        It trains the xgboost model.

        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, Defaults to True (optional).
        """

        logging.info("Started training XGBoost model.")
        try:
            if fine_tuning:
                hy_opt = HyperparameterOptimization(
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost_regressor, n_trials=10)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                maX_depth = trial.params["maX_depth"]
                reg = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    maX_depth=maX_depth,
                )
                reg.fit(self.X_train, self.y_train)
                return reg

            else:
                model = xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.01, maX_depth=20
                )
                model.fit(self.X_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training XGBoost model.")
            logging.error(e)
            return None
