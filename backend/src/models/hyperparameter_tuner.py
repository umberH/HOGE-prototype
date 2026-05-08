"""
Hyperparameter tuning module using Optuna and GridSearchCV.
Supports multiple model types with config-driven tuning.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple
import json
import joblib
from pathlib import Path


class HyperparameterTuner:
    """Hyperparameter tuning for ML models using Optuna or GridSearch."""

    def __init__(self, model_config: Dict[str, Any], pipeline_config: Dict[str, Any]):
        """
        Args:
            model_config: Model configuration from model_hyperparameters.json
            pipeline_config: Pipeline configuration from pipeline_config.json
        """
        self.model_config = model_config
        self.pipeline_config = pipeline_config
        self.model_type = model_config['model_type']
        self.tuning_strategy = model_config.get('tuning_strategy', 'optuna')
        self.best_params = None
        self.best_score = None
        self.study = None

    def _get_model_class(self):
        """Get model class from string name."""
        model_classes = {
            'XGBClassifier': XGBClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
        }
        if self.model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model_classes[self.model_type]

    def _get_monotonic_constraints_string(self, feature_names):
        """Build monotonic constraints string for XGBoost."""
        if 'monotonic_constraints' not in self.model_config:
            return None

        mono_map = self.model_config['monotonic_constraints']
        constraints = []

        for feat in feature_names:
            if feat in mono_map:
                constraints.append(mono_map[feat])
            else:
                constraints.append(mono_map.get('other_features', 0))

        return "(" + ",".join(map(str, constraints)) + ")"

    def tune_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 100,
        timeout: int = 3600
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            timeout: Timeout in seconds

        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"Starting Optuna hyperparameter tuning for {self.model_type}...")
        print(f"Trials: {n_trials}, Timeout: {timeout}s")

        tuning_config = self.pipeline_config.get('hyperparameter_tuning', {})
        n_trials = tuning_config.get('n_trials', n_trials)
        timeout = tuning_config.get('timeout_seconds', timeout)

        def objective(trial):
            # Sample hyperparameters from tuning grid
            params = self.model_config['fixed_params'].copy()

            for param_name, param_values in self.model_config['tuning_grid'].items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Numeric parameters
                        if all(isinstance(v, int) for v in param_values):
                            params[param_name] = trial.suggest_int(
                                param_name, min(param_values), max(param_values)
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name, min(param_values), max(param_values)
                            )
                    else:
                        # Categorical parameters
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_values
                        )

            # Add monotonic constraints for XGBoost
            if self.model_type == 'XGBClassifier' and 'monotonic_constraints' in self.model_config:
                feature_names = X_train.columns.tolist()
                mono_string = self._get_monotonic_constraints_string(feature_names)
                if mono_string:
                    params['monotone_constraints'] = mono_string

            # Create model
            model_class = self._get_model_class()
            model = model_class(**params)

            # Cross-validation
            cv_config = self.pipeline_config['evaluation']['cross_validation']
            cv_folds = cv_config.get('folds', 5)

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model.fit(X_tr, y_tr)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)

            return np.mean(scores)

        # Create study
        direction = self.pipeline_config['hyperparameter_tuning'].get(
            'optimization_direction', 'maximize'
        )
        pruner = optuna.pruners.MedianPruner() if tuning_config.get('pruning', True) else None

        self.study = optuna.create_study(direction=direction, pruner=pruner)
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\nBest ROC-AUC: {self.best_score:.4f}")
        print(f"Best parameters: {json.dumps(self.best_params, indent=2)}")

        return self.best_params, self.best_score

    def tune_with_gridsearch(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"Starting GridSearchCV for {self.model_type}...")

        # Build parameter grid
        param_grid = self.model_config['tuning_grid']

        # Create base model
        model_class = self._get_model_class()
        base_params = self.model_config['fixed_params'].copy()

        # Add monotonic constraints for XGBoost
        if self.model_type == 'XGBClassifier' and 'monotonic_constraints' in self.model_config:
            feature_names = X_train.columns.tolist()
            mono_string = self._get_monotonic_constraints_string(feature_names)
            if mono_string:
                base_params['monotone_constraints'] = mono_string

        model = model_class(**base_params)

        # Setup cross-validation
        cv_config = self.pipeline_config['evaluation']['cross_validation']
        cv_folds = cv_config.get('folds', 5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # GridSearchCV
        scorer = make_scorer(roc_auc_score, needs_proba=True)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_

        print(f"\nBest ROC-AUC: {self.best_score:.4f}")
        print(f"Best parameters: {json.dumps(self.best_params, indent=2)}")

        return self.best_params, self.best_score

    def tune_with_randomsearch(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_iter: int = 50
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using RandomizedSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            n_iter: Number of random iterations

        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"Starting RandomizedSearchCV for {self.model_type}...")

        # Build parameter distribution
        param_dist = self.model_config['tuning_grid']
        n_iter = self.model_config.get('n_iter', n_iter)

        # Create base model
        model_class = self._get_model_class()
        base_params = self.model_config['fixed_params'].copy()

        # Add monotonic constraints for XGBoost
        if self.model_type == 'XGBClassifier' and 'monotonic_constraints' in self.model_config:
            feature_names = X_train.columns.tolist()
            mono_string = self._get_monotonic_constraints_string(feature_names)
            if mono_string:
                base_params['monotone_constraints'] = mono_string

        model = model_class(**base_params)

        # Setup cross-validation
        cv_config = self.pipeline_config['evaluation']['cross_validation']
        cv_folds = cv_config.get('folds', 5)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # RandomizedSearchCV
        scorer = make_scorer(roc_auc_score, needs_proba=True)

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_

        print(f"\nBest ROC-AUC: {self.best_score:.4f}")
        print(f"Best parameters: {json.dumps(self.best_params, indent=2)}")

        return self.best_params, self.best_score

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using configured strategy.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (best_params, best_score)
        """
        strategy = self.tuning_strategy.lower()

        if strategy == 'optuna':
            return self.tune_with_optuna(X_train, y_train)
        elif strategy == 'grid':
            return self.tune_with_gridsearch(X_train, y_train)
        elif strategy == 'random':
            return self.tune_with_randomsearch(X_train, y_train)
        else:
            raise ValueError(f"Unknown tuning strategy: {strategy}. "
                           f"Use 'optuna', 'grid', or 'random'")

    def save_results(self, output_path: str):
        """Save tuning results to JSON file."""
        results = {
            'model_type': self.model_type,
            'tuning_strategy': self.tuning_strategy,
            'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score else None,
        }

        if self.study:
            results['optuna_study'] = {
                'n_trials': len(self.study.trials),
                'best_trial_number': self.study.best_trial.number,
                'best_value': self.study.best_value,
            }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Tuning results saved to: {output_path}")
