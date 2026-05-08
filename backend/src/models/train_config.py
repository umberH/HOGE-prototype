"""
Config-driven model training with hyperparameter tuning support.
Replaces the original train_model.py with a more flexible, config-based approach.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import get_config_loader
from src.models.hyperparameter_tuner import HyperparameterTuner


class ConfigDrivenTrainer:
    """Train ML models using JSON configuration files."""

    def __init__(self, config_dir: str = "config"):
        """
        Args:
            config_dir: Path to configuration directory
        """
        self.config_loader = get_config_loader(config_dir)
        self.pipeline_config = self.config_loader.get_pipeline_config()
        self.pipeline = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess raw data."""
        print("Loading data...")

        data_path = self.pipeline_config['data']['raw_data_path']
        df = pd.read_csv(data_path)

        # Drop index column if exists
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

        # Clean Total_Income
        if 'Total_Income' in df.columns:
            df["Total_Income"] = (df["Total_Income"].astype(str)
                                .str.replace("$", "", regex=False)
                                .str.replace(",", "", regex=False))
            df["Total_Income"] = pd.to_numeric(df["Total_Income"], errors="coerce")

        # Convert target
        target_col = self.pipeline_config['features']['target']
        if target_col in df.columns:
            df[target_col] = df[target_col].map({"Y": 1, "N": 0})

        # Remove ID column (keep for later)
        id_col = self.pipeline_config['features']['id_column']
        if id_col in df.columns:
            self.loan_ids = df[id_col].copy()
            df = df.drop(columns=[id_col])
        else:
            self.loan_ids = pd.Series(range(len(df)), name="row_id")

        # Ensure numeric columns
        num_cols = self.pipeline_config['features']['numeric']
        for col in num_cols:
            if col in df.columns and col != 'DTI':  # DTI is created later
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Feature engineering: DTI
        if self.pipeline_config['preprocessing']['feature_engineering']['create_dti']:
            df["DTI"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)
            df["DTI"] = pd.to_numeric(df["DTI"], errors="coerce")

        print(f"Loaded {len(df)} records with {df.shape[1]} features")
        return df

    def prepare_features(self, df: pd.DataFrame):
        """Split data into features and target."""
        target_col = self.pipeline_config['features']['target']

        X = df.drop(columns=[target_col])
        y = df[target_col]

        return X, y

    def build_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build preprocessing pipeline based on config."""
        numeric_features = [col for col in self.pipeline_config['features']['numeric'] if col in X.columns]
        categorical_features = [col for col in self.pipeline_config['features']['categorical'] if col in X.columns]

        # Numeric transformer
        numeric_steps = [
            ("imputer", SimpleImputer(
                strategy=self.pipeline_config['preprocessing']['numeric_imputation_strategy']
            ))
        ]

        if self.pipeline_config['preprocessing']['scaling']:
            numeric_steps.append(("scale", StandardScaler()))

        numeric_transformer = Pipeline(steps=numeric_steps)

        # Categorical transformer
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(
                    strategy=self.pipeline_config['preprocessing']['categorical_imputation_strategy']
                )),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        # Combine
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return preprocessor

    def build_model(self, model_name: str, params: dict = None) -> object:
        """Build model from config.

        Args:
            model_name: Name of model in config
            params: Optional parameter overrides

        Returns:
            Instantiated model
        """
        model_config = self.config_loader.load_model_config(model_name)
        model_type = model_config['model_type']

        # Merge params
        all_params = model_config['fixed_params'].copy()

        if params:
            all_params.update(params)
        else:
            all_params.update(model_config['default_params'])

        # Add monotonic constraints for XGBoost if configured
        if model_type == 'XGBClassifier' and 'monotonic_constraints' in model_config:
            # Will be set dynamically when we know feature names
            pass

        # Instantiate model
        model_classes = {
            'XGBClassifier': XGBClassifier,
            'RandomForestClassifier': RandomForestClassifier,
            'LogisticRegression': LogisticRegression,
        }

        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")

        return model_classes[model_type](**all_params)

    def train(
        self,
        model_name: str,
        tune_hyperparameters: bool = False,
        custom_params: dict = None
    ) -> dict:
        """
        Train a model with optional hyperparameter tuning.

        Args:
            model_name: Name of model from config
            tune_hyperparameters: Whether to tune hyperparameters
            custom_params: Optional custom parameters (overrides config)

        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}\n")

        # Load data
        df = self.load_data()
        X, y = self.prepare_features(df)

        # Train/test split
        test_size = self.pipeline_config['data']['test_size']
        random_state = self.pipeline_config['data']['random_state']
        stratify = y if self.pipeline_config['data']['stratify'] else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        print(f"Train set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")

        # Build preprocessing pipeline
        preprocessor = self.build_preprocessing_pipeline(X)

        # Hyperparameter tuning
        best_params = custom_params
        if tune_hyperparameters and not custom_params:
            print("\n--- Hyperparameter Tuning ---")
            model_config = self.config_loader.load_model_config(model_name)
            tuner = HyperparameterTuner(model_config, self.pipeline_config)

            # Transform training data for tuning
            X_train_transformed = preprocessor.fit_transform(self.X_train)
            X_train_df = pd.DataFrame(
                X_train_transformed,
                columns=self._get_feature_names(preprocessor)
            )

            best_params, best_score = tuner.tune(X_train_df, self.y_train)

            # Save tuning results
            tuner.save_results(f"data/evaluation/tuning_results_{model_name}.json")

        # Build model
        model = self.build_model(model_name, best_params)

        # Create full pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model)
            ]
        )

        # Train
        print("\n--- Training Model ---")
        self.pipeline.fit(self.X_train, self.y_train)
        print("Training complete ✔")

        # Evaluate
        results = self.evaluate()

        return results

    def _get_feature_names(self, preprocessor) -> list:
        """Extract feature names from preprocessor."""
        output = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "named_steps"):
                if "onehot" in trans.named_steps:
                    ohe = trans.named_steps["onehot"]
                    output.extend(ohe.get_feature_names_out(cols))
                else:
                    output.extend(cols)
            else:
                output.extend(cols)
        return output

    def evaluate(self) -> dict:
        """Evaluate trained model."""
        print("\n--- Model Evaluation ---")

        # Predictions
        y_pred = self.pipeline.predict(self.X_test)
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

        # Metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_proba),
            'f1': f1_score(self.y_test, y_pred),
        }

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        return metrics

    def save_model(self, model_name: str):
        """Save trained model pipeline."""
        save_path = Path(self.pipeline_config['output']['model_save_path'])
        save_path.mkdir(parents=True, exist_ok=True)

        model_file = save_path / f"{model_name}.joblib"
        joblib.dump(self.pipeline, model_file)
        print(f"\nModel saved to: {model_file}")

    def save_predictions(self, model_name: str):
        """Save predictions on full dataset."""
        # Load full data
        df = self.load_data()
        X, y = self.prepare_features(df)

        # Predict
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)[:, 1]

        # Create output dataframe
        output_df = pd.DataFrame({
            'Loan_ID': self.loan_ids,
            'Actual': y,
            'Predicted': predictions,
            'Probability': probabilities
        })

        # Save
        save_path = Path(self.pipeline_config['output']['predictions_save_path'])
        save_path.mkdir(parents=True, exist_ok=True)

        output_file = save_path / f"scored_applications_{model_name}.csv"
        output_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ML models with config-driven approach")
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost_monotonic',
        help='Model name from config (default: xgboost_monotonic)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning'
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='config',
        help='Path to config directory (default: config)'
    )

    args = parser.parse_args()

    # Train model
    trainer = ConfigDrivenTrainer(config_dir=args.config_dir)
    results = trainer.train(
        model_name=args.model,
        tune_hyperparameters=args.tune
    )

    # Save model and predictions
    trainer.save_model(args.model)
    trainer.save_predictions(args.model)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
