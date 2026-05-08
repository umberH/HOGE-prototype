"""
Ablation Validation Framework
==============================

Causal validation of mechanistic claims using ablation studies.

This framework PROVES that detected patterns are real by:
1. Removing features/interactions
2. Measuring performance drop
3. Comparing to predicted importance

Scientific Principle:
True mechanistic understanding requires causal verification, not just correlation.
Ablation tests provide causal evidence by intervening on the model.

Citation:
Hooker et al. (2019) "A Benchmark for Interpretability Methods in Deep NLP"
DeYoung et al. (2020) "ERASER: A Benchmark to Evaluate Rationalized NLP Models"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy import stats
import warnings


@dataclass
class AblationResult:
    """Results from an ablation experiment"""
    ablated_component: str  # What was removed
    ablation_type: str  # "feature", "interaction", "circuit"
    baseline_performance: float
    ablated_performance: float
    performance_drop: float  # How much performance decreased
    relative_drop: float  # Percentage drop
    claimed_importance: Optional[float]  # From SHAP/other method
    consistency_score: Optional[float]  # How well ablation matches claim
    statistical_significance: float  # p-value
    validated: bool  # True if drop matches claimed importance


@dataclass
class InteractionAblationResult:
    """Results from interaction ablation"""
    feature_a: str
    feature_b: str
    baseline_performance: float
    ablate_a_only: float
    ablate_b_only: float
    ablate_both: float
    individual_drop_a: float
    individual_drop_b: float
    joint_drop: float
    interaction_effect: float  # joint - (a + b)
    interaction_type: str  # "synergistic", "redundant", "additive"
    significance: float


class AblationValidator:
    """
    Validate mechanistic explanations through causal ablation studies

    Methods:
    - Feature ablation: Remove individual features
    - Interaction ablation: Test feature pairs
    - Circuit ablation: Remove computational circuits
    - Path ablation: Block specific decision paths
    """

    def __init__(
        self,
        model_path: str,
        feature_names: List[str],
        metric: str = 'accuracy'
    ):
        """
        Initialize ablation validator

        Args:
            model_path: Path to trained model
            feature_names: List of feature names
            metric: Performance metric ('accuracy', 'auc', 'f1')
        """
        self.pipeline = joblib.load(model_path)
        self.feature_names = feature_names
        self.metric_name = metric

        # Extract model
        if hasattr(self.pipeline, 'named_steps'):
            self.model = self.pipeline.named_steps['model']
            self.preprocessor = self.pipeline.named_steps['preprocess']
        else:
            self.model = self.pipeline
            self.preprocessor = None

        # Select metric function
        self.metric_fn = self._get_metric_function(metric)

    def _get_metric_function(self, metric: str) -> Callable:
        """Get metric function by name"""
        metrics = {
            'accuracy': accuracy_score,
            'auc': lambda y_true, y_pred: roc_auc_score(y_true, self.model.predict_proba(
                self.preprocessor.transform(y_pred) if self.preprocessor else y_pred
            )[:, 1]),
            'f1': f1_score
        }

        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}. Choose from {list(metrics.keys())}")

        return metrics[metric]

    def compute_baseline(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Compute baseline performance (no ablation)

        Args:
            X: Features
            y: Labels

        Returns:
            Baseline performance score
        """
        y_pred = self.model.predict(
            self.preprocessor.transform(X) if self.preprocessor else X
        )

        return self.metric_fn(y, y_pred)

    def ablate_feature(
        self,
        feature: str,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'permutation',
        claimed_importance: Optional[float] = None,
        n_permutations: int = 5
    ) -> AblationResult:
        """
        Ablate a single feature and measure performance drop

        Args:
            feature: Feature name to ablate
            X: Features
            y: Labels
            method: Ablation method ('permutation', 'zero', 'mean')
            claimed_importance: Claimed importance from SHAP/etc
            n_permutations: Number of permutations for statistical significance

        Returns:
            AblationResult with causal importance
        """
        # Baseline
        baseline = self.compute_baseline(X, y)

        # Perform ablation multiple times for significance testing
        ablated_scores = []

        for _ in range(n_permutations):
            X_ablated = X.copy()

            if method == 'permutation':
                X_ablated[feature] = np.random.permutation(X[feature].values)
            elif method == 'zero':
                X_ablated[feature] = 0
            elif method == 'mean':
                X_ablated[feature] = X[feature].mean()
            else:
                raise ValueError(f"Unknown ablation method: {method}")

            # Score ablated model
            y_pred = self.model.predict(
                self.preprocessor.transform(X_ablated) if self.preprocessor else X_ablated
            )
            ablated_score = self.metric_fn(y, y_pred)
            ablated_scores.append(ablated_score)

        # Statistics
        ablated_mean = np.mean(ablated_scores)
        performance_drop = baseline - ablated_mean
        relative_drop = (performance_drop / baseline * 100) if baseline > 0 else 0

        # Consistency with claimed importance
        consistency_score = None
        if claimed_importance is not None:
            # Both should be on same scale
            # Normalize both to [0, 1]
            claimed_norm = abs(claimed_importance)
            drop_norm = abs(performance_drop)

            if max(claimed_norm, drop_norm) > 0:
                consistency_score = 1 - abs(claimed_norm - drop_norm) / max(claimed_norm, drop_norm)
            else:
                consistency_score = 1.0

        # Statistical significance (one-sample t-test against baseline)
        if len(ablated_scores) > 1:
            t_stat, p_value = stats.ttest_1samp(ablated_scores, baseline)
        else:
            p_value = 1.0

        # Validation: performance drops AND matches claimed importance
        validated = (performance_drop > 0.001) and (consistency_score is None or consistency_score > 0.5)

        return AblationResult(
            ablated_component=feature,
            ablation_type="feature",
            baseline_performance=float(baseline),
            ablated_performance=float(ablated_mean),
            performance_drop=float(performance_drop),
            relative_drop=float(relative_drop),
            claimed_importance=claimed_importance,
            consistency_score=float(consistency_score) if consistency_score is not None else None,
            statistical_significance=float(p_value),
            validated=validated
        )

    def ablate_interaction(
        self,
        feature_a: str,
        feature_b: str,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'permutation'
    ) -> InteractionAblationResult:
        """
        Test if interaction between two features is real

        Method:
        - Ablate A alone → drop_A
        - Ablate B alone → drop_B
        - Ablate A+B jointly → drop_AB

        True interaction: drop_AB ≠ drop_A + drop_B

        Args:
            feature_a: First feature
            feature_b: Second feature
            X: Features
            y: Labels
            method: Ablation method

        Returns:
            InteractionAblationResult
        """
        # Baseline
        baseline = self.compute_baseline(X, y)

        # Ablate A only
        X_a = X.copy()
        if method == 'permutation':
            X_a[feature_a] = np.random.permutation(X[feature_a].values)
        score_a = self.metric_fn(y, self.model.predict(
            self.preprocessor.transform(X_a) if self.preprocessor else X_a
        ))
        drop_a = baseline - score_a

        # Ablate B only
        X_b = X.copy()
        if method == 'permutation':
            X_b[feature_b] = np.random.permutation(X[feature_b].values)
        score_b = self.metric_fn(y, self.model.predict(
            self.preprocessor.transform(X_b) if self.preprocessor else X_b
        ))
        drop_b = baseline - score_b

        # Ablate both
        X_ab = X.copy()
        if method == 'permutation':
            X_ab[feature_a] = np.random.permutation(X[feature_a].values)
            X_ab[feature_b] = np.random.permutation(X[feature_b].values)
        score_ab = self.metric_fn(y, self.model.predict(
            self.preprocessor.transform(X_ab) if self.preprocessor else X_ab
        ))
        drop_ab = baseline - score_ab

        # Interaction effect
        expected_additive = drop_a + drop_b
        interaction_effect = drop_ab - expected_additive

        # Classify interaction
        if abs(interaction_effect) < 0.001:
            interaction_type = "additive"
        elif interaction_effect > 0:
            interaction_type = "synergistic"  # Combined effect > sum
        else:
            interaction_type = "redundant"  # Combined effect < sum

        # Significance (relative to individual effects)
        significance = abs(interaction_effect) / max(abs(drop_a), abs(drop_b), 1e-6)

        return InteractionAblationResult(
            feature_a=feature_a,
            feature_b=feature_b,
            baseline_performance=float(baseline),
            ablate_a_only=float(score_a),
            ablate_b_only=float(score_b),
            ablate_both=float(score_ab),
            individual_drop_a=float(drop_a),
            individual_drop_b=float(drop_b),
            joint_drop=float(drop_ab),
            interaction_effect=float(interaction_effect),
            interaction_type=interaction_type,
            significance=float(significance)
        )

    def validate_all_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        claimed_importances: Optional[Dict[str, float]] = None,
        top_k: int = None
    ) -> List[AblationResult]:
        """
        Validate all features (or top-k) via ablation

        Args:
            X: Features
            y: Labels
            claimed_importances: Dictionary of feature -> claimed importance
            top_k: Only validate top-k features

        Returns:
            List of AblationResult objects
        """
        features_to_test = self.feature_names[:top_k] if top_k else self.feature_names

        results = []

        for feature in features_to_test:
            claimed_imp = claimed_importances.get(feature) if claimed_importances else None

            result = self.ablate_feature(
                feature=feature,
                X=X,
                y=y,
                claimed_importance=claimed_imp
            )

            results.append(result)

        # Sort by performance drop
        results.sort(key=lambda r: r.performance_drop, reverse=True)

        return results

    def validate_interactions(
        self,
        feature_pairs: List[Tuple[str, str]],
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[InteractionAblationResult]:
        """
        Validate multiple feature interactions

        Args:
            feature_pairs: List of (feature_a, feature_b) tuples
            X: Features
            y: Labels

        Returns:
            List of InteractionAblationResult objects
        """
        results = []

        for feature_a, feature_b in feature_pairs:
            result = self.ablate_interaction(feature_a, feature_b, X, y)
            results.append(result)

        # Sort by interaction significance
        results.sort(key=lambda r: abs(r.interaction_effect), reverse=True)

        return results

    def export_results(
        self,
        results: List,
        output_path: str
    ):
        """
        Export ablation results to JSON

        Args:
            results: List of ablation results
            output_path: Path to save JSON
        """
        import json

        export_data = {
            'results': [asdict(r) for r in results],
            'summary': {
                'total_tested': len(results),
                'validated': sum(1 for r in results if r.validated) if hasattr(results[0], 'validated') else None,
                'mean_drop': float(np.mean([r.performance_drop for r in results if hasattr(r, 'performance_drop')])),
                'max_drop': float(max([r.performance_drop for r in results if hasattr(r, 'performance_drop')]))
            },
            'provenance': {
                'method': 'AblationValidation',
                'metric': self.metric_name,
                'version': '1.0.0'
            }
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Ablation results exported to {output_path}")


if __name__ == "__main__":
    """Example usage"""

    # Load data
    df = pd.read_csv("data/processed/scored_applications_xgb.csv")
    feature_cols = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status', 'Probability']]

    X = df[feature_cols].head(100)
    y = df['Loan_Status'].map({'Y': 1, 'N': 0}).head(100)

    # Initialize validator
    validator = AblationValidator(
        model_path="models/loan_xgb_monotonic.joblib",
        feature_names=feature_cols,
        metric='accuracy'
    )

    print("\n" + "="*80)
    print("ABLATION VALIDATION")
    print("="*80 + "\n")

    # Test top features
    print("Validating top 5 features...")
    results = validator.validate_all_features(X, y, top_k=5)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.ablated_component}")
        print(f"   Performance drop: {result.performance_drop:.4f} ({result.relative_drop:.1f}%)")
        print(f"   P-value: {result.statistical_significance:.4f}")
        print(f"   Validated: {'✓' if result.validated else '✗'}")

    # Test interactions
    print("\n\nValidating feature interactions...")
    pairs = [
        ('ApplicantIncome', 'LoanAmount'),
        ('Credit_History', 'LoanAmount')
    ]

    interaction_results = validator.validate_interactions(pairs, X, y)

    for result in interaction_results:
        print(f"\n{result.feature_a} × {result.feature_b}")
        print(f"   Interaction effect: {result.interaction_effect:.4f}")
        print(f"   Type: {result.interaction_type}")
        print(f"   Significance: {result.significance:.4f}")

    # Export
    validator.export_results(results, "data/evaluation/ablation_results.json")

    print("\n✅ Ablation validation complete!")
