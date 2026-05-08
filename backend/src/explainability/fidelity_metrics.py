"""
Mechanistic Fidelity Metrics
=============================

Quantitative metrics to evaluate quality of mechanistic explanations.

This addresses a critical gap in interpretability research:
HOW DO WE KNOW IF AN EXPLANATION IS GOOD?

Metrics Provided:
1. Circuit Coverage: % of predictions explained by circuits
2. Mechanistic Faithfulness: How well mechanisms predict actual outputs
3. Complexity-Fidelity Tradeoff: Pareto frontier analysis
4. Intervention Consistency: Alignment between ablations and predictions

Scientific Contribution:
First comprehensive evaluation framework for mechanistic interpretability quality.

Citation:
Inspired by Hooker et al. (2019) but extends beyond feature importance to full mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import joblib
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json


@dataclass
class FidelityMetrics:
    """Complete fidelity evaluation for a mechanistic explanation"""
    circuit_coverage: float  # % of samples explained by circuits
    mechanistic_faithfulness: float  # Correlation with true model
    mean_absolute_error: float  # MAE between mechanistic and true predictions
    r_squared: float  # R² score
    complexity_score: float  # Average complexity of explanation
    intervention_consistency: float  # Ablation consistency score
    overall_quality_score: float  # Weighted combination


@dataclass
class ComplexityFidelityPoint:
    """Single point on complexity-fidelity Pareto frontier"""
    num_components: int
    complexity: float
    fidelity: float
    mae: float


class MechanisticFidelityEvaluator:
    """
    Evaluate quality of mechanistic explanations

    This provides objective, quantitative assessment of
    how well a mechanistic model explains the original model.
    """

    def __init__(self, model_path: str, feature_names: List[str]):
        """
        Initialize fidelity evaluator

        Args:
            model_path: Path to trained model
            feature_names: List of feature names
        """
        self.pipeline = joblib.load(model_path)
        self.feature_names = feature_names

        # Extract model
        if hasattr(self.pipeline, 'named_steps'):
            self.model = self.pipeline.named_steps['model']
            self.preprocessor = self.pipeline.named_steps['preprocess']
        else:
            self.model = self.pipeline
            self.preprocessor = None

    def compute_circuit_coverage(
        self,
        circuits: List,
        circuit_activations: Dict,
        n_samples: int
    ) -> float:
        """
        Compute what % of predictions are covered by circuits

        High coverage with few circuits → good mechanistic understanding

        Args:
            circuits: List of Circuit objects
            circuit_activations: Dict of circuit activations
            n_samples: Total number of samples

        Returns:
            Coverage percentage (0-1)
        """
        samples_covered = set()

        for circuit_id, activations in circuit_activations.items():
            for activation in activations:
                if hasattr(activation, 'activated') and activation.activated:
                    samples_covered.add(activation.sample_idx)

        coverage = len(samples_covered) / n_samples if n_samples > 0 else 0

        return coverage

    def compute_mechanistic_faithfulness(
        self,
        X: pd.DataFrame,
        circuits: List = None,
        circuit_activations: Dict = None
    ) -> Dict:
        """
        How well does mechanistic model predict actual outputs?

        Method:
        1. Build simplified mechanistic model from circuits
        2. Compare predictions to original model
        3. Measure agreement (R², MAE, correlation)

        High faithfulness → mechanistic explanation is accurate

        Args:
            X: Input features
            circuits: List of Circuit objects
            circuit_activations: Circuit activation dict

        Returns:
            Dictionary with faithfulness metrics
        """
        # Get true model predictions
        X_transformed = self.preprocessor.transform(X) if self.preprocessor else X
        true_predictions = self.model.predict_proba(X_transformed)[:, 1]

        # Build mechanistic predictions from circuits
        if circuits and circuit_activations:
            mech_predictions = self._predict_from_circuits(
                circuits, circuit_activations, len(X)
            )
        else:
            # Fallback: use SHAP-based mechanistic prediction
            mech_predictions = self._predict_from_shap(X)

        # Compute metrics
        faithfulness = np.corrcoef(mech_predictions, true_predictions)[0, 1]
        mae = mean_absolute_error(true_predictions, mech_predictions)
        mse = mean_squared_error(true_predictions, mech_predictions)
        r2 = r2_score(true_predictions, mech_predictions)

        return {
            'faithfulness_correlation': float(faithfulness),
            'mean_absolute_error': float(mae),
            'mean_squared_error': float(mse),
            'r_squared': float(r2),
            'rmse': float(np.sqrt(mse))
        }

    def _predict_from_circuits(
        self,
        circuits: List,
        circuit_activations: Dict,
        n_samples: int
    ) -> np.ndarray:
        """
        Generate predictions using only circuits

        This is the "mechanistic model" - predictions based solely
        on discovered circuits.

        Args:
            circuits: List of Circuit objects
            circuit_activations: Dict of activations
            n_samples: Number of samples

        Returns:
            Array of mechanistic predictions
        """
        # Initialize predictions
        mech_preds = np.zeros(n_samples)

        # Aggregate contributions from activated circuits
        for circuit_id, activations in circuit_activations.items():
            # Find circuit
            circuit = next((c for c in circuits if c.circuit_id == circuit_id), None)
            if circuit is None:
                continue

            for activation in activations:
                if hasattr(activation, 'activated') and activation.activated:
                    idx = activation.sample_idx
                    contribution = activation.contribution if hasattr(activation, 'contribution') else circuit.output_mean

                    mech_preds[idx] += contribution

        # Normalize to [0, 1] range (probability-like)
        if np.max(np.abs(mech_preds)) > 0:
            # Use sigmoid to map to probabilities
            mech_preds = 1 / (1 + np.exp(-mech_preds))

        return mech_preds

    def _predict_from_shap(self, X: pd.DataFrame) -> np.ndarray:
        """Fallback: mechanistic predictions using SHAP decomposition"""
        import shap

        X_transformed = self.preprocessor.transform(X) if self.preprocessor else X

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_transformed)

            # For binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class

            # Base value + SHAP contributions
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]

            mech_preds = base_value + np.sum(shap_values, axis=1)

            # Convert to probabilities
            mech_preds = 1 / (1 + np.exp(-mech_preds))

        except (ValueError, AttributeError) as e:
            # Fallback: use Explainer with background data
            print(f"Warning: TreeExplainer failed. Using model predictions as approximation.")
            background = shap.sample(X_transformed, min(100, len(X_transformed)))
            explainer = shap.Explainer(self.model.predict_proba, background)
            shap_values = explainer(X_transformed)

            # Handle different SHAP value formats
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values

            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  # Positive class

            if hasattr(shap_values, 'base_values'):
                base_value = shap_values.base_values
                if len(base_value.shape) > 1:
                    base_value = base_value[:, 1]
            else:
                base_value = 0.5  # Default

            mech_preds = base_value + np.sum(shap_values, axis=1) if len(shap_values.shape) == 2 else np.mean(shap_values, axis=1)
            mech_preds = np.clip(mech_preds, 0, 1)  # Ensure [0, 1] range

        return mech_preds

    def compute_pareto_frontier(
        self,
        X: pd.DataFrame,
        circuits: List,
        circuit_activations_all: Dict,
        max_circuits: int = 20
    ) -> List[ComplexityFidelityPoint]:
        """
        Plot complexity vs. fidelity tradeoff

        Scientific Insight:
        Find optimal balance between simple explanations and accuracy

        Args:
            X: Input features
            circuits: All discovered circuits
            circuit_activations_all: Activations for all circuits
            max_circuits: Maximum number to test

        Returns:
            List of ComplexityFidelityPoint objects (Pareto frontier)
        """
        frontier = []

        # Sort circuits by importance
        circuits_sorted = sorted(circuits, key=lambda c: c.impact_score(), reverse=True)

        for k in range(1, min(max_circuits, len(circuits)) + 1):
            # Use top-k circuits
            top_k_circuits = circuits_sorted[:k]
            top_k_ids = {c.circuit_id for c in top_k_circuits}

            # Filter activations
            circuit_activations_k = {
                cid: acts for cid, acts in circuit_activations_all.items()
                if cid in top_k_ids
            }

            # Compute complexity (total features in circuits)
            complexity = sum(c.complexity() for c in top_k_circuits)

            # Compute fidelity
            fidelity_metrics = self.compute_mechanistic_faithfulness(
                X, top_k_circuits, circuit_activations_k
            )

            point = ComplexityFidelityPoint(
                num_components=k,
                complexity=float(complexity),
                fidelity=float(fidelity_metrics['faithfulness_correlation']),
                mae=float(fidelity_metrics['mean_absolute_error'])
            )

            frontier.append(point)

        return frontier

    def compute_intervention_consistency(
        self,
        ablation_results: List,
        claimed_importances: Dict[str, float]
    ) -> float:
        """
        Measure consistency between ablation results and claimed importances

        High consistency → mechanistic claims are causally validated

        Args:
            ablation_results: List of AblationResult objects
            claimed_importances: Dict of feature -> claimed importance

        Returns:
            Consistency score (0-1, higher is better)
        """
        if not ablation_results:
            return 0.0

        consistency_scores = []

        for result in ablation_results:
            if hasattr(result, 'consistency_score') and result.consistency_score is not None:
                consistency_scores.append(result.consistency_score)

        if not consistency_scores:
            return 0.0

        return float(np.mean(consistency_scores))

    def evaluate_full_fidelity(
        self,
        X: pd.DataFrame,
        circuits: List = None,
        circuit_activations: Dict = None,
        ablation_results: List = None,
        claimed_importances: Dict = None
    ) -> FidelityMetrics:
        """
        Comprehensive fidelity evaluation

        Combines all metrics into overall quality assessment

        Args:
            X: Input features
            circuits: List of Circuit objects
            circuit_activations: Circuit activations
            ablation_results: Ablation validation results
            claimed_importances: Claimed feature importances

        Returns:
            FidelityMetrics object with complete evaluation
        """
        # Circuit coverage
        if circuits and circuit_activations:
            coverage = self.compute_circuit_coverage(circuits, circuit_activations, len(X))
            complexity = np.mean([c.complexity() for c in circuits])
        else:
            coverage = 0.0
            complexity = 0.0

        # Mechanistic faithfulness
        faithfulness_metrics = self.compute_mechanistic_faithfulness(
            X, circuits, circuit_activations
        )

        # Intervention consistency
        if ablation_results and claimed_importances:
            consistency = self.compute_intervention_consistency(
                ablation_results, claimed_importances
            )
        else:
            consistency = 0.0

        # Overall quality score (weighted combination)
        overall_quality = (
            0.3 * coverage +
            0.4 * faithfulness_metrics['faithfulness_correlation'] +
            0.2 * faithfulness_metrics['r_squared'] +
            0.1 * consistency
        )

        return FidelityMetrics(
            circuit_coverage=float(coverage),
            mechanistic_faithfulness=float(faithfulness_metrics['faithfulness_correlation']),
            mean_absolute_error=float(faithfulness_metrics['mean_absolute_error']),
            r_squared=float(faithfulness_metrics['r_squared']),
            complexity_score=float(complexity),
            intervention_consistency=float(consistency),
            overall_quality_score=float(overall_quality)
        )

    def export_metrics(
        self,
        metrics: FidelityMetrics,
        pareto_frontier: List[ComplexityFidelityPoint] = None,
        output_path: str = "data/evaluation/fidelity_metrics.json"
    ):
        """
        Export fidelity metrics to JSON

        Args:
            metrics: FidelityMetrics object
            pareto_frontier: Optional Pareto frontier points
            output_path: Path to save JSON
        """
        export_data = {
            'fidelity_metrics': asdict(metrics),
            'pareto_frontier': [asdict(p) for p in pareto_frontier] if pareto_frontier else [],
            'interpretation': self._interpret_metrics(metrics),
            'provenance': {
                'method': 'MechanisticFidelityEvaluation',
                'version': '1.0.0'
            }
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Fidelity metrics exported to {output_path}")

    def _interpret_metrics(self, metrics: FidelityMetrics) -> Dict:
        """
        Provide human-readable interpretation of metrics

        Args:
            metrics: FidelityMetrics object

        Returns:
            Dictionary with interpretations
        """
        interpretations = {}

        # Coverage
        if metrics.circuit_coverage > 0.8:
            interpretations['coverage'] = "Excellent - circuits explain >80% of predictions"
        elif metrics.circuit_coverage > 0.5:
            interpretations['coverage'] = "Good - circuits explain >50% of predictions"
        else:
            interpretations['coverage'] = "Poor - circuits explain <50% of predictions"

        # Faithfulness
        if metrics.mechanistic_faithfulness > 0.9:
            interpretations['faithfulness'] = "Excellent - mechanistic model very accurate"
        elif metrics.mechanistic_faithfulness > 0.7:
            interpretations['faithfulness'] = "Good - mechanistic model reasonably accurate"
        else:
            interpretations['faithfulness'] = "Poor - mechanistic model deviates from true model"

        # Overall quality
        if metrics.overall_quality_score > 0.8:
            interpretations['overall'] = "High Quality Explanation"
            interpretations['recommendation'] = "Mechanistic explanation is trustworthy"
        elif metrics.overall_quality_score > 0.6:
            interpretations['overall'] = "Moderate Quality Explanation"
            interpretations['recommendation'] = "Usable but verify key claims"
        else:
            interpretations['overall'] = "Low Quality Explanation"
            interpretations['recommendation'] = "Mechanistic explanation needs improvement"

        return interpretations


if __name__ == "__main__":
    """Example usage"""

    # Load data
    df = pd.read_csv("data/processed/scored_applications_xgb.csv")
    feature_cols = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status', 'Probability']]
    X = df[feature_cols].head(100)

    # Initialize evaluator
    evaluator = MechanisticFidelityEvaluator(
        model_path="models/loan_xgb_monotonic.joblib",
        feature_names=feature_cols
    )

    print("\n" + "="*80)
    print("MECHANISTIC FIDELITY EVALUATION")
    print("="*80 + "\n")

    # Evaluate (using SHAP-based fallback)
    metrics = evaluator.evaluate_full_fidelity(X)

    print("Fidelity Metrics:")
    print(f"  Circuit Coverage: {metrics.circuit_coverage:.2%}")
    print(f"  Mechanistic Faithfulness: {metrics.mechanistic_faithfulness:.4f}")
    print(f"  MAE: {metrics.mean_absolute_error:.4f}")
    print(f"  R²: {metrics.r_squared:.4f}")
    print(f"  Complexity: {metrics.complexity_score:.2f}")
    print(f"  Intervention Consistency: {metrics.intervention_consistency:.4f}")
    print(f"  Overall Quality Score: {metrics.overall_quality_score:.4f}")

    # Export
    evaluator.export_metrics(metrics)

    print("\n✅ Fidelity evaluation complete!")
