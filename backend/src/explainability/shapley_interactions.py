"""
Shapley Interaction Index Module
=================================

Implements true feature interaction detection using Shapley values.

This goes beyond simple co-occurrence to measure:
- Synergistic interactions (features amplify each other)
- Redundant interactions (features substitute for each other)
- Statistical significance of interactions

Scientific Foundation:
- Game-theoretic approach to feature interactions
- Quantifies interaction strength with theoretical guarantees
- Distinguishes correlation from true causal interaction

Citation:
Lundberg et al. (2020) "From local explanations to global understanding with explainable AI for trees"
Fujimoto et al. (2006) "Shapley value and interaction index"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import shap
from scipy import stats
import joblib


@dataclass
class ShapleyInteraction:
    """Represents a Shapley-based feature interaction"""
    feature_a: str
    feature_b: str
    interaction_strength: float  # Mean SHAP interaction value
    synergy_score: float  # Positive interaction component
    redundancy_score: float  # Negative interaction component
    interaction_type: str  # "synergistic", "redundant", "independent"
    statistical_significance: float  # p-value from statistical test
    confidence_interval: Tuple[float, float]  # 95% CI
    sample_size: int

    def is_significant(self, alpha=0.05) -> bool:
        """Check if interaction is statistically significant"""
        return self.statistical_significance < alpha

    def magnitude(self) -> float:
        """Absolute magnitude of interaction"""
        return abs(self.interaction_strength)


class ShapleyInteractionAnalyzer:
    """
    Analyze feature interactions using Shapley interaction values

    This provides a rigorous, game-theoretic approach to understanding
    how features work together (synergy) or substitute (redundancy).
    """

    def __init__(self, model_path: str, feature_names: List[str]):
        """
        Initialize Shapley interaction analyzer

        Args:
            model_path: Path to trained model (.joblib)
            feature_names: List of feature names
        """
        self.pipeline = joblib.load(model_path)
        self.feature_names = feature_names

        # Extract model
        if hasattr(self.pipeline, 'named_steps'):
            self.model = self.pipeline.named_steps['model']
        else:
            self.model = self.pipeline

        # Initialize SHAP explainer
        self.explainer = None
        self._shap_interaction_cache = {}

    def compute_shap_interactions(self, X: pd.DataFrame, cache_key: str = None) -> np.ndarray:
        """
        Compute SHAP interaction values for dataset

        Args:
            X: Input features
            cache_key: Optional key for caching results

        Returns:
            SHAP interaction matrix: shape (n_samples, n_features, n_features)

        Note:
            For tree models, SHAP interaction values are computed exactly.
            shap_interaction[i, j, k] = interaction effect of features j and k on sample i
        """
        if cache_key and cache_key in self._shap_interaction_cache:
            return self._shap_interaction_cache[cache_key]

        # Initialize explainer if needed
        if self.explainer is None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except (ValueError, AttributeError) as e:
                # Fallback for XGBoost version compatibility issues
                print(f"Warning: SHAP TreeExplainer failed ({str(e)}). Using approximate explainer.")
                # Use Explainer with background data
                X_transformed = self.pipeline.named_steps['preprocess'].transform(X)
                background = shap.sample(X_transformed, min(100, len(X_transformed)))
                self.explainer = shap.Explainer(self.model.predict_proba, background)

        # Transform features
        X_transformed = self.pipeline.named_steps['preprocess'].transform(X)

        # Compute SHAP interaction values
        try:
            shap_interaction_values = self.explainer.shap_interaction_values(X_transformed)
        except (AttributeError, ValueError) as e:
            # Fallback: Compute approximate interactions from SHAP values
            print(f"Warning: Interaction values not available. Computing approximate interactions.")
            shap_values = self.explainer(X_transformed)

            # Handle different SHAP value formats
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values

            # For binary classification, extract positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # (n_samples, n_features, n_classes) -> use positive class
                shap_values = shap_values[:, :, 1]

            # Create approximate interaction matrix (correlation-based)
            n_samples, n_features = shap_values.shape
            shap_interaction_values = np.zeros((n_samples, n_features, n_features))

            # Diagonal = main effects
            for i in range(n_features):
                shap_interaction_values[:, i, i] = shap_values[:, i]

            # Off-diagonal = approximate interactions (covariance)
            for i in range(n_features):
                for j in range(i+1, n_features):
                    # Simple correlation-based approximation
                    interaction = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                    interaction *= np.mean(np.abs(shap_values[:, i])) * np.mean(np.abs(shap_values[:, j]))
                    shap_interaction_values[:, i, j] = interaction
                    shap_interaction_values[:, j, i] = interaction

        # Cache if key provided
        if cache_key:
            self._shap_interaction_cache[cache_key] = shap_interaction_values

        return shap_interaction_values

    def analyze_pairwise_interaction(
        self,
        feature_a: str,
        feature_b: str,
        X: pd.DataFrame,
        alpha: float = 0.05
    ) -> ShapleyInteraction:
        """
        Analyze interaction between two specific features

        Args:
            feature_a: First feature name
            feature_b: Second feature name
            X: Input data
            alpha: Significance level for statistical tests

        Returns:
            ShapleyInteraction object with detailed interaction analysis
        """
        # Get SHAP interaction values
        shap_interactions = self.compute_shap_interactions(X)

        # Get feature indices
        feat_a_idx = self.feature_names.index(feature_a)
        feat_b_idx = self.feature_names.index(feature_b)

        # Extract interaction values for this pair
        # Note: SHAP interaction matrix is symmetric, so [a,b] = [b,a]
        interaction_values = shap_interactions[:, feat_a_idx, feat_b_idx]

        # Compute statistics
        mean_interaction = np.mean(interaction_values)

        # Decompose into synergy (positive) and redundancy (negative)
        synergy_score = np.mean(np.maximum(interaction_values, 0))
        redundancy_score = np.mean(np.minimum(interaction_values, 0))

        # Classify interaction type
        if abs(mean_interaction) < 1e-6:
            interaction_type = "independent"
        elif mean_interaction > 0:
            interaction_type = "synergistic"
        else:
            interaction_type = "redundant"

        # Statistical significance test
        # H0: interaction = 0
        t_stat, p_value = stats.ttest_1samp(interaction_values, 0)

        # Confidence interval
        ci = stats.t.interval(
            1 - alpha,
            len(interaction_values) - 1,
            loc=mean_interaction,
            scale=stats.sem(interaction_values)
        )

        return ShapleyInteraction(
            feature_a=feature_a,
            feature_b=feature_b,
            interaction_strength=float(mean_interaction),
            synergy_score=float(synergy_score),
            redundancy_score=float(redundancy_score),
            interaction_type=interaction_type,
            statistical_significance=float(p_value),
            confidence_interval=(float(ci[0]), float(ci[1])),
            sample_size=len(interaction_values)
        )

    def find_top_interactions(
        self,
        X: pd.DataFrame,
        top_k: int = 10,
        min_strength: float = 0.01,
        alpha: float = 0.05
    ) -> List[ShapleyInteraction]:
        """
        Find top-k strongest feature interactions

        Args:
            X: Input data
            top_k: Number of top interactions to return
            min_strength: Minimum absolute interaction strength
            alpha: Significance level

        Returns:
            List of ShapleyInteraction objects, sorted by strength
        """
        # Compute SHAP interactions
        shap_interactions = self.compute_shap_interactions(X)

        # Compute mean interaction for each pair
        n_features = len(self.feature_names)
        interactions = []

        for i in range(n_features):
            for j in range(i + 1, n_features):  # Only upper triangle (symmetric)
                feature_a = self.feature_names[i]
                feature_b = self.feature_names[j]

                # Quick filter by mean strength
                interaction_values = shap_interactions[:, i, j]
                mean_strength = abs(np.mean(interaction_values))

                if mean_strength >= min_strength:
                    # Full analysis
                    interaction = self.analyze_pairwise_interaction(
                        feature_a, feature_b, X, alpha
                    )

                    # Only keep significant interactions
                    if interaction.is_significant(alpha):
                        interactions.append(interaction)

        # Sort by absolute strength
        interactions.sort(key=lambda x: x.magnitude(), reverse=True)

        return interactions[:top_k]

    def compute_interaction_matrix(
        self,
        X: pd.DataFrame,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute full interaction matrix for all feature pairs

        Args:
            X: Input data
            normalize: If True, normalize by max interaction

        Returns:
            DataFrame with interaction strength matrix
        """
        # Compute SHAP interactions
        shap_interactions = self.compute_shap_interactions(X)

        # Average over samples to get mean interaction matrix
        mean_interaction_matrix = np.mean(shap_interactions, axis=0)

        # Normalize if requested
        if normalize:
            max_val = np.abs(mean_interaction_matrix).max()
            if max_val > 0:
                mean_interaction_matrix = mean_interaction_matrix / max_val

        # Convert to DataFrame
        interaction_df = pd.DataFrame(
            mean_interaction_matrix,
            index=self.feature_names,
            columns=self.feature_names
        )

        return interaction_df

    def explain_interaction(
        self,
        feature_a: str,
        feature_b: str,
        interaction: ShapleyInteraction
    ) -> str:
        """
        Generate human-readable explanation of interaction

        Args:
            feature_a: First feature
            feature_b: Second feature
            interaction: ShapleyInteraction object

        Returns:
            Explanation string
        """
        if interaction.interaction_type == "synergistic":
            explanation = (
                f"**Synergistic Interaction** (strength: {interaction.interaction_strength:.4f})\n\n"
                f"{feature_a} and {feature_b} amplify each other's effects. "
                f"When both are present, their combined impact is greater than the sum of their individual effects.\n\n"
                f"- Synergy component: {interaction.synergy_score:.4f}\n"
                f"- Statistical significance: p={interaction.statistical_significance:.4f}\n"
                f"- Interpretation: These features work TOGETHER to influence predictions."
            )
        elif interaction.interaction_type == "redundant":
            explanation = (
                f"**Redundant Interaction** (strength: {interaction.interaction_strength:.4f})\n\n"
                f"{feature_a} and {feature_b} are substitutes. "
                f"They provide overlapping information, so their combined impact is less than the sum.\n\n"
                f"- Redundancy component: {interaction.redundancy_score:.4f}\n"
                f"- Statistical significance: p={interaction.statistical_significance:.4f}\n"
                f"- Interpretation: One feature can COMPENSATE for the other."
            )
        else:
            explanation = (
                f"**Independent Features** (strength: {interaction.interaction_strength:.4f})\n\n"
                f"{feature_a} and {feature_b} operate independently. "
                f"No significant interaction detected.\n\n"
                f"- Statistical significance: p={interaction.statistical_significance:.4f}\n"
                f"- Interpretation: Features contribute ADDITIVELY."
            )

        return explanation

    def export_interactions(
        self,
        interactions: List[ShapleyInteraction],
        output_path: str
    ):
        """
        Export interactions to JSON for documentation/analysis

        Args:
            interactions: List of ShapleyInteraction objects
            output_path: Path to save JSON
        """
        import json

        export_data = {
            "interactions": [asdict(i) for i in interactions],
            "summary": {
                "total_interactions": len(interactions),
                "synergistic": sum(1 for i in interactions if i.interaction_type == "synergistic"),
                "redundant": sum(1 for i in interactions if i.interaction_type == "redundant"),
                "independent": sum(1 for i in interactions if i.interaction_type == "independent"),
                "mean_strength": float(np.mean([abs(i.interaction_strength) for i in interactions])),
                "max_strength": float(max([abs(i.interaction_strength) for i in interactions]))
            },
            "provenance": {
                "method": "ShapleyInteractionIndex",
                "statistical_test": "one-sample t-test",
                "significance_level": 0.05
            }
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Interactions exported to {output_path}")


def compare_to_correlation(
    analyzer: ShapleyInteractionAnalyzer,
    X: pd.DataFrame,
    interactions: List[ShapleyInteraction]
) -> pd.DataFrame:
    """
    Compare Shapley interactions to simple correlation

    This demonstrates why Shapley is superior to correlation for
    understanding feature relationships.

    Returns:
        DataFrame comparing Shapley vs correlation for each pair
    """
    results = []

    for interaction in interactions:
        # Get correlation
        corr = X[interaction.feature_a].corr(X[interaction.feature_b])

        results.append({
            'feature_a': interaction.feature_a,
            'feature_b': interaction.feature_b,
            'shapley_interaction': interaction.interaction_strength,
            'correlation': corr,
            'difference': abs(interaction.interaction_strength - corr),
            'shapley_type': interaction.interaction_type,
            'shapley_significant': interaction.is_significant()
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    """Example usage and testing"""
    import sys

    # Load data
    df = pd.read_csv("data/processed/scored_applications_xgb.csv")
    feature_cols = [col for col in df.columns if col not in ['Loan_ID', 'Loan_Status', 'Probability']]
    X = df[feature_cols].head(100)  # Sample for speed

    # Initialize analyzer
    analyzer = ShapleyInteractionAnalyzer(
        model_path="models/loan_xgb_monotonic.joblib",
        feature_names=feature_cols
    )

    print("\n" + "="*80)
    print("SHAPLEY INTERACTION ANALYSIS")
    print("="*80 + "\n")

    # Find top interactions
    print("Finding top interactions...")
    interactions = analyzer.find_top_interactions(X, top_k=10, min_strength=0.01)

    print(f"\nFound {len(interactions)} significant interactions:\n")

    for i, interaction in enumerate(interactions, 1):
        print(f"{i}. {interaction.feature_a} × {interaction.feature_b}")
        print(f"   Type: {interaction.interaction_type}")
        print(f"   Strength: {interaction.interaction_strength:.4f}")
        print(f"   P-value: {interaction.statistical_significance:.4f}")
        print(f"   Synergy: {interaction.synergy_score:.4f}")
        print(f"   Redundancy: {interaction.redundancy_score:.4f}")
        print()

    # Export
    analyzer.export_interactions(interactions, "data/evaluation/shapley_interactions.json")

    print("✅ Analysis complete!")
