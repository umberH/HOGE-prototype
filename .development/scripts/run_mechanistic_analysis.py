"""
Comprehensive Mechanistic Interpretability Analysis Script
===========================================================

Run complete mechanistic analysis pipeline:
1. Shapley Interaction Analysis
2. Circuit Discovery
3. Ablation Validation
4. Fidelity Metrics Evaluation

Usage:
    python scripts/run_mechanistic_analysis.py LP001006
    python scripts/run_mechanistic_analysis.py --all
    python scripts/run_mechanistic_analysis.py --application LP001006 --skip-ablation
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.explainability.shapley_interactions import ShapleyInteractionAnalyzer
from src.explainability.circuit_discovery import CircuitDiscovery
from src.explainability.ablation_validation import AblationValidator
from src.explainability.fidelity_metrics import MechanisticFidelityEvaluator


class MechanisticAnalysisPipeline:
    """
    Complete mechanistic interpretability analysis pipeline

    This is the main scientific contribution - a comprehensive framework
    for rigorously analyzing model mechanisms.
    """

    def __init__(
        self,
        model_path: str = "models/loan_xgb_monotonic.joblib",
        data_path: str = "data/processed/scored_applications_xgb.csv",
        output_dir: str = "data/evaluation/mechanistic"
    ):
        """
        Initialize analysis pipeline

        Args:
            model_path: Path to trained model
            data_path: Path to processed data
            output_dir: Output directory for results
        """
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir

        # Load data
        self.df = pd.read_csv(data_path)
        self.feature_cols = [col for col in self.df.columns
                            if col not in ['Loan_ID', 'Loan_Status', 'Probability',
                                          'Predicted_Status', 'Approval_Probability']]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Results storage
        self.results = {}

    def run_shapley_interactions(
        self,
        X: pd.DataFrame,
        top_k: int = 10
    ) -> dict:
        """
        Run Shapley interaction analysis

        Args:
            X: Input features
            top_k: Number of top interactions to find

        Returns:
            Dictionary with interaction results
        """
        print("\n" + "="*80)
        print("1. SHAPLEY INTERACTION ANALYSIS")
        print("="*80)

        analyzer = ShapleyInteractionAnalyzer(
            model_path=self.model_path,
            feature_names=self.feature_cols
        )

        print(f"Finding top {top_k} interactions...")
        interactions = analyzer.find_top_interactions(
            X, top_k=top_k, min_strength=0.01, alpha=0.05
        )

        print(f"\nFound {len(interactions)} significant interactions:")
        for i, interaction in enumerate(interactions[:5], 1):
            print(f"{i}. {interaction.feature_a} × {interaction.feature_b}")
            print(f"   Type: {interaction.interaction_type}, Strength: {interaction.interaction_strength:.4f}")

        # Export
        output_file = os.path.join(self.output_dir, "shapley_interactions.json")
        analyzer.export_interactions(interactions, output_file)

        return {
            'interactions': interactions,
            'analyzer': analyzer,
            'output_file': output_file
        }

    def run_circuit_discovery(
        self,
        X: pd.DataFrame,
        min_frequency: float = 0.05,
        max_complexity: int = 4
    ) -> dict:
        """
        Run circuit discovery

        Args:
            X: Input features
            min_frequency: Minimum circuit frequency
            max_complexity: Maximum features per circuit

        Returns:
            Dictionary with circuit results
        """
        print("\n" + "="*80)
        print("2. CIRCUIT DISCOVERY")
        print("="*80)

        discoverer = CircuitDiscovery(
            model_path=self.model_path,
            feature_names=self.feature_cols
        )

        print(f"Discovering circuits (min_freq={min_frequency}, max_complexity={max_complexity})...")
        circuits = discoverer.discover_circuits(
            min_frequency=min_frequency,
            min_importance=0.01,
            max_complexity=max_complexity
        )

        print(f"\nDiscovered {len(circuits)} circuits")
        print("\nTop 5 circuits:")
        for i, circuit in enumerate(circuits[:5], 1):
            print(f"{i}. {circuit.circuit_id[:8]}... - Features: {', '.join(circuit.features_involved[:3])}")
            print(f"   Frequency: {circuit.frequency:.2%}, Impact: {circuit.impact_score():.4f}")

        # Compute coverage
        activations = discoverer.find_circuit_activations(X, circuits)
        coverage = discoverer.compute_circuit_coverage(X, top_k=10)

        print(f"\nTop-10 Circuit Coverage: {coverage['coverage']:.2%}")

        # Export
        output_file = os.path.join(self.output_dir, "circuits.json")
        discoverer.export_circuits(output_file)

        return {
            'circuits': circuits,
            'activations': activations,
            'coverage': coverage,
            'discoverer': discoverer,
            'output_file': output_file
        }

    def run_ablation_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_k_features: int = 10,
        test_interactions: bool = True
    ) -> dict:
        """
        Run ablation validation

        Args:
            X: Input features
            y: Labels
            top_k_features: Number of top features to validate
            test_interactions: Whether to test interactions

        Returns:
            Dictionary with ablation results
        """
        print("\n" + "="*80)
        print("3. ABLATION VALIDATION")
        print("="*80)

        validator = AblationValidator(
            model_path=self.model_path,
            feature_names=self.feature_cols,
            metric='accuracy'
        )

        print(f"Validating top {top_k_features} features via ablation...")
        feature_results = validator.validate_all_features(X, y, top_k=top_k_features)

        print("\nFeature Ablation Results:")
        for i, result in enumerate(feature_results[:5], 1):
            validated = "✓" if result.validated else "✗"
            print(f"{i}. {result.ablated_component}: drop={result.performance_drop:.4f} {validated}")

        # Test interactions
        interaction_results = []
        if test_interactions:
            print("\nTesting feature interactions...")
            # Test top 3 pairs
            pairs = [
                (feature_results[0].ablated_component, feature_results[1].ablated_component),
                (feature_results[0].ablated_component, feature_results[2].ablated_component),
            ]

            interaction_results = validator.validate_interactions(pairs, X, y)

            for result in interaction_results:
                print(f"  {result.feature_a} × {result.feature_b}: {result.interaction_type}")

        # Export
        output_file = os.path.join(self.output_dir, "ablation_results.json")
        validator.export_results(feature_results, output_file)

        return {
            'feature_results': feature_results,
            'interaction_results': interaction_results,
            'validator': validator,
            'output_file': output_file
        }

    def run_fidelity_evaluation(
        self,
        X: pd.DataFrame,
        circuits_result: dict = None,
        ablation_result: dict = None
    ) -> dict:
        """
        Run fidelity metrics evaluation

        Args:
            X: Input features
            circuits_result: Results from circuit discovery
            ablation_result: Results from ablation validation

        Returns:
            Dictionary with fidelity metrics
        """
        print("\n" + "="*80)
        print("4. FIDELITY METRICS EVALUATION")
        print("="*80)

        evaluator = MechanisticFidelityEvaluator(
            model_path=self.model_path,
            feature_names=self.feature_cols
        )

        # Full evaluation
        print("Computing fidelity metrics...")
        circuits = circuits_result['circuits'] if circuits_result else None
        activations = circuits_result['activations'] if circuits_result else None
        ablation_results = ablation_result['feature_results'] if ablation_result else None

        metrics = evaluator.evaluate_full_fidelity(
            X,
            circuits=circuits,
            circuit_activations=activations,
            ablation_results=ablation_results
        )

        print("\nFidelity Metrics:")
        print(f"  Circuit Coverage: {metrics.circuit_coverage:.2%}")
        print(f"  Mechanistic Faithfulness: {metrics.mechanistic_faithfulness:.4f}")
        print(f"  R²: {metrics.r_squared:.4f}")
        print(f"  MAE: {metrics.mean_absolute_error:.4f}")
        print(f"  Overall Quality: {metrics.overall_quality_score:.4f}")

        # Pareto frontier
        pareto_frontier = None
        if circuits and activations:
            print("\nComputing complexity-fidelity tradeoff...")
            pareto_frontier = evaluator.compute_pareto_frontier(
                X, circuits, activations, max_circuits=min(20, len(circuits))
            )

        # Export
        output_file = os.path.join(self.output_dir, "fidelity_metrics.json")
        evaluator.export_metrics(metrics, pareto_frontier, output_file)

        return {
            'metrics': metrics,
            'pareto_frontier': pareto_frontier,
            'evaluator': evaluator,
            'output_file': output_file
        }

    def run_full_analysis(
        self,
        application_id: str = None,
        sample_size: int = 200,
        skip_ablation: bool = False
    ):
        """
        Run complete mechanistic analysis pipeline

        Args:
            application_id: Specific application to analyze (None = sample)
            sample_size: Number of samples to use
            skip_ablation: Skip ablation (faster but less rigorous)
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MECHANISTIC INTERPRETABILITY ANALYSIS")
        print("="*80)

        # Prepare data
        if application_id:
            data_subset = self.df[self.df['Loan_ID'] == application_id]
            if data_subset.empty:
                print(f"Error: Application {application_id} not found")
                return None
        else:
            data_subset = self.df.sample(min(sample_size, len(self.df)))

        X = data_subset[self.feature_cols]
        # Use Predicted_Status (already 0/1) or convert if needed
        if 'Loan_Status' in data_subset.columns:
            y = data_subset['Loan_Status'].map({'Y': 1, 'N': 0})
        else:
            y = data_subset['Predicted_Status']

        print(f"\nAnalyzing {len(X)} samples...")

        # Run analyses
        self.results['shapley'] = self.run_shapley_interactions(X, top_k=10)
        self.results['circuits'] = self.run_circuit_discovery(X, min_frequency=0.05)

        if not skip_ablation:
            self.results['ablation'] = self.run_ablation_validation(X, y, top_k_features=10)
        else:
            print("\nSkipping ablation validation (--skip-ablation)")
            self.results['ablation'] = None

        self.results['fidelity'] = self.run_fidelity_evaluation(
            X,
            circuits_result=self.results['circuits'],
            ablation_result=self.results['ablation']
        )

        # Generate summary report
        self._generate_summary_report()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}/")

        return self.results

    def _generate_summary_report(self):
        """
        Generate comprehensive summary report

        This combines all analyses into a single publishable report.
        """
        summary = {
            'analysis_type': 'comprehensive_mechanistic_interpretability',
            'model': self.model_path,
            'data_samples': len(self.df),
            'num_features': len(self.feature_cols),
            'shapley_interactions': {
                'total_interactions': len(self.results['shapley']['interactions']),
                'synergistic': sum(1 for i in self.results['shapley']['interactions']
                                  if i.interaction_type == 'synergistic'),
                'redundant': sum(1 for i in self.results['shapley']['interactions']
                                if i.interaction_type == 'redundant')
            },
            'circuits': {
                'total_discovered': len(self.results['circuits']['circuits']),
                'coverage': self.results['circuits']['coverage']['coverage'],
                'avg_complexity': np.mean([c.complexity() for c in self.results['circuits']['circuits']])
            },
            'fidelity': {
                'overall_quality': self.results['fidelity']['metrics'].overall_quality_score,
                'faithfulness': self.results['fidelity']['metrics'].mechanistic_faithfulness,
                'coverage': self.results['fidelity']['metrics'].circuit_coverage
            }
        }

        if self.results['ablation']:
            summary['ablation'] = {
                'features_validated': len(self.results['ablation']['feature_results']),
                'features_confirmed': sum(1 for r in self.results['ablation']['feature_results']
                                         if r.validated)
            }

        # Save summary
        output_file = os.path.join(self.output_dir, "analysis_summary.json")
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary report saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive mechanistic interpretability analysis"
    )

    parser.add_argument(
        '--application',
        type=str,
        default=None,
        help="Specific application ID to analyze"
    )

    parser.add_argument(
        '--sample-size',
        type=int,
        default=200,
        help="Number of samples to use (if not analyzing specific application)"
    )

    parser.add_argument(
        '--skip-ablation',
        action='store_true',
        help="Skip ablation validation (faster but less rigorous)"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default="data/evaluation/mechanistic",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = MechanisticAnalysisPipeline(output_dir=args.output_dir)

    results = pipeline.run_full_analysis(
        application_id=args.application,
        sample_size=args.sample_size,
        skip_ablation=args.skip_ablation
    )

    if results:
        print("\nAnalysis complete! Check output directory for results.")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
