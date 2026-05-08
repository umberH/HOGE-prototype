"""
Enhanced Provenance Tracking for HOGE Framework
================================================

Comprehensive metadata capture for full auditability and reproducibility.

Usage:
    from src.provenance.enhanced_provenance import ProvenanceTracker

    tracker = ProvenanceTracker()
    tracker.add_model_provenance(model_path, training_metadata)
    tracker.add_llm_provenance(llm_response, prompt_config)
    provenance = tracker.export_provenance()
"""

import datetime
import hashlib
import json
import platform
import sys
import subprocess
from typing import Dict, Any, List, Optional
try:
    import pkg_resources
except ImportError:
    pkg_resources = None


class ProvenanceTracker:
    """Comprehensive provenance tracking for HOGE framework"""

    def __init__(self):
        self.provenance_data = {
            "hoge_version": "1.0.0",
            "provenance_schema_version": "2.0",
            "framework_name": "HOGE",
            "framework_description": "Human-Centric Ontology-Grounded Explanation Framework"
        }

    def add_model_provenance(self, model_metadata: Dict) -> None:
        """
        Add model training provenance

        Args:
            model_metadata: Dictionary containing:
                - model_path: Path to saved model
                - model_type: Type of model (e.g., "XGBoost")
                - training_samples: Number of training samples
                - test_samples: Number of test samples
                - hyperparameters: Model hyperparameters
                - performance_metrics: Model performance metrics
                - monotonic_constraints: List of monotonic constraints
        """
        self.provenance_data["model_provenance"] = {
            "model_id": f"xgb_monotonic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **model_metadata
        }

    def add_shap_provenance(self, shap_metadata: Dict) -> None:
        """
        Add SHAP computation provenance

        Args:
            shap_metadata: Dictionary containing:
                - shap_version: SHAP library version
                - explainer_type: Type of explainer used
                - num_features_explained: Number of features
                - num_samples_explained: Number of samples
                - base_value: Expected value E[f(X)]
                - computation_duration_ms: Time taken
        """
        self.provenance_data["shap_provenance"] = {
            "computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **shap_metadata
        }

    def add_kg_provenance(self, kg_metadata: Dict) -> None:
        """
        Add knowledge graph loading provenance

        Args:
            kg_metadata: Dictionary containing:
                - kg_loader_version: Version of KG loader
                - neo4j_version: Neo4j version
                - database_name: Database name
                - node_counts: Dictionary of node type counts
                - relationship_counts: Dictionary of relationship counts
                - total_nodes: Total number of nodes
                - total_relationships: Total number of relationships
        """
        self.provenance_data["kg_provenance"] = {
            "loaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **kg_metadata
        }

    def add_llm_provenance(self, llm_response: Any = None, prompt_config: Dict = None) -> None:
        """
        Add LLM explanation generation provenance

        Args:
            llm_response: OpenAI API response object
            prompt_config: Dictionary containing:
                - model: Model name (e.g., "gpt-4o")
                - temperature: Temperature setting
                - max_tokens: Max tokens setting
                - prompt_template_version: Version of prompt template
                - grounding_rules_applied: List of grounding rules
        """
        llm_prov = {
            "provider": "OpenAI",
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        if prompt_config:
            llm_prov.update(prompt_config)

        if llm_response and hasattr(llm_response, 'usage'):
            llm_prov.update({
                "prompt_tokens": llm_response.usage.prompt_tokens,
                "completion_tokens": llm_response.usage.completion_tokens,
                "total_tokens": llm_response.usage.total_tokens
            })

        if llm_response and hasattr(llm_response, 'choices'):
            llm_prov["finish_reason"] = llm_response.choices[0].finish_reason

        self.provenance_data["llm_provenance"] = llm_prov

    def add_explanation_provenance(self, explanation: Dict, context: Dict, audience: str = "technical") -> None:
        """
        Add explanation output provenance

        Args:
            explanation: Generated explanation dictionary
            context: Context dictionary with application data
            audience: Target audience
        """
        self.provenance_data["explanation_provenance"] = {
            "explanation_id": f"exp_{context.get('application_id', 'unknown')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "application_id": context.get('application_id'),
            "target_audience": audience,
            "concept_mode_enabled": explanation.get('used_concepts', False),
            "num_concepts_used": len(explanation.get('business_concepts', [])) if explanation.get('used_concepts') else 0,
            "num_features_total": len(context.get('shap_details', [])),
            "num_features_in_narrative": len(explanation.get('features_used', [])),
            "num_evidence_items": len(context.get('evidence_bundle', [])),
            "evidence_types": self._count_evidence_types(context.get('evidence_bundle', [])),
            "num_positive_drivers": len(explanation.get('positive_drivers', [])),
            "num_negative_drivers": len(explanation.get('negative_drivers', [])),
            "num_policy_violations": len(explanation.get('policy_violations', [])),
            "has_counterfactual": bool(explanation.get('counterfactual_what_if')),
            "has_recommendation": bool(explanation.get('recommendation')),
            "narrative_length_chars": len(explanation.get('narrative', '')),
            "narrative_length_words": len(explanation.get('narrative', '').split()),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def add_evaluation_provenance(self, eval_metrics: Dict) -> None:
        """
        Add evaluation metrics provenance

        Args:
            eval_metrics: Dictionary containing evaluation metrics:
                - faithfulness_score
                - semantic_fidelity
                - evidence_coverage
                - hallucination_rate
                - precision_at_k
        """
        self.provenance_data["evaluation_provenance"] = {
            **eval_metrics,
            "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def add_system_environment(self) -> None:
        """Capture system environment for reproducibility"""
        # Get Git info
        git_info = self._get_git_info()

        # Get installed packages (if pkg_resources available)
        dependencies = {}
        if pkg_resources:
            try:
                installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
                dependencies = {
                    "xgboost": installed.get("xgboost", "unknown"),
                    "shap": installed.get("shap", "unknown"),
                    "neo4j": installed.get("neo4j", "unknown"),
                    "openai": installed.get("openai", "unknown"),
                    "pandas": installed.get("pandas", "unknown"),
                    "numpy": installed.get("numpy", "unknown"),
                    "scikit-learn": installed.get("scikit-learn", "unknown"),
                    "streamlit": installed.get("streamlit", "unknown")
                }
            except:
                dependencies = {"error": "Could not retrieve package versions"}

        self.provenance_data["system_environment"] = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "node": platform.node(),
            "dependencies": dependencies,
            **git_info,
            "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def add_audit_trail(self, user_id: str = "system", session_id: str = None,
                       action: str = "", metadata: Dict = None) -> None:
        """
        Add audit trail entry

        Args:
            user_id: User identifier
            session_id: Session identifier
            action: Action being performed
            metadata: Additional metadata
        """
        if "audit_trail" not in self.provenance_data:
            self.provenance_data["audit_trail"] = {
                "user_id": user_id,
                "session_id": session_id or f"sess_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "access_log": []
            }

        self.provenance_data["audit_trail"]["access_log"].append({
            "action": action,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "metadata": metadata or {}
        })

    def add_visualization_metadata(self, viz_id: str, viz_config: Dict) -> None:
        """
        Add visualization provenance

        Args:
            viz_id: Unique visualization identifier
            viz_config: Dictionary containing:
                - viz_type: Type of visualization
                - viz_library: Library used
                - data_source: Source of data
                - num_data_points: Number of data points
        """
        if "visualizations" not in self.provenance_data:
            self.provenance_data["visualizations"] = []

        self.provenance_data["visualizations"].append({
            "viz_id": viz_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **viz_config
        })

    def add_mechanistic_metadata(self, mech_metadata: Dict) -> None:
        """
        Add mechanistic interpretability analysis metadata

        Args:
            mech_metadata: Dictionary containing:
                - num_trees_analyzed: Number of trees
                - num_interactions_detected: Number of interactions
                - analysis_duration_ms: Time taken
                - tree_depth_stats: Statistics on tree depths
        """
        self.provenance_data["mechanistic_analysis"] = {
            "analyzed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **mech_metadata
        }

    def _count_evidence_types(self, evidence_bundle: List[Dict]) -> Dict[str, int]:
        """Count evidence items by type"""
        type_counts = {}
        for item in evidence_bundle:
            claim_type = item.get('claim_type', 'unknown')
            type_counts[claim_type] = type_counts.get(claim_type, 0) + 1
        return type_counts

    def _get_git_info(self) -> Dict:
        """Get Git repository information"""
        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()

            git_dirty = subprocess.call(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                stderr=subprocess.DEVNULL
            ) != 0

            return {
                "code_version": f"git:{git_commit[:7]}",
                "git_branch": git_branch,
                "git_commit_hash": git_commit,
                "git_commit_short": git_commit[:7],
                "uncommitted_changes": git_dirty
            }
        except:
            return {
                "code_version": "unknown",
                "git_branch": "unknown",
                "git_commit_hash": "unknown",
                "uncommitted_changes": False
            }

    def compute_data_hash(self, data: Any) -> str:
        """
        Compute SHA256 hash of data for reproducibility

        Args:
            data: Data to hash (will be JSON-serialized)

        Returns:
            SHA256 hash string
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(data_str.encode()).hexdigest()[:16]}"

    def export_provenance(self, output_path: str = None) -> Dict:
        """
        Export full provenance record

        Args:
            output_path: Optional path to save JSON file

        Returns:
            Complete provenance dictionary
        """
        provenance_export = {
            "provenance_id": f"prov_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "export_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **self.provenance_data
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(provenance_export, f, indent=2, default=str)
            print(f"Provenance exported to: {output_path}")

        return provenance_export

    def get_summary(self) -> str:
        """Get human-readable provenance summary"""
        summary = []
        summary.append("=" * 70)
        summary.append("HOGE FRAMEWORK - PROVENANCE SUMMARY")
        summary.append("=" * 70)

        if "model_provenance" in self.provenance_data:
            mp = self.provenance_data["model_provenance"]
            summary.append(f"\n📊 MODEL")
            summary.append(f"   ID: {mp.get('model_id', 'N/A')}")
            summary.append(f"   Type: {mp.get('model_type', 'N/A')}")
            perf = mp.get('performance_metrics', {})
            if perf:
                summary.append(f"   ROC-AUC: {perf.get('roc_auc', 'N/A')}")
                summary.append(f"   Accuracy: {perf.get('accuracy', 'N/A')}")

        if "shap_provenance" in self.provenance_data:
            sp = self.provenance_data["shap_provenance"]
            summary.append(f"\n📈 SHAP ANALYSIS")
            summary.append(f"   Features Explained: {sp.get('num_features_explained', 'N/A')}")
            summary.append(f"   Samples Explained: {sp.get('num_samples_explained', 'N/A')}")

        if "kg_provenance" in self.provenance_data:
            kp = self.provenance_data["kg_provenance"]
            summary.append(f"\n🕸️  KNOWLEDGE GRAPH")
            summary.append(f"   Total Nodes: {kp.get('total_nodes', 'N/A'):,}")
            summary.append(f"   Total Relationships: {kp.get('total_relationships', 'N/A'):,}")

        if "llm_provenance" in self.provenance_data:
            lp = self.provenance_data["llm_provenance"]
            summary.append(f"\n🤖 LLM GENERATION")
            summary.append(f"   Model: {lp.get('model', 'N/A')}")
            summary.append(f"   Total Tokens: {lp.get('total_tokens', 'N/A')}")
            summary.append(f"   Temperature: {lp.get('temperature', 'N/A')}")

        if "explanation_provenance" in self.provenance_data:
            ep = self.provenance_data["explanation_provenance"]
            summary.append(f"\n📝 EXPLANATION")
            summary.append(f"   ID: {ep.get('explanation_id', 'N/A')}")
            summary.append(f"   Application: {ep.get('application_id', 'N/A')}")
            summary.append(f"   Audience: {ep.get('target_audience', 'N/A')}")
            summary.append(f"   Concept Mode: {ep.get('concept_mode_enabled', False)}")
            summary.append(f"   Evidence Items: {ep.get('num_evidence_items', 0)}")
            summary.append(f"   Narrative Length: {ep.get('narrative_length_words', 0)} words")

        if "evaluation_provenance" in self.provenance_data:
            evp = self.provenance_data["evaluation_provenance"]
            summary.append(f"\n✅ EVALUATION")
            summary.append(f"   Faithfulness: {evp.get('faithfulness_score', 'N/A')}")
            summary.append(f"   Semantic Fidelity: {evp.get('semantic_fidelity', 'N/A')}")
            summary.append(f"   Evidence Coverage: {evp.get('evidence_coverage', 'N/A')}")
            summary.append(f"   Hallucination Rate: {evp.get('hallucination_rate', 'N/A')}")

        if "system_environment" in self.provenance_data:
            se = self.provenance_data["system_environment"]
            summary.append(f"\n🖥️  SYSTEM")
            summary.append(f"   Python: {se.get('python_version', 'N/A')}")
            summary.append(f"   Platform: {se.get('platform', 'N/A')}")
            summary.append(f"   Git Branch: {se.get('git_branch', 'N/A')}")
            summary.append(f"   Git Commit: {se.get('git_commit_short', 'N/A')}")

        summary.append("\n" + "=" * 70)

        return "\n".join(summary)

    def get_compact_provenance(self) -> Dict:
        """Get compact provenance for embedding in explanation JSON"""
        compact = {
            "hoge_version": self.provenance_data.get("hoge_version"),
            "schema_version": self.provenance_data.get("provenance_schema_version")
        }

        if "model_provenance" in self.provenance_data:
            compact["model_name"] = self.provenance_data["model_provenance"].get("model_id")
            compact["model_performance"] = self.provenance_data["model_provenance"].get("performance_metrics", {}).get("roc_auc")

        if "shap_provenance" in self.provenance_data:
            compact["xai_method"] = "TreeSHAP"

        if "kg_provenance" in self.provenance_data:
            compact["kg_loader_version"] = self.provenance_data["kg_provenance"].get("kg_loader_version")
            compact["kg_total_nodes"] = self.provenance_data["kg_provenance"].get("total_nodes")

        if "llm_provenance" in self.provenance_data:
            compact["llm_model"] = self.provenance_data["llm_provenance"].get("model")
            compact["llm_tokens"] = self.provenance_data["llm_provenance"].get("total_tokens")

        if "explanation_provenance" in self.provenance_data:
            compact["explanation_id"] = self.provenance_data["explanation_provenance"].get("explanation_id")
            compact["target_audience"] = self.provenance_data["explanation_provenance"].get("target_audience")
            compact["evidence_items"] = self.provenance_data["explanation_provenance"].get("num_evidence_items")

        compact["retrieval_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return compact


# Usage example
if __name__ == "__main__":
    print("Enhanced Provenance Tracker - Demo\n")

    tracker = ProvenanceTracker()

    # Add system environment
    tracker.add_system_environment()

    # Add model provenance
    tracker.add_model_provenance({
        "model_path": "models/loan_xgb_monotonic.joblib",
        "model_type": "XGBoost",
        "training_samples": 400,
        "test_samples": 100,
        "hyperparameters": {
            "n_estimators": 350,
            "max_depth": 4,
            "learning_rate": 0.1
        },
        "performance_metrics": {
            "roc_auc": 0.839,
            "accuracy": 0.80
        }
    })

    # Add LLM provenance
    tracker.add_llm_provenance(prompt_config={
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 1000
    })

    # Print summary
    print(tracker.get_summary())

    # Export
    print("\n📦 Exporting full provenance...")
    prov = tracker.export_provenance("provenance_demo.json")
    print(f"✅ Exported {len(prov)} top-level keys")
