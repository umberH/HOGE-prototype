"""
Test Script for Explanation Logging and Drift Detection

Demonstrates the comprehensive logging system by:
1. Generating explanations for multiple applications
2. Testing different audiences and models
3. Running drift detection analysis
4. Generating comparison reports
5. Creating visualizations
"""

import sys
import json
from src.explainability.llm_explainer import get_application_explanation_data, call_llm_for_explanation
from src.explainability.explanation_logger import ExplanationLogger
from src.explainability.explanation_analyzer import ExplanationAnalyzer
from src.explainability.explanation_dashboard import ExplanationDashboard


def test_logging_for_single_application(application_id: str):
    """
    Test logging for a single application across multiple audiences
    """
    print(f"\n{'='*80}")
    print(f"Testing Explanation Logging for Application: {application_id}")
    print(f"{'='*80}\n")

    # Get context data
    try:
        context = get_application_explanation_data(application_id)
        print(f"✓ Retrieved context for {application_id}")
    except Exception as e:
        print(f"✗ Error retrieving context: {e}")
        return

    audiences = ["technical", "non_technical", "executive", "business"]

    for audience in audiences:
        print(f"\n--- Generating {audience} explanation ---")

        try:
            # Generate explanation (logging is enabled by default)
            result = call_llm_for_explanation(
                context=context,
                audience=audience,
                enable_logging=True
            )

            log_id = result.get("log_id")
            summary = result.get("summary", "")

            print(f"✓ Generated explanation")
            print(f"  Log ID: {log_id}")
            print(f"  Summary: {summary[:100]}...")
            print(f"  Features mentioned: {len(result.get('features_used', []))}")
            print(f"  Used concepts: {result.get('used_concepts', False)}")

        except Exception as e:
            print(f"✗ Error generating {audience} explanation: {e}")


def test_drift_detection(application_id: str):
    """
    Test drift detection for an application
    """
    print(f"\n{'='*80}")
    print(f"Testing Drift Detection for Application: {application_id}")
    print(f"{'='*80}\n")

    analyzer = ExplanationAnalyzer()

    # Temporal drift analysis
    print("--- Temporal Drift Analysis ---")
    drift = analyzer.detect_explanation_drift(
        application_id=application_id,
        audience="technical",
        window_size=10
    )

    print(f"Status: {drift['status']}")
    if drift["status"] == "success":
        print(f"Number of explanations analyzed: {drift['num_explanations']}")
        print(f"\nDrift Metrics:")
        for metric, values in drift["drift_metrics"].items():
            print(f"  {metric}:")
            print(f"    Mean: {values['mean']:.3f}")
            print(f"    Std: {values['std']:.3f}")
            print(f"    Coefficient of Variation: {values['coefficient_of_variation']:.3f}")

        print(f"\nAlerts:")
        for alert in drift.get("alerts", []):
            print(f"  ⚠️  {alert}")

    # Consistency check
    print("\n--- Consistency Analysis ---")
    consistency = analyzer.detect_consistency_issues(
        application_id=application_id,
        threshold=0.3
    )

    print(f"Status: {consistency['status']}")
    if consistency["status"] == "success":
        print(f"Total explanations: {consistency['total_explanations']}")
        print(f"Issues found: {consistency['issues_found']}")
        print(f"Verdict: {consistency['verdict']}")

        if consistency['issues']:
            print(f"\nIssues:")
            for issue in consistency['issues']:
                print(f"  - {issue['message']}")

    analyzer.close()


def test_audience_comparison(application_id: str):
    """
    Test audience comparison functionality
    """
    print(f"\n{'='*80}")
    print(f"Testing Audience Comparison for Application: {application_id}")
    print(f"{'='*80}\n")

    logger = ExplanationLogger()

    comparison = logger.compare_audiences(application_id=application_id)

    print(f"Application: {comparison['application_id']}")
    print(f"Audiences compared: {', '.join(comparison['audiences_compared'])}\n")

    for audience, metrics in comparison["comparisons"].items():
        print(f"--- {audience.upper()} ---")
        print(f"  Narrative length: {metrics['narrative_length']} chars")
        print(f"  Technical terms: {metrics['technical_terms_count']}")
        print(f"  Feature coverage: {metrics['feature_coverage_ratio']:.2%}")
        print(f"  Citation quality: {metrics['citation_quality_score']:.3f}")
        print(f"  Used concepts: {metrics['used_concepts']}")
        print(f"  Summary: {metrics['summary'][:100]}...")
        print()

    print("Insights:")
    for insight in comparison["insights"]:
        print(f"  • {insight}")

    logger.close()


def test_quality_report():
    """
    Test comprehensive quality report generation
    """
    print(f"\n{'='*80}")
    print(f"Testing Quality Report Generation")
    print(f"{'='*80}\n")

    analyzer = ExplanationAnalyzer()

    report = analyzer.generate_quality_report(days=30)

    print(f"Report Date: {report['report_date']}")
    print(f"Period: {report['period_days']} days")
    print(f"Quality Grade: {report['quality_grade']}\n")

    print("--- Overview ---")
    overview = report['overview']
    print(f"  Total explanations: {overview['total_explanations']}")
    print(f"  Avg citation quality: {overview['avg_citation_quality']:.3f}")
    print(f"  Avg feature coverage: {overview['avg_feature_coverage']:.3f}")
    print(f"  Avg generation time: {overview['avg_generation_time_ms']:.1f} ms")
    print(f"  Audiences: {', '.join(overview['audiences_used'])}")
    print(f"  Models: {', '.join(overview['models_used'])}")

    print("\n--- Top Features ---")
    for feature in report['feature_trends']['top_features'][:5]:
        print(f"  {feature['feature']}: {feature['mentions']} mentions "
              f"(quality: {feature['avg_citation_quality']:.3f})")

    # Save full report to JSON
    output_file = "data/evaluation/quality_report_latest.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✓ Full report saved to: {output_file}")

    analyzer.close()


def test_visualization_dashboard():
    """
    Test dashboard visualization generation
    """
    print(f"\n{'='*80}")
    print(f"Testing Visualization Dashboard")
    print(f"{'='*80}\n")

    dashboard = ExplanationDashboard()

    # Generate full report with visualizations
    print("Generating comprehensive HTML report with visualizations...")
    report_path = dashboard.generate_full_report(days=30)

    print(f"\n✓ Dashboard report generated: {report_path}")
    print("  Open this file in a web browser to view interactive visualizations")

    dashboard.close()


def main():
    """
    Main test runner
    """
    print("\n" + "="*80)
    print(" EXPLANATION LOGGING & DRIFT DETECTION TEST SUITE")
    print("="*80)

    # Test applications (use actual IDs from your dataset)
    test_applications = ["LP001002", "LP001003", "LP001005"]

    # Test 1: Generate logged explanations for multiple applications and audiences
    print("\n\n### TEST 1: Explanation Generation & Logging ###")
    for app_id in test_applications:
        test_logging_for_single_application(app_id)

    # Test 2: Drift detection
    print("\n\n### TEST 2: Drift Detection ###")
    test_drift_detection(test_applications[0])

    # Test 3: Audience comparison
    print("\n\n### TEST 3: Audience Comparison ###")
    test_audience_comparison(test_applications[0])

    # Test 4: Quality report
    print("\n\n### TEST 4: Quality Report ###")
    test_quality_report()

    # Test 5: Visualization dashboard
    print("\n\n### TEST 5: Visualization Dashboard ###")
    test_visualization_dashboard()

    print("\n\n" + "="*80)
    print(" TEST SUITE COMPLETED")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check Neo4j database for ExplanationLog nodes")
    print("  2. Review JSON logs in data/explanation_logs/")
    print("  3. Open HTML reports in data/explanation_reports/")
    print("  4. Query drift metrics using ExplanationAnalyzer")


if __name__ == "__main__":
    # Allow specifying application IDs via command line
    if len(sys.argv) > 1:
        application_ids = sys.argv[1:]
        print(f"Testing with applications: {application_ids}")
        for app_id in application_ids:
            test_logging_for_single_application(app_id)
        test_drift_detection(application_ids[0])
        test_audience_comparison(application_ids[0])
    else:
        # Run full test suite
        main()
