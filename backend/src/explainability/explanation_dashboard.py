"""
Explanation Drift Visualization Dashboard

Provides interactive visualizations for:
- Temporal drift trends
- Model comparison charts
- Audience analysis
- Quality metrics over time
"""

import os
import json
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from src.explainability.explanation_logger import ExplanationLogger
from src.explainability.explanation_analyzer import ExplanationAnalyzer


class ExplanationDashboard:
    """
    Interactive dashboard for visualizing explanation drift and quality metrics
    """

    def __init__(self):
        self.logger = ExplanationLogger()
        self.analyzer = ExplanationAnalyzer()

    def plot_temporal_drift(
        self,
        days: int = 30,
        audience: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create multi-panel temporal drift visualization

        Args:
            days: Number of days to look back
            audience: Optional filter by audience
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        # Get historical data
        history = self.logger.get_explanation_history(
            audience=audience,
            limit=1000
        )

        if not history:
            print("No explanation data found")
            return None

        # Filter by date range
        cutoff = datetime.now() - timedelta(days=days)
        history = [
            h for h in history
            if datetime.fromisoformat(h["timestamp"].replace('Z', '+00:00')) >= cutoff
        ]

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Narrative Length Over Time',
                'Feature Coverage Over Time',
                'Citation Quality Over Time',
                'Technical Terms Count',
                'Generation Time (ms)',
                'Explanations by Audience'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "pie"}]
            ]
        )

        # Plot 1: Narrative Length
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['narrative_length'],
                mode='lines+markers',
                name='Narrative Length',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Plot 2: Feature Coverage
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['feature_coverage_ratio'],
                mode='lines+markers',
                name='Feature Coverage',
                line=dict(color='green')
            ),
            row=1, col=2
        )

        # Plot 3: Citation Quality
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['citation_quality_score'],
                mode='lines+markers',
                name='Citation Quality',
                line=dict(color='orange')
            ),
            row=2, col=1
        )

        # Add threshold line for citation quality
        fig.add_hline(
            y=0.7, line_dash="dash", line_color="red",
            annotation_text="Minimum Threshold",
            row=2, col=1
        )

        # Plot 4: Technical Terms
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['technical_terms_count'],
                mode='lines+markers',
                name='Technical Terms',
                line=dict(color='purple')
            ),
            row=2, col=2
        )

        # Plot 5: Generation Time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['generation_time_ms'],
                mode='lines+markers',
                name='Generation Time',
                line=dict(color='red')
            ),
            row=3, col=1
        )

        # Plot 6: Audience Distribution (Pie Chart)
        audience_counts = df['audience'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=audience_counts.index,
                values=audience_counts.values,
                name='Audience Distribution'
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f"Explanation Drift Analysis - Last {days} Days",
            showlegend=True,
            height=1200,
            width=1400
        )

        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Characters", row=1, col=1)
        fig.update_yaxes(title_text="Ratio (0-1)", row=1, col=2)
        fig.update_yaxes(title_text="Score (0-1)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_yaxes(title_text="Milliseconds", row=3, col=1)

        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to {save_path}")

        return fig

    def plot_model_comparison(
        self,
        model1: str,
        model2: str,
        audience: str = "technical",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create side-by-side comparison of two models

        Args:
            model1: First model name
            model2: Second model name
            audience: Audience type to compare
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        comparison = self.analyzer.compare_model_versions(
            model1=model1,
            model2=model2,
            audience=audience
        )

        if comparison["status"] != "success":
            print(f"Insufficient data: {comparison}")
            return None

        # Extract metrics
        metrics = list(comparison["comparison"].keys())
        model1_values = [comparison["comparison"][m][f"{model1}_mean"] for m in metrics]
        model2_values = [comparison["comparison"][m][f"{model2}_mean"] for m in metrics]

        # Create grouped bar chart
        fig = go.Figure(data=[
            go.Bar(name=model1, x=metrics, y=model1_values, marker_color='lightblue'),
            go.Bar(name=model2, x=metrics, y=model2_values, marker_color='lightcoral')
        ])

        # Update layout
        fig.update_layout(
            title=f"Model Comparison: {model1} vs {model2} ({audience} audience)",
            xaxis_title="Metric",
            yaxis_title="Value",
            barmode='group',
            height=600,
            width=1000
        )

        # Add recommendation annotation
        fig.add_annotation(
            text=f"Recommendation: {comparison['recommendation']}",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=14, color="green"),
            bgcolor="lightyellow"
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Comparison chart saved to {save_path}")

        return fig

    def plot_audience_heatmap(
        self,
        application_id: str,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap showing metric variations across audiences

        Args:
            application_id: Application to analyze
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        audiences = ["technical", "non_technical", "executive", "business"]

        # Get latest explanation for each audience
        data = []
        for audience in audiences:
            history = self.logger.get_explanation_history(
                application_id=application_id,
                audience=audience,
                limit=1
            )
            if history:
                data.append(history[0])

        if not data:
            print(f"No explanations found for application {application_id}")
            return None

        # Create matrix
        metrics = ['narrative_length', 'feature_coverage_ratio', 'citation_quality_score', 'technical_terms_count']
        matrix = []

        for metric in metrics:
            row = []
            for aud_data in data:
                value = aud_data.get(metric, 0)
                row.append(value if value is not None else 0)
            matrix.append(row)

        # Normalize each row for better visualization
        matrix_normalized = []
        for row in matrix:
            max_val = max(row) if max(row) > 0 else 1
            matrix_normalized.append([v / max_val for v in row])

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix_normalized,
            x=[d['audience'] for d in data],
            y=metrics,
            colorscale='Viridis',
            text=matrix,
            texttemplate='%{text:.1f}',
            textfont={"size": 10}
        ))

        fig.update_layout(
            title=f"Audience Comparison Heatmap - {application_id}",
            xaxis_title="Audience",
            yaxis_title="Metric",
            height=500,
            width=800
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Heatmap saved to {save_path}")

        return fig

    def plot_quality_trends(
        self,
        days: int = 30,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot quality metrics trends with alerts

        Args:
            days: Number of days to look back
            save_path: Optional path to save HTML file

        Returns:
            Plotly figure object
        """
        drift_analysis = self.analyzer.analyze_temporal_drift(days=days)

        if drift_analysis["status"] != "success":
            print(f"No drift data available: {drift_analysis}")
            return None

        trends = drift_analysis["trends"]

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add citation quality trend
        fig.add_trace(
            go.Scatter(
                x=["Historical", "Current"],
                y=[trends["citation_quality"]["historical_avg"],
                   trends["citation_quality"]["current_avg"]],
                name="Citation Quality",
                mode='lines+markers',
                line=dict(color='green', width=3)
            ),
            secondary_y=False
        )

        # Add feature coverage trend
        fig.add_trace(
            go.Scatter(
                x=["Historical", "Current"],
                y=[trends["feature_coverage"]["historical_avg"],
                   trends["feature_coverage"]["current_avg"]],
                name="Feature Coverage",
                mode='lines+markers',
                line=dict(color='blue', width=3)
            ),
            secondary_y=False
        )

        # Add generation time on secondary axis
        fig.add_trace(
            go.Scatter(
                x=["Historical", "Current"],
                y=[trends["generation_time_ms"]["historical_avg"],
                   trends["generation_time_ms"]["current_avg"]],
                name="Generation Time (ms)",
                mode='lines+markers',
                line=dict(color='red', width=2, dash='dot')
            ),
            secondary_y=True
        )

        # Update layout
        fig.update_layout(
            title=f"Quality Trends - Last {days} Days",
            height=600,
            width=1000
        )

        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Quality Score (0-1)", secondary_y=False)
        fig.update_yaxes(title_text="Time (ms)", secondary_y=True)

        # Add alerts as annotations
        alerts = drift_analysis.get("alerts", [])
        if alerts:
            alert_text = "<br>".join([f"⚠️ {alert}" for alert in alerts])
            fig.add_annotation(
                text=alert_text,
                xref="paper", yref="paper",
                x=0.5, y=-0.2,
                showarrow=False,
                font=dict(size=10, color="red"),
                bgcolor="lightyellow",
                bordercolor="red",
                borderwidth=2
            )

        if save_path:
            fig.write_html(save_path)
            print(f"Quality trends saved to {save_path}")

        return fig

    def generate_full_report(
        self,
        days: int = 7,
        output_dir: str = "data/explanation_reports"
    ):
        """
        Generate comprehensive HTML report with all visualizations

        Args:
            days: Number of days to analyze
            output_dir: Directory to save reports
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate quality report data
        quality_report = self.analyzer.generate_quality_report(days=days)

        # Create visualizations
        print("Generating temporal drift visualization...")
        drift_fig = self.plot_temporal_drift(days=days)

        print("Generating quality trends...")
        quality_fig = self.plot_quality_trends(days=days)

        # Save individual figures
        if drift_fig:
            drift_fig.write_html(f"{output_dir}/drift_analysis_{timestamp}.html")

        if quality_fig:
            quality_fig.write_html(f"{output_dir}/quality_trends_{timestamp}.html")

        # Generate summary report HTML
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Explanation Quality Report - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .grade {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        .alert {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>📊 Explanation Quality Report</h1>
    <p><strong>Report Date:</strong> {quality_report['report_date']}</p>
    <p><strong>Analysis Period:</strong> Last {quality_report['period_days']} days</p>

    <h2>Overall Quality Grade</h2>
    <div class="metric">
        <span class="grade">{quality_report['quality_grade']}</span>
    </div>

    <h2>Key Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Total Explanations</td>
            <td>{quality_report['overview']['total_explanations']}</td>
        </tr>
        <tr>
            <td>Avg Citation Quality</td>
            <td>{quality_report['overview']['avg_citation_quality']:.3f}</td>
        </tr>
        <tr>
            <td>Avg Feature Coverage</td>
            <td>{quality_report['overview']['avg_feature_coverage']:.3f}</td>
        </tr>
        <tr>
            <td>Avg Generation Time</td>
            <td>{quality_report['overview']['avg_generation_time_ms']:.1f} ms</td>
        </tr>
        <tr>
            <td>Audiences Used</td>
            <td>{', '.join(quality_report['overview']['audiences_used'])}</td>
        </tr>
        <tr>
            <td>Models Used</td>
            <td>{', '.join(quality_report['overview']['models_used'])}</td>
        </tr>
    </table>

    <h2>Top Features Mentioned</h2>
    <table>
        <tr>
            <th>Feature</th>
            <th>Mentions</th>
            <th>Avg Citation Quality</th>
        </tr>
        {"".join([
            f"<tr><td>{f['feature']}</td><td>{f['mentions']}</td><td>{f['avg_citation_quality']:.3f}</td></tr>"
            for f in quality_report['feature_trends']['top_features'][:10]
        ])}
    </table>

    <h2>Drift Alerts</h2>
    {"".join([
        f'<div class="alert">{alert}</div>'
        for alert in quality_report['temporal_drift'].get('alerts', [])
    ])}

    <h2>Interactive Visualizations</h2>
    <ul>
        <li><a href="drift_analysis_{timestamp}.html">Temporal Drift Analysis</a></li>
        <li><a href="quality_trends_{timestamp}.html">Quality Trends</a></li>
    </ul>

    <hr>
    <p><em>Generated by HOGE Explanation Monitoring System</em></p>
</body>
</html>
"""

        # Save main report
        report_path = f"{output_dir}/explanation_report_{timestamp}.html"
        with open(report_path, "w") as f:
            f.write(report_html)

        # Also save JSON version
        json_path = f"{output_dir}/explanation_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(quality_report, f, indent=2, default=str)

        print(f"\n✅ Full report generated:")
        print(f"   HTML: {report_path}")
        print(f"   JSON: {json_path}")
        print(f"   Overall Grade: {quality_report['quality_grade']}")

        return report_path

    def close(self):
        """Close connections"""
        if self.logger:
            self.logger.close()
        if self.analyzer:
            self.analyzer.close()


if __name__ == "__main__":
    # Example usage
    dashboard = ExplanationDashboard()

    print("Generating comprehensive explanation quality report...")
    report_path = dashboard.generate_full_report(days=30)

    print(f"\nReport available at: {report_path}")

    dashboard.close()
