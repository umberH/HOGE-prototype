"""
Application Service
===================
Service for managing loan applications and retrieving application data.
"""

import sys
from pathlib import Path
from typing import List, Optional
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.api.models.application_dto import ApplicationInfo, ApplicationListResponse

load_dotenv()


class ApplicationService:
    """Service for application-related operations."""

    def __init__(self):
        """Initialize with Neo4j connection."""
        self.uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "test1234")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None

    def _get_driver(self):
        """Get or create Neo4j driver."""
        if self.driver is None:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
        return self.driver

    def get_all_applications(self, limit: Optional[int] = None) -> ApplicationListResponse:
        """
        Fetch all available application IDs from Neo4j.

        Args:
            limit: Optional limit on number of applications to return

        Returns:
            ApplicationListResponse with list of applications

        Raises:
            Exception: If Neo4j connection fails
        """
        driver = self._get_driver()

        try:
            with driver.session(database=self.database) as session:
                query = """
                    MATCH (app:LoanApplication)
                    OPTIONAL MATCH (app)-[:HAS_SHAP_EXPLANATION]->(exp:ModelExplanation)
                    RETURN app.application_id AS application_id,
                           app.status AS decision,
                           exp.probability AS probability
                    ORDER BY app.application_id
                """

                if limit:
                    query += f" LIMIT {limit}"

                result = session.run(query)

                applications = []
                for record in result:
                    app_id = record["application_id"]
                    decision = record.get("decision", "Unknown")
                    prob = record.get("probability")

                    applications.append(
                        ApplicationInfo(
                            application_id=app_id,
                            decision=decision,
                            probability=prob,
                            probability_str=f"{prob*100:.1f}%" if prob is not None else "N/A"
                        )
                    )

                return ApplicationListResponse.from_list(applications)

        except Exception as e:
            raise Exception(f"Failed to fetch applications from Neo4j: {str(e)}")

    def get_application_by_id(self, application_id: str) -> Optional[ApplicationInfo]:
        """
        Get a single application by ID.

        Args:
            application_id: Application ID to fetch

        Returns:
            ApplicationInfo if found, else None
        """
        driver = self._get_driver()

        try:
            with driver.session(database=self.database) as session:
                query = """
                    MATCH (app:LoanApplication {application_id: $application_id})
                    OPTIONAL MATCH (app)-[:HAS_SHAP_EXPLANATION]->(exp:ModelExplanation)
                    RETURN app.application_id AS application_id,
                           app.status AS decision,
                           exp.probability AS probability
                """

                result = session.run(query, {"application_id": application_id})
                record = result.single()

                if record:
                    prob = record.get("probability")
                    return ApplicationInfo(
                        application_id=record["application_id"],
                        decision=record.get("decision", "Unknown"),
                        probability=prob,
                        probability_str=f"{prob*100:.1f}%" if prob is not None else "N/A"
                    )

                return None

        except Exception as e:
            print(f"Error fetching application {application_id}: {e}")
            return None

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.driver = None
