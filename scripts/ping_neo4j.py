#!/usr/bin/env python3
"""
Neo4j Aura Keep-Alive Script
=============================
Runs a simple query to prevent auto-pause.

Usage:
    python scripts/ping_neo4j.py

For GitHub Actions:
    Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD as environment variables.
"""

# Standard library
import os
from datetime import datetime

# Third-party
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Read from environment
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def ping_neo4j():
    """Send a simple query to keep Aura alive."""
    if not URI or not PASSWORD:
        print("❌ Missing NEO4J_URI or NEO4J_PASSWORD environment variable")
        exit(1)

    try:
        print(f"📡 Connecting to Neo4j at {URI}...")
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

        with driver.session(database=DATABASE) as session:
            # Simple count query (minimal resource usage)
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]

            print(f"✅ Neo4j Aura is alive! Total nodes: {count}")
            print(f"📅 Timestamp: {datetime.now().isoformat()}")

        driver.close()
        print("🔌 Connection closed successfully")

    except Exception as e:
        print(f"❌ Error pinging Neo4j: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    ping_neo4j()
