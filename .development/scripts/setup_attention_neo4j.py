"""
Setup Separate Neo4j Database for Attention Analysis
=====================================================
Creates a new database specifically for multimodal attention circuits.

This keeps your attention research separate from the existing HOGE loan data.

Usage:
    python .development/scripts/setup_attention_neo4j.py
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("SETUP ATTENTION ANALYSIS DATABASE")
print("="*80)

# Get Neo4j connection details
use_remote = os.getenv("USE_REMOTE_NEO4J", "local").lower() == "remote"

if use_remote:
    uri = os.getenv("NEO4J_REMOTE_URI")
    user = os.getenv("NEO4J_REMOTE_USER", "neo4j")
    password = os.getenv("NEO4J_REMOTE_PASSWORD")
    system_db = "system"  # Neo4j Aura uses 'neo4j' as default
    attention_db = "neo4j"  # Aura doesn't support multiple databases, use default
    print("\n[INFO] Using Neo4j Aura (Remote)")
    print("[WARN] Neo4j Aura doesn't support multiple databases")
    print("[INFO] Will use default 'neo4j' database with label prefix 'Attention_'")
else:
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    system_db = "system"
    attention_db = "attention"  # Local Neo4j can have multiple databases
    print("\n[INFO] Using Local Neo4j")
    print(f"[INFO] Will create separate database: '{attention_db}'")

if not password:
    print("\n[ERROR] NEO4J_PASSWORD not set in .env file!")
    print("\nAdd to your .env:")
    if use_remote:
        print("  NEO4J_REMOTE_PASSWORD=your_aura_password")
    else:
        print("  NEO4J_PASSWORD=your_local_password")
    exit(1)

print(f"\nConnecting to: {uri}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))

    if not use_remote:
        # Local Neo4j: Create separate database
        print(f"\n[1/3] Creating database '{attention_db}'...")

        with driver.session(database=system_db) as session:
            # Check if database exists
            result = session.run("SHOW DATABASES")
            existing_dbs = [record['name'] for record in result]

            if attention_db in existing_dbs:
                print(f"[SKIP] Database '{attention_db}' already exists")
            else:
                try:
                    session.run(f"CREATE DATABASE {attention_db}")
                    print(f"[OK] Created database '{attention_db}'")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        print(f"[SKIP] Database already exists")
                    else:
                        raise

    # Create constraints and indexes
    print(f"\n[2/3] Creating constraints...")

    db_to_use = attention_db if not use_remote else "neo4j"
    prefix = "" if not use_remote else "Attention_"

    constraints = [
        f"CREATE CONSTRAINT {prefix}utterance_id IF NOT EXISTS FOR (u:{prefix}Utterance) REQUIRE u.utterance_id IS UNIQUE",
        f"CREATE CONSTRAINT {prefix}attention_head IF NOT EXISTS FOR (h:{prefix}AttentionHead) REQUIRE (h.layer, h.head) IS UNIQUE",
        f"CREATE CONSTRAINT {prefix}circuit_id IF NOT EXISTS FOR (c:{prefix}AttentionCircuit) REQUIRE c.circuit_id IS UNIQUE",
    ]

    with driver.session(database=db_to_use) as session:
        for constraint in constraints:
            try:
                session.run(constraint)
                constraint_name = constraint.split("FOR")[1].split("REQUIRE")[0].strip()
                print(f"  [OK] {constraint_name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    print(f"  [SKIP] Constraint already exists")
                else:
                    print(f"  [WARN] {e}")

    # Create indexes for performance
    print(f"\n[3/3] Creating indexes...")

    indexes = [
        f"CREATE INDEX {prefix}utterance_text IF NOT EXISTS FOR (u:{prefix}Utterance) ON (u.text)",
        f"CREATE INDEX {prefix}head_entropy IF NOT EXISTS FOR (h:{prefix}AttentionHead) ON (h.entropy)",
        f"CREATE INDEX {prefix}circuit_type IF NOT EXISTS FOR (c:{prefix}AttentionCircuit) ON (c.circuit_type)",
    ]

    with driver.session(database=db_to_use) as session:
        for index in indexes:
            try:
                session.run(index)
                index_name = index.split("FOR")[1].split("ON")[0].strip()
                print(f"  [OK] {index_name}")
            except Exception as e:
                if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                    print(f"  [SKIP] Index already exists")
                else:
                    print(f"  [WARN] {e}")

    driver.close()

    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)

    if not use_remote:
        print(f"\nAttention database ready: '{attention_db}'")
        print("\nTo use this database, update your .env:")
        print(f"  ATTENTION_NEO4J_DATABASE={attention_db}")
        print("\nOr pass database parameter:")
        print(f"  loader = AttentionCircuitLoader(database='{attention_db}')")
    else:
        print(f"\nUsing default 'neo4j' database with '{prefix}' label prefix")
        print("\nNodes will be labeled as:")
        print(f"  - {prefix}Utterance")
        print(f"  - {prefix}AttentionHead")
        print(f"  - {prefix}AttentionCircuit")

    print("\nNext steps:")
    print("  1. Run: python .development/scripts/load_attention_to_neo4j.py")
    print("  2. Query via Neo4j Browser or Cypher queries")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nMake sure:")
    print("  1. Neo4j is running")
    print("  2. Credentials are correct in .env")
    print("  3. For local Neo4j: Version 4.0+ required for multiple databases")
