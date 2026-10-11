"""Ingest Career Navigator CSVs into Neo4j.

Only uses packages from requirements.txt: neo4j, python-dotenv (stdlib csv/pathlib/argparse).
No pandas. Python-side UNWIND works with AuraDB (LOAD CSV can't read local files).

Usage:
    python ingest.py                          # ingest data/, clear DB first
    python ingest.py --no-clear               # keep existing data
    python ingest.py --data ./data

Env (.env or shell):
    NEO4J_URI=bolt://localhost:7687  (or neo4j+s://xxx.databases.neo4j.io)
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=xxxx
"""
import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

DATA_DIR = Path(__file__).parent / "data"

CONSTRAINTS = [
    "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.skill_id IS UNIQUE",
    "CREATE CONSTRAINT role_id IF NOT EXISTS FOR (r:JobRole) REQUIRE r.role_id IS UNIQUE",
    "CREATE CONSTRAINT resource_id IF NOT EXISTS FOR (r:Resource) REQUIRE r.resource_id IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
]


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD) )


def run_cypher(driver, query, params=None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return [dict(r) for r in result]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(v, default=0) -> int:
    try:
        return int(str(v).strip())
    except (ValueError, TypeError, AttributeError):
        return default


def ingest(data_dir: Path = DATA_DIR, clear: bool = True):
    files = ["skills.csv", "roles.csv", "role_skills.csv",
             "skill_prereqs.csv", "skill_related.csv", "resources.csv"]
    for fn in files:
        p = data_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    driver = get_driver()
    try:
        # 0. verify connection early with a friendly error
        try:
            driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Neo4j at {NEO4J_URI}: {e}\n"
                "Check NEO4J_URI/USERNAME/PASSWORD and that AuraDB/database is running."
            ) from e

        for q in CONSTRAINTS:
            try:
                run_cypher(driver, q)
            except Exception as e:
                print(f"constraint skipped: {e}")

        if clear:
            run_cypher(driver, "MATCH (n) DETACH DELETE n")
            print("cleared DB")

        # 1. skills
        skills = read_csv(data_dir / "skills.csv")
        run_cypher(driver, """
            UNWIND $rows AS row
            MERGE (s:Skill {skill_id: row.skill_id})
            SET s.name = row.name, s.category = row.category,
                s.difficulty = row.difficulty
        """, {"rows": skills})
        print(f"skills: {len(skills)}")

        # 2. roles
        roles = read_csv(data_dir / "roles.csv")
        run_cypher(driver, """
            UNWIND $rows AS row
            MERGE (r:JobRole {role_id: row.role_id})
            SET r.title = row.title, r.level = row.level, r.demand = row.demand,
                r.salary = row.avg_salary_inr_lpa, r.description = row.description
        """, {"rows": roles})
        print(f"roles: {len(roles)}")

        # 3. role -> skill (REQUIRES)
        role_skills = read_csv(data_dir / "role_skills.csv")
        for row in role_skills:
            row["min_level"] = to_int(row.get("min_level"), 2)
        run_cypher(driver, """
            UNWIND $rows AS row
            MATCH (r:JobRole {role_id: row.role_id})
            MATCH (s:Skill {skill_id: row.skill_id})
            MERGE (r)-[rel:REQUIRES]->(s)
            SET rel.importance = row.importance, rel.min_level = row.min_level
        """, {"rows": role_skills})
        print(f"role_skills: {len(role_skills)}")

        # 4. prereqs: (prereq)-[:PREREQUISITE_FOR]->(skill)
        prereqs = read_csv(data_dir / "skill_prereqs.csv")
        run_cypher(driver, """
            UNWIND $rows AS row
            MATCH (s:Skill {skill_id: row.skill_id})
            MATCH (pre:Skill {skill_id: row.prereq_skill_id})
            MERGE (pre)-[:PREREQUISITE_FOR]->(s)
        """, {"rows": prereqs})
        print(f"skill_prereqs: {len(prereqs)}")

        # 5. related
        related = read_csv(data_dir / "skill_related.csv")
        run_cypher(driver, """
            UNWIND $rows AS row
            MATCH (a:Skill {skill_id: row.skill_id_a})
            MATCH (b:Skill {skill_id: row.skill_id_b})
            MERGE (a)-[rel:RELATED_TO]-(b)
            SET rel.relation = row.relation
        """, {"rows": related})
        print(f"skill_related: {len(related)}")

        # 6. resources + LEARN_VIA
        resources = read_csv(data_dir / "resources.csv")
        for row in resources:
            row["duration_hrs"] = to_int(row.get("duration_hrs"), 0)
        run_cypher(driver, """
            UNWIND $rows AS row
            MATCH (s:Skill {skill_id: row.skill_id})
            MERGE (res:Resource {resource_id: row.resource_id})
            SET res.title = row.title, res.type = row.type,
                res.platform = row.platform, res.duration_hrs = row.duration_hrs,
                res.difficulty = row.difficulty, res.url = row.url, res.cost = row.cost
            MERGE (s)-[:LEARN_VIA {priority: 1}]->(res)
        """, {"rows": resources})
        print(f"resources: {len(resources)}")

        counts = run_cypher(driver, """
            MATCH (s:Skill) WITH count(s) AS skills
            MATCH (r:JobRole) WITH skills, count(r) AS roles
            MATCH (res:Resource) WITH skills, roles, count(res) AS resources
            MATCH ()-[q:REQUIRES]->() WITH skills, roles, resources, count(q) AS requires
            MATCH ()-[pr:PREREQUISITE_FOR]->() WITH skills, roles, resources, requires, count(pr) AS prereqs
            RETURN skills, roles, resources, requires, prereqs
        """)
        print("DB counts:", counts[0] if counts else "unknown")
        return counts
    finally:
        driver.close()



parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(DATA_DIR))
parser.add_argument("--no-clear", action="store_true",
                        help="skip DETACH DELETE, merge into existing DB")
args = parser.parse_args()
ingest(Path(args.data), clear=not args.no_clear)
