"""Neo4j helpers — driver + Person/gap queries (uses neo4j + python-dotenv)."""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return _driver


def run_cypher(query, params=None):
    with get_driver().session() as session:
        return [dict(r) for r in session.run(query, params or {})]


def create_person(name, education="", experience_years=0, target_role_id="R01", skills=None):
    """MERGE Person + TARGETS role + HAS_SKILL levels. skills={skill_id_or_name: 1-5}."""
    run_cypher(
        """
        MERGE (p:Person {name: $name})
        SET p.education = $education, p.experience_years = $exp
        WITH p
        MATCH (r:JobRole {role_id: $role_id})
        MERGE (p)-[:TARGETS]->(r)
        """,
        {"name": name, "education": education, "exp": experience_years, "role_id": target_role_id},
    )
    if skills:
        run_cypher(
            """
            UNWIND $rows AS row
            MATCH (p:Person {name: $name})
            MATCH (s:Skill) WHERE s.skill_id = row.sid OR s.name = row.sid
            MERGE (p)-[h:HAS_SKILL]->(s)
            SET h.level = row.level
            """,
            {"name": name, "rows": [{"sid": k, "level": int(v)} for k, v in skills.items()]},
        )
    return get_person(name)


def get_person(name):
    rows = run_cypher(
        """
        MATCH (p:Person {name: $name})
        OPTIONAL MATCH (p)-[:TARGETS]->(r:JobRole)
        OPTIONAL MATCH (p)-[h:HAS_SKILL]->(s:Skill)
        RETURN p.name AS name, p.education AS education,
               p.experience_years AS experience_years,
               r.role_id AS target_role_id, r.title AS target_role,
               collect(DISTINCT {skill: s.name, skill_id: s.skill_id, level: h.level}) AS skills
        """,
        {"name": name},
    )
    return rows[0] if rows else None


def get_gap_context(user, role_id=None):
    """Deterministic graph retrieval: gaps + ordered path + resources."""
    if role_id is None:
        person = get_person(user)
        if not person or not person["target_role_id"]:
            raise ValueError(f"unknown user '{user}' and no role_id given")
        role_id = person["target_role_id"]
    gaps = run_cypher(
        """
        MATCH (p:Person {name: $user})-[:TARGETS]->(r:JobRole {role_id: $role_id})
        MATCH (r)-[req:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (p)-[h:HAS_SKILL]->(s)
        WITH s, req, coalesce(h.level, 0) AS have
        RETURN s.skill_id AS skill_id, s.name AS skill,
               req.min_level AS need, have, req.importance AS importance,
               CASE WHEN have >= req.min_level THEN 'OK' ELSE 'GAP' END AS status
        ORDER BY status DESC, (req.min_level - have) DESC
        """,
        {"user": user, "role_id": role_id},
    )
    gap_ids = [g["skill_id"] for g in gaps if g["status"] == "GAP"]
    resources = run_cypher(
        """
        MATCH (s:Skill)-[:LEARN_VIA]->(res:Resource)
        WHERE s.skill_id IN $gap_ids
        RETURN s.name AS skill, res.title AS title, res.platform AS platform,
               res.duration_hrs AS duration_hrs, res.difficulty AS difficulty
        ORDER BY s.name
        """,
        {"gap_ids": gap_ids},
    ) if gap_ids else []
    return {"user": user, "role_id": role_id, "gaps": gaps, "resources": resources}

