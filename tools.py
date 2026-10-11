"""Phase 1 — Graph retrieval tools for the career agent (GraphRAG retrieval layer).

Only uses installed packages: neo4j, python-dotenv, langchain-core.
Every tool returns a JSON string so the LLM reasons over deterministic graph output.
"""
import json
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
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


def _run(query, params=None):
    with get_driver().session() as session:
        return [dict(r) for r in session.run(query, params or {})]


def _dump(obj):
    return json.dumps(obj, default=str)


@tool
def get_learner_profile(user: str) -> str:
    """Get a learner's profile: education, target role, current skills with levels. Args: user (person name)."""
    rows = _run(
        """
        MATCH (p:Person {name: $user})
        OPTIONAL MATCH (p)-[:TARGETS]->(r:JobRole)
        OPTIONAL MATCH (p)-[h:HAS_SKILL]->(s:Skill)
        RETURN p.name AS name, p.education AS education,
               r.role_id AS target_role_id, r.title AS target_role,
               collect(DISTINCT {skill: s.name, skill_id: s.skill_id, level: h.level}) AS skills
        """,
        {"user": user},
    )
    if not rows:
        return _dump({"error": "unknown user '{}'. Ask for their name, skills and goal first.".format(user)})
    return _dump(rows[0])


@tool
def analyze_skill_gap(user: str, role_id: str) -> str:
    """Skill-gap analysis: required vs current level per skill, status GAP/OK. Args: user, role_id (e.g. R01)."""
    role = _run("MATCH (r:JobRole {role_id: $rid}) RETURN r.title AS title", {"rid": role_id})
    if not role:
        roles = _run("MATCH (r:JobRole) RETURN r.role_id AS role_id, r.title AS title ORDER BY r.role_id")
        return _dump({"error": "unknown role_id", "available_roles": roles})
    gaps = _run(
        """
        MATCH (p:Person {name: $user})
        MATCH (r:JobRole {role_id: $role_id})
        MATCH (r)-[req:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (p)-[h:HAS_SKILL]->(s)
        WITH s, req, coalesce(h.level, 0) AS have
        RETURN s.skill_id AS skill_id, s.name AS skill, s.category AS category,
               req.min_level AS need, have, req.importance AS importance,
               CASE WHEN have >= req.min_level THEN 'OK' ELSE 'GAP' END AS status
        ORDER BY status DESC, (req.min_level - have) DESC
        """,
        {"user": user, "role_id": role_id},
    )
    return _dump({"role": role[0]["title"], "role_id": role_id, "gaps": gaps})


@tool
def get_learning_path(user: str, role_id: str) -> str:
    """Ordered learning path for gaps: unmet prerequisites first, core before bonus, with resources. Args: user, role_id."""
    gaps = _run(
        """
        MATCH (p:Person {name: $user})
        MATCH (r:JobRole {role_id: $role_id})
        MATCH (r)-[req:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (p)-[h:HAS_SKILL]->(s)
        WHERE h IS NULL OR h.level < req.min_level
        OPTIONAL MATCH (pre:Skill)-[:PREREQUISITE_FOR]->(s)
        WHERE NOT EXISTS { MATCH (p)-[h2:HAS_SKILL]->(pre) WHERE h2.level >= 2 }
        OPTIONAL MATCH (s)-[:LEARN_VIA]->(res:Resource)
        RETURN s.skill_id AS skill_id, s.name AS skill, req.importance AS importance,
               req.min_level AS need, coalesce(h.level, 0) AS have,
               collect(DISTINCT pre.name) AS unmet_prereqs,
               collect(DISTINCT {title: res.title, platform: res.platform,
                                 hours: res.duration_hrs}) AS resources
        ORDER BY req.importance ASC
        """,
        {"user": user, "role_id": role_id},
    )
    return _dump({"role_id": role_id, "steps": gaps})


@tool
def compare_roles(user: str) -> str:
    """Readiness % of one learner across ALL roles. Use for 'which role fits me?' Args: user."""
    rows = _run(
        """
        MATCH (p:Person {name: $user})
        MATCH (r:JobRole)-[req:REQUIRES]->(s:Skill)
        OPTIONAL MATCH (p)-[h:HAS_SKILL]->(s)
        WITH r, sum(req.min_level) AS total,
             sum(CASE WHEN h.level IS NULL THEN 0
                      WHEN h.level > req.min_level THEN req.min_level
                      ELSE h.level END) AS covered,
             sum(CASE WHEN h.level IS NULL OR h.level < req.min_level THEN 1 ELSE 0 END) AS gaps
        RETURN r.role_id AS role_id, r.title AS title,
               round(100.0 * covered / total) AS readiness_pct, gaps
        ORDER BY readiness_pct DESC
        """,
        {"user": user},
    )
    return _dump({"fit": rows})


@tool
def save_skill(user: str, skill: str, level: int) -> str:
    """Save learning progress: set/updates HAS_SKILL level (skill id or name, level 1-5). Args: user, skill, level."""
    _run(
        """
        MATCH (p:Person {name: $user})
        MATCH (s:Skill) WHERE s.skill_id = $sk OR s.name = $sk
        MERGE (p)-[h:HAS_SKILL]->(s)
        SET h.level = $level
        """,
        {"user": user, "sk": skill, "level": int(level)},
    )
    return _dump({"saved": {"user": user, "skill": skill, "level": int(level)}})


@tool
def register_learner(user: str, education: str = "", target_role_id: str = "R01", skills: str = "{}") -> str:
    """Register a new learner with goal + current skills. skills = JSON like '{"S01": 4, "SQL": 3}'. Args: user, education, target_role_id, skills."""
    try:
        skill_map = json.loads(skills) if isinstance(skills, str) else dict(skills)
    except Exception:
        skill_map = {}
    _run(
        """
        MERGE (p:Person {name: $user})
        SET p.education = $edu
        WITH p
        MATCH (r:JobRole {role_id: $rid})
        MERGE (p)-[:TARGETS]->(r)
        """,
        {"user": user, "edu": education, "rid": target_role_id},
    )
    if skill_map:
        _run(
            """
            UNWIND $rows AS row
            MATCH (p:Person {name: $user})
            MATCH (s:Skill) WHERE s.skill_id = row.sid OR s.name = row.sid
            MERGE (p)-[h:HAS_SKILL]->(s)
            SET h.level = row.level
            """,
            {"user": user, "rows": [{"sid": k, "level": int(v)} for k, v in skill_map.items()]},
        )
    return _dump({"registered": user, "target": target_role_id, "skills": skill_map})


@tool
def explore_skill(skill: str) -> str:
    """Explore any skill: prerequisites, related skills, roles needing it, resources. Args: skill (id or name)."""
    info = _run(
        """
        MATCH (s:Skill) WHERE s.skill_id = $sk OR s.name = $sk
        OPTIONAL MATCH (pre:Skill)-[:PREREQUISITE_FOR]->(s)
        OPTIONAL MATCH (s)-[:PREREQUISITE_FOR]->(nxt:Skill)
        OPTIONAL MATCH (s)-[rel:RELATED_TO]-(rel_s:Skill)
        OPTIONAL MATCH (r:JobRole)-[:REQUIRES]->(s)
        OPTIONAL MATCH (s)-[:LEARN_VIA]->(res:Resource)
        RETURN s.skill_id AS skill_id, s.name AS name, s.category AS category,
               collect(DISTINCT pre.name) AS prerequisites,
               collect(DISTINCT nxt.name) AS unlocks,
               collect(DISTINCT rel_s.name) AS related,
               collect(DISTINCT r.title) AS needed_by_roles,
               collect(DISTINCT {title: res.title, platform: res.platform}) AS resources
        """,
        {"sk": skill},
    )
    if not info:
        return _dump({"error": "unknown skill"})
    return _dump(info[0])


@tool
def list_roles() -> str:
    """List all career roles in the graph with level and demand. Takes no args."""
    rows = _run("MATCH (r:JobRole) RETURN r.role_id AS role_id, r.title AS title, r.level AS level, r.demand AS demand ORDER BY r.role_id")
    return _dump({"roles": rows})


ALL_TOOLS = [get_learner_profile, analyze_skill_gap, get_learning_path,
             compare_roles, save_skill, register_learner, explore_skill, list_roles]
