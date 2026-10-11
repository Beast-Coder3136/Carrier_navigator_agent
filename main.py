from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from nodes import create_person, get_gap_context
load_dotenv()

app = FastAPI(title="Career Navigator & Skill Gap Agent")


class AddPersonRequest(BaseModel):
    name: str = Field(min_length=1)
    education: str = ""
    experience_years: int = 0
    target_role_id: str = "R01"
    skills: dict[str, int] = {}  # {skill_id_or_name: level 1-5}


class QueryRequest(BaseModel):
    user: str = Field(min_length=1)
    query: str = Field(min_length=1)  # goal / desire, e.g. "want to become AI engineer"
    role_id: str | None = None  # override stored TARGETS; e.g. "R01"


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/add_person")
def add_person(req: AddPersonRequest):
    """Route 1: add Person (+TARGETS role +HAS_SKILL levels) to the graph."""
    try:
        person = create_person(req.name, req.education, req.experience_years,
                               req.target_role_id, req.skills)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if person is None:
        raise HTTPException(status_code=500, detail="person not created")
    return person


@app.post("/query")
def query(req: QueryRequest):
    """Route 2: take user query/desire -> graph retrieval + LLM answer."""
    try:
        ctx = get_gap_context(req.user, req.role_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    answer = _answer_with_llm(req.query, ctx)
    return {"answer": answer, "context": ctx}


def _answer_with_llm(query: str, ctx: dict) -> str:
    gaps = [g for g in ctx["gaps"] if g["status"] == "GAP"]
    have = [g for g in ctx["gaps"] if g["status"] == "OK"]
    gap_bits = ["{} (need {}, have {})".format(g["skill"], g["need"], g["have"]) for g in gaps]
    res_bits = [
        "{}: {} [{}, {}h]".format(r["skill"], r["title"], r["platform"], r["duration_hrs"])
        for r in ctx["resources"][:8]
    ]
    lines = [
        "User: {}, target role: {}.".format(ctx["user"], ctx["role_id"]),
        "Have ({}): {}".format(len(have), ", ".join(g["skill"] for g in have) if have else "none yet."),
        "Gaps ({}): {}".format(len(gaps), "; ".join(gap_bits) if gap_bits else "none - job-ready."),
        "Resources: {}".format("; ".join(res_bits) if res_bits else "none."),
        "Question: {}".format(query),
    ]
    prompt = (
        "You are a career navigator. Using ONLY the graph context below, "
        "give a short skill-gap analysis + ordered 3-phase learning path "
        "(prerequisites first, core before bonus). Cite skill names.\n\n" + "\n".join(lines)
    )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite" ,  )
        return llm.invoke(prompt).content
    except Exception:
        # No API key / no network - still return deterministic graph summary
        return "\n".join(lines)
