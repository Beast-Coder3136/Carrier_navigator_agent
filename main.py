"""Phase 3 — Chat API: single POST /chat + serve the Phase-4 chat UI."""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Career Navigator & Skill Gap Agent")
STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    user: str = Field(default="Aarav", min_length=1)
    message: str = Field(min_length=1)


class AddPersonRequest(BaseModel):
    name: str = Field(min_length=1)
    education: str = ""
    target_role_id: str = "R01"
    skills: dict[str, int] = {}  # {skill_id_or_name: level 1-5}


@app.get("/")
def root():
    page = STATIC_DIR / "chat.html"
    if page.exists():
        return FileResponse(page)
    return {"status": "ok", "hint": "POST /chat with {user, message}"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """Chat in, answer out. Thread per user keeps conversation memory."""
    try:
        from agent import chat as agent_chat
    except Exception as e:
        raise HTTPException(status_code=500, detail="agent unavailable: {}".format(e))
    try:
        reply = agent_chat(req.message, thread_id=req.user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"reply": reply, "user": req.user}


@app.post("/persons")
def add_person(req: AddPersonRequest):
    """Register a new learner: Person + TARGETS role + HAS_SKILL levels."""
    print("[persons] register name={} target={} skills={}".format(
        req.name, req.target_role_id, req.skills), flush=True)
    try:
        from tools import register_learner
        raw = register_learner.invoke({
            "user": req.name,
            "education": req.education,
            "target_role_id": req.target_role_id,
            "skills": json.dumps(req.skills),
        })
        result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
