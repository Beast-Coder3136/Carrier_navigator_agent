"""Phase 4 — Streamlit chat UI. Talks to FastAPI (main.py) via /persons + /chat."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

ROLES = {
    "R01": "AI Engineer",
    "R02": "ML Engineer",
    "R03": "Data Analyst",
    "R04": "Data Engineer",
    "R05": "Backend Developer AI APIs",
}

SUGGESTED = [
    "I want to become an AI engineer. What am I missing?",
    "Give me a step-by-step learning path with resources",
    "Which role fits me best right now?",
]

st.set_page_config(page_title="Career Navigator", page_icon="🎓", layout="wide")


def api_ok():
    try:
        r = requests.get(API_URL + "/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------- sidebar: who is chatting ----------------
with st.sidebar:
    st.header("👤 Learner")
    user = st.text_input("Your name", value=st.session_state.get("user", "Aarav"))
    st.session_state["user"] = user
    education = st.text_input("Education", value="BTech CSE 3rd year")
    role_id = st.selectbox("Career goal", list(ROLES.keys()),
                           format_func=lambda rid: "{} — {}".format(rid, ROLES[rid]))
    if st.button("Register / Reset profile"):
        try:
            resp = requests.post(API_URL + "/persons", json={
                "name": user, "education": education,
                "target_role_id": role_id, "skills": {},
            }, timeout=30)
            if resp.status_code == 200:
                st.success("Registered {} → {}".format(user, ROLES[role_id]))
                st.session_state["messages"] = []
            else:
                st.error("Register failed: {}".format(resp.text[:200]))
        except Exception as e:
            st.error("Backend unreachable: {}".format(e))
    st.divider()
    st.subheader("📊 Current skills")
    try:
        r = requests.get(API_URL + "/persons/" + user, timeout=10)
        if r.status_code == 200:
            prof = r.json()
            st.caption("Goal: {}".format(prof.get("target_role") or prof.get("target_role_id") or "?"))
            skills = [s for s in prof.get("skills", []) if s.get("skill")]
            if skills:
                for s in sorted(skills, key=lambda x: str(x.get("skill"))):
                    lvl = s.get("level") or 0
                    st.write("{} — Lv {}/5".format(s.get("skill"), lvl))
                    st.progress(min(int(lvl), 5) / 5.0)
            else:
                st.caption("No skills recorded yet — mention them in chat.")
        else:
            st.caption("Not registered yet — hit Register.")
    except Exception:
        st.caption("Profile unavailable (backend down).")
    st.divider()
    st.subheader("🎯 Role fit")
    try:
        fr = requests.get(API_URL + "/fit/" + user, timeout=15)
        if fr.status_code == 200:
            fit = fr.json().get("fit", [])
            for row in fit:
                pct = row.get("readiness_pct") or 0
                st.write("{} — {}% ({} gaps)".format(
                    row.get("title"), pct, row.get("gaps")))
                st.progress(max(min(float(pct), 100), 0) / 100.0)
        else:
            st.caption("Fit unavailable.")
    except Exception:
        st.caption("Fit unavailable (backend down).")
    st.divider()
    st.caption("Backend: {}".format(API_URL))
    st.caption("Neo4j + Gemini GraphRAG agent")

# ---------------- main: chat ----------------
st.title("🎓 Career Navigator & Skill Gap Agent")
if not api_ok():
    st.warning("Backend not reachable at {}. Start it: venv\\Scripts\\python -m uvicorn main:app --reload".format(API_URL))

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

cols = st.columns(len(SUGGESTED))
for i, q in enumerate(SUGGESTED):
    if cols[i].button(q, key="sug{}".format(i)):
        st.session_state["pending"] = q

pending = st.session_state.pop("pending", None)
text = st.chat_input("Ask about skills, roles, learning paths...")
user_msg = pending or text

if user_msg:
    st.session_state["messages"].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Consulting the knowledge graph..."):
            try:
                resp = requests.post(API_URL + "/chat",
                                     json={"user": user, "message": user_msg},
                                     timeout=120)
                if resp.status_code == 200:
                    reply = resp.json().get("reply", "(empty reply)")
                else:
                    reply = "**Error {}:** {}".format(resp.status_code, resp.text[:300])
            except Exception as e:
                reply = "**Backend unreachable:** {}".format(e)
        st.markdown(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
