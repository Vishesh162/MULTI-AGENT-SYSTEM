import re, time
import streamlit as st
from scene3d import scene
from tools import get_secret
from agents import (build_search_agent, build_reader_agent, build_factcheck_agent,
                    writer_chain, critic_chain)
try:
    from langchain_google_genai.chat_models import GoogleRateLimitError
except ImportError:
    GoogleRateLimitError = Exception

TARGET_SCORE = 8.5
MAX_ROUNDS = 2

st.set_page_config(page_title="Research desk", page_icon="🔭", layout="wide",
                   initial_sidebar_state="collapsed")

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500&display=swap');
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.stApp { background: #0E1A2B; color: #E6EBF2; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button { font-family: 'IBM Plex Sans', sans-serif; }
.block-container { max-width: 1200px; padding: 2rem 1.5rem 4rem; }

.mast { display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 2rem; margin-bottom: 1.6rem;
        animation: rise .7s cubic-bezier(.2,.7,.2,1) both; }
.mast h1 { font-family: 'Bricolage Grotesque', sans-serif; font-size: clamp(2.6rem, 6vw, 4.6rem);
           line-height: .92; letter-spacing: -0.035em; margin: 0; color: #E6EBF2; font-weight: 700; }
.mast p { margin: 0; max-width: 34ch; color: #9AABC4; font-size: .98rem; line-height: 1.55; }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } }
@media (max-width: 700px) { .mast { grid-template-columns: 1fr; gap: .8rem; } }

div[data-testid="stTextInput"] input { font-size: 1.15rem; padding: .95rem 1.1rem; background: #15263D;
        color: #E6EBF2; border: 1.5px solid #2A3D58; border-radius: 8px; caret-color: #F2B134; }
div[data-testid="stTextInput"] input::placeholder { color: #5d7598; }
div[data-testid="stTextInput"] input:focus { border-color: #F2B134; box-shadow: 0 0 0 3px rgba(242,177,52,.15); }
div[data-testid="stTextInput"] > div { border: 0 !important; background: transparent !important; }
div[data-testid="InputInstructions"] { display: none; }
.stFormSubmitButton button { background: #F2B134 !important; color: #1d1400 !important; border: 0 !important;
        border-radius: 8px; font-weight: 500; padding: .65rem 1.5rem; transition: transform .12s; }
.stFormSubmitButton button:hover { transform: translateY(-1px); background: #f5c058 !important; }
.stFormSubmitButton button:focus-visible { outline: 2px solid #E6EBF2; outline-offset: 2px; }

.rail { background: #15263D; border: 1px solid #22344D; border-radius: 12px; padding: .4rem 1.1rem; }
.step { display: grid; grid-template-columns: 18px 1fr auto; gap: .7rem; align-items: center;
        padding: .6rem 0; border-top: 1px solid #1d2f47; }
.step:first-child { border-top: 0; }
.dot { width: 10px; height: 10px; border-radius: 50%; border: 2px solid #3A5274; }
.step.run .dot { border-color: #F2B134; background: #F2B134; animation: pulse 1.1s ease-in-out infinite; }
.step.ok .dot { border-color: #5FB3A6; background: #5FB3A6; }
.step b { font-weight: 500; color: #C9D3E1; font-size: .92rem; }
.step.run b { color: #F2B134; } .step.ok b { color: #E6EBF2; }
.step em { font-style: normal; font-size: .8rem; color: #9FD3CA; }
@keyframes pulse { 50% { opacity: .35; transform: scale(.8); } }
@media (prefers-reduced-motion: reduce) { .step.run .dot, .mast { animation: none; } }

.empty { border: 1.5px dashed #2A3D58; border-radius: 12px; padding: 3rem 2rem; color: #7d8ea8; text-align: center; }
.report-wrap { background: #F4F6F9; color: #18263A; border-radius: 10px; padding: clamp(1.4rem, 4vw, 2.6rem) clamp(1.2rem, 4vw, 3rem);
               animation: rise .6s cubic-bezier(.2,.7,.2,1) both; }
.report-wrap * { color: #18263A; }
.report-wrap h1, .report-wrap h2, .report-wrap h3 { font-family: 'Bricolage Grotesque', sans-serif; color: #0E1A2B; letter-spacing: -0.02em; }
.report-wrap p, .report-wrap li { line-height: 1.7; max-width: 72ch; }
.report-wrap a { color: #1F5F8B; }
.report-wrap code, .report-wrap pre { background: #E4E9EF; }
.stDownloadButton button { border-radius: 8px; border: 1px solid #3A5274; }
</style>
""")

st.markdown("""<div class="mast"><h1>Research desk</h1>
<p>Name a topic. Agents search the web, read the best pages, write a cited report,
then grade and rewrite it until it holds up.</p></div>""", unsafe_allow_html=True)

STEPS = [("search", "Search", "Finds recent sources and data"),
         ("read", "Read", "Pulls facts and numbers from the best pages"),
         ("write", "Write", "Drafts a cited report"),
         ("critic", "Grade", "Scores the draft out of 10"),
         ("fact", "Fact-check", "Checks claims before a rewrite")]

def rail_html(status, notes, score=None, round_no=0):
    rows = []
    for key, name, desc in STEPS:
        cls = {"run": "run", "ok": "ok"}.get(status.get(key, ""), "")
        rows.append(f'<div class="step {cls}" title="{desc}"><div class="dot"></div>'
                    f'<b>{name}</b><em>{notes.get(key, "")}</em></div>')
    return f'<div class="rail">{"".join(rows)}</div>'

KEYS = [k for k, _, _ in STEPS]
def active_done(status):
    act = next((i for i, k in enumerate(KEYS) if status.get(k) == "run"), -1)
    return act, [i for i, k in enumerate(KEYS) if status.get(k) == "ok"]

def text_of(msg):
    return getattr(msg, "text", None) or str(msg.content)

def get_score(fb) -> float:
    m = re.search(r"Score:\s*([\d.]+)", str(fb))
    return float(m.group(1)) if m else 0.0


left, right = st.columns([2.1, 1], gap="large")

with left:
    keys_missing = (not get_secret("ROUTIX_API_KEY") or get_secret("ROUTIX_API_KEY") == "missing_key") or (not get_secret("TAVILY_API_KEY") or get_secret("TAVILY_API_KEY") == "missing_key")
    if keys_missing:
        st.warning("⚠️ **API keys not found in Secrets!** Please go to Streamlit Cloud **Manage app > Settings > Secrets** and add `ROUTIX_API_KEY` and `TAVILY_API_KEY`.")
    with st.form("ask", border=False):
        topic = st.text_input("Topic", placeholder="What do you want researched?",
                              label_visibility="collapsed")
        submitted = st.form_submit_button("Write report", type="primary")
    report_slot = st.container()

with right:
    scene_slot = st.empty()
    rail = st.empty()

res = st.session_state.get("res")
def paint(status, notes, score=None, rnd=0):
    a, d = active_done(status)
    with scene_slot:
        scene(a, d, score)
    rail.markdown(rail_html(status, notes, score, rnd), unsafe_allow_html=True)

if res:
    paint(res["status"], res["notes"], res["score"], res["rounds"])
else:
    paint({}, {})


def run(topic):
    status, notes = {}, {}
    def show(score=None, rnd=0):
        paint(status, notes, score, rnd)

    status["search"] = "run"; show()
    r = build_search_agent().invoke({"messages": [("user",
        f"Find recent, reliable and detailed information about: {topic}. "
        f"Include statistics, expert reports, and criticism/risks. Return URLs.")]})
    search = text_of(r["messages"][-1])
    n_urls = len(set(re.findall(r"https?://[^\s\)\]]+", search)))
    status["search"] = "ok"; notes["search"] = f"{n_urls} sources"

    status["read"] = "run"; show()
    r = build_reader_agent().invoke({"messages": [("user",
        f"Based on the following search results about '{topic}', pick the 3-4 most relevant URLs "
        f"(different websites) and scrape each one. Keep exact numbers, dates and the source URL "
        f"for every fact.\n\nSearch Results:\n{search}")]})
    scraped = text_of(r["messages"][-1])
    status["read"] = "ok"; notes["read"] = "done"

    research = f"SEARCH RESULTS:\n{search}\n\nSCRAPED CONTENT:\n{scraped}"
    extra, best, scores = "", {"score": 0, "report": "", "feedback": ""}, []

    for n in range(1, MAX_ROUNDS + 1):
        try:
            status["write"] = "run"; notes["write"] = f"draft {n}"; show(best["score"] or None, n - 1)
            report = writer_chain.invoke({"topic": topic, "research": research + extra})
            status["write"] = "ok"

            status["critic"] = "run"; show(best["score"] or None, n - 1)
            fb = critic_chain.invoke({"report": report})
            s = get_score(fb); scores.append(s)
            status["critic"] = "ok"; notes["critic"] = " → ".join(str(x) for x in scores)
            if s > best["score"]:
                best = {"score": s, "report": report, "feedback": fb}
            show(s, n)
            if s >= TARGET_SCORE or n == MAX_ROUNDS:
                break

            status["fact"] = "run"; show(s, n)
            r = build_factcheck_agent().invoke({"messages": [("user",
                f"Research:\n{research}\n\nReport:\n{report}")]})
            fc = text_of(r["messages"][-1])
            status["fact"] = "ok"; notes["fact"] = "sent back"
            extra = (f"\n\nPREVIOUS REPORT:\n{report}\n\nCRITIC FEEDBACK (fix all of this):\n{fb}"
                     f"\n\nFACT CHECK (fix WRONG, remove UNSUPPORTED):\n{fc}")
        except GoogleRateLimitError:
            notes["write"] = "quota hit"
            break

    show(best["score"], len(scores))
    return {"status": status, "notes": notes, "score": best["score"], "rounds": len(scores),
            "report": best["report"], "feedback": best["feedback"],
            "search": search, "scraped": scraped, "topic": topic}


if submitted:
    if not topic.strip():
        with report_slot:
            st.warning("Enter a topic to start.")
    elif keys_missing:
        with report_slot:
            st.error("Cannot run research: `ROUTIX_API_KEY` or `TAVILY_API_KEY` is missing in Secrets.")
    else:
        try:
            st.session_state.res = run(topic.strip())
        except GoogleRateLimitError:
            with report_slot:
                st.error("Gemini’s daily limit is used up. Try again later or use a paid key.")
        except Exception as e:
            print("[backend] error:", e)
            with report_slot:
                st.error("The report couldn’t be made. Check the terminal for the error.")

res = st.session_state.get("res")
with report_slot:
    if res and res["report"]:
        st.write("")
        st.markdown(f'<div class="report-wrap">\n\n{res["report"]}\n\n</div>', unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns([1, 3])
        c1.download_button("Download report", res["report"],
                           file_name=f"report_{int(time.time())}.md", mime="text/markdown")
        with st.expander("What the grader said"):
            st.markdown(res["feedback"])
        with st.expander("Sources the agents used"):
            st.text(res["search"])
    elif not submitted:
        st.write("")
        st.markdown('<div class="empty">Your report will show up here, with sources you can click.</div>',
                    unsafe_allow_html=True)