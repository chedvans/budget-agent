"""
ממשק צ'אט ליועץ התקציב.

הרצה:
    python -m streamlit run app.py
ייפתח דפדפן עם חלון שיחה. אפשר להשתמש בנתוני הדוגמה או להעלות קבצים משלכם.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from agent import (DEFAULT_BUDGET, DEFAULT_EXPENSES, DEFAULT_MODEL, GEMINI_BASE_URL,
                   REQUEST_TIMEOUT, BudgetAgent)
from tools import BudgetState, run_tool

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "יועץ התקציב"
USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/account_balance_wallet:"

TOOL_LABELS = {
    "load_expenses": "קריאת ההוצאות",
    "load_budget": "קריאת התקציב",
    "compare_to_budget": "השוואה לתקציב",
    "get_transactions": "בדיקת עסקאות",
    "find_recurring": "איתור מנויים קבועים",
    "simulate_savings_plan": "בדיקת תוכנית החיסכון",
    "finish": "סיכום",
}

EXAMPLES = [
    "איפה חרגתי מהתקציב החודש?",
    "ממה נבעה ההוצאה על מסעדות ובתי קפה?",
    "בנה לי תוכנית לחסוך 1,500 ש\"ח בחודש הבא",
]

st.set_page_config(page_title=APP_NAME, page_icon=":material/account_balance_wallet:", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700&display=swap');
html, body, .stApp, p, h1, h2, h3, h4, li, label, button, input, textarea, div {
    font-family: 'Heebo', sans-serif;
}
.stApp, .stMarkdown, .stChatMessage, textarea, section[data-testid="stSidebar"],
[data-testid="stExpander"], [data-testid="stFileUploader"] {
    direction: rtl; text-align: right;
}
code, pre { direction: ltr; text-align: left; }
ul, ol { padding-right: 1.4em; padding-left: 0; }
li, li p { direction: rtl; text-align: right !important; }

.block-container { padding-top: 2.2rem; max-width: 820px; }

/* כותרת */
.hero { margin-bottom: 1.4rem; }
.hero .eyebrow { color: #0F766E; font-weight: 600; font-size: .82rem; letter-spacing: .04em; }
.hero h1 { font-size: 2rem; font-weight: 700; color: #111827; margin: .15rem 0 .35rem; padding: 0; }
.hero p { color: #6B7280; font-size: 1rem; margin: 0; }

/* כרטיסי סיכום */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: 0 0 1.6rem; }
.kpi { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; padding: .85rem 1rem; }
.kpi .label { color: #6B7280; font-size: .8rem; }
.kpi .value { color: #111827; font-size: 1.35rem; font-weight: 600; margin-top: .15rem; }
.kpi .value.bad { color: #B91C1C; }
.kpi .value.good { color: #047857; }
@media (max-width: 640px) { .kpis { grid-template-columns: repeat(2, 1fr); } }

/* הודעות */
[data-testid="stChatMessage"] { background: transparent; padding: .6rem 0; }
[data-testid="stChatMessageAvatarUser"] { background: #E5E7EB; color: #374151; }
[data-testid="stChatMessageAvatarAssistant"] { background: #0F766E; color: #FFFFFF; }

/* כפתורי הצעה */
div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    border-radius: 999px; border: 1px solid #D1D5DB; background: #FFFFFF;
    color: #374151; font-size: .88rem; padding: .45rem .9rem;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    border-color: #0F766E; color: #0F766E;
}
.hint { color: #6B7280; font-size: .9rem; margin-bottom: .4rem; }

section[data-testid="stSidebar"] h3 { font-size: .95rem; font-weight: 600; color: #111827; }
.model-tag { color: #9CA3AF; font-size: .78rem; margin-top: 1.5rem; direction: ltr; text-align: right; }
</style>
"""
st.html(CSS)  # st.html מזריק את העיצוב לעמוד (st.markdown היה מציג אותו כטקסט)


# ---------------------------------------------------------------------------
# הגדרות
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

st.markdown(
    f"""<div class="hero">
        <div class="eyebrow">ניהול תקציב אישי</div>
        <h1>{APP_NAME}</h1>
        <p>שאלו על ההוצאות שלכם או בקשו תוכנית חיסכון. היועץ בודק את הנתונים בעצמו ומאמת כל המלצה.</p>
    </div>""",
    unsafe_allow_html=True,
)

if not api_key:
    st.error("חסר מפתח API. צרו קובץ .env בתיקיית הפרויקט עם השורה GEMINI_API_KEY=המפתח שלכם, והפעילו מחדש.",
             icon=":material/key:")
    st.stop()


def save_upload(uploaded, suffix: str) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "budget_agent_uploads"
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / f"{uploaded.file_id}{suffix}"
    path.write_bytes(uploaded.getvalue())
    return path


with st.sidebar:
    st.markdown("### מקור הנתונים")
    exp_upload = st.file_uploader("קובץ הוצאות (CSV)", type=["csv"])
    bud_upload = st.file_uploader("קובץ תקציב (JSON)", type=["json"])
    st.caption("ללא העלאה, היועץ עובד עם נתוני הדוגמה.")
    st.markdown("### תצוגה")
    show_steps = st.toggle("הצגת שלבי הבדיקה", value=True)
    new_chat = st.button("שיחה חדשה", icon=":material/add_comment:", use_container_width=True)
    st.markdown(f'<div class="model-tag">{model}</div>', unsafe_allow_html=True)

expenses_file = save_upload(exp_upload, ".csv") if exp_upload else DEFAULT_EXPENSES
budget_file = save_upload(bud_upload, ".json") if bud_upload else DEFAULT_BUDGET
files_key = (str(expenses_file), str(budget_file), model)

# יוצרים שיחה חדשה בהתחלה, בלחיצה על "שיחה חדשה", או כשהוחלפו הקבצים
if new_chat or st.session_state.get("files_key") != files_key:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)
    st.session_state.agent = BudgetAgent(client, model, expenses_file, budget_file)
    st.session_state.history = []
    st.session_state.files_key = files_key


# ---------------------------------------------------------------------------
# כרטיסי סיכום - מחושבים ישירות בכלים, בלי לפנות למודל (לא צורך מכסה)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def month_summary(expenses_path: str, budget_path: str) -> dict | None:
    state = BudgetState()
    if "error" in run_tool(state, "load_expenses", {"file_path": expenses_path}):
        return None
    if "error" in run_tool(state, "load_budget", {"file_path": budget_path}):
        return None
    return run_tool(state, "compare_to_budget", {})


def money(x: float) -> str:
    return f"₪{x:,.0f}"


summary = month_summary(str(expenses_file), str(budget_file))
if summary and "error" not in summary:
    balance = summary.get("balance")
    over = len(summary["over_budget_categories"])
    cards = [
        ("סך ההוצאות", money(summary["total_spent"]), ""),
        ("הכנסה חודשית", money(summary["monthly_income"]) if summary.get("monthly_income") else "—", ""),
        ("יתרה", money(balance) if balance is not None else "—",
         "good" if balance and balance > 0 else "bad"),
        ("קטגוריות בחריגה", str(over), "bad" if over else "good"),
    ]
    st.markdown(
        '<div class="kpis">' + "".join(
            f'<div class="kpi"><div class="label">{label}</div><div class="value {cls}">{value}</div></div>'
            for label, value, cls in cards
        ) + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# השיחה
# ---------------------------------------------------------------------------

def step_line(name: str, result: dict) -> str:
    label = TOOL_LABELS.get(name, name)
    if "error" in result:
        return f":orange[:material/error:] **{label}** · {result['error']}"
    return f":green[:material/check_circle:] {label}"


def avatar(role: str) -> str:
    return USER_AVATAR if role == "user" else BOT_AVATAR


for item in st.session_state.history:
    with st.chat_message(item["role"], avatar=avatar(item["role"])):
        if item.get("steps") and show_steps:
            with st.expander(f"שלבי הבדיקה ({len(item['steps'])})", icon=":material/fact_check:"):
                for s in item["steps"]:
                    st.markdown(step_line(s["tool"], s["result"]))
        st.markdown(item["content"])

prompt = st.chat_input("כתבו שאלה על התקציב...") or st.session_state.pop("pending", None)

if not st.session_state.history and not prompt:
    st.markdown('<div class="hint">אפשר להתחיל מאחת השאלות:</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, text in zip(cols, EXAMPLES):
        if col.button(text, use_container_width=True):
            st.session_state.pending = text
            st.rerun()

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        status = st.status("מנתח את הנתונים...", expanded=show_steps)

        def on_step(step, name, args, result):
            status.markdown(step_line(name, result))

        def on_log(text=""):
            status.markdown(f":gray[:material/schedule:] {text.strip(' !')}")

        try:
            reply = st.session_state.agent.ask(prompt, on_step=on_step, on_log=on_log)
        except SystemExit as exc:          # מפתח, שם מודל או מכסה
            status.update(label="לא הושלם", state="error")
            st.error(str(exc), icon=":material/error:")
            st.session_state.history.append({"role": "assistant", "content": f"**לא הושלם:** {exc}"})
            st.stop()
        except Exception as exc:           # שגיאה לא צפויה - מציגים ולא קורסים
            status.update(label="לא הושלם", state="error")
            st.error(f"אירעה שגיאה: {type(exc).__name__}: {exc}", icon=":material/error:")
            st.session_state.history.append(
                {"role": "assistant", "content": f"**אירעה שגיאה:** {type(exc).__name__}: {exc}"})
            st.stop()

        n = len(reply["steps"])
        status.update(label=f"הבדיקה הושלמה · {n} שלבים" if n else "תשובה",
                      state="complete", expanded=False)
        if reply.get("finished") and reply.get("success") is False:
            st.warning("לא נמצאה דרך מציאותית להגיע ליעד במלואו. הפירוט בתשובה.", icon=":material/info:")
        st.markdown(reply["answer"])

    st.session_state.history.append(
        {"role": "assistant", "content": reply["answer"], "steps": reply["steps"]}
    )
