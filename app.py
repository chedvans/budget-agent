"""
ממשק צ'אט לאג'נט התקציב.

הרצה:
    streamlit run app.py
ייפתח דפדפן עם חלון שיחה. אפשר להשתמש בנתוני הדוגמה או להעלות קבצים משלכם.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from agent import DEFAULT_BUDGET, DEFAULT_EXPENSES, DEFAULT_MODEL, GEMINI_BASE_URL, REQUEST_TIMEOUT
from chat_agent import ChatAgent

BASE_DIR = Path(__file__).resolve().parent

TOOL_LABELS = {
    "load_expenses": "טעינת הוצאות",
    "load_budget": "טעינת תקציב",
    "compare_to_budget": "השוואה לתקציב",
    "get_transactions": "בדיקת עסקאות",
    "find_recurring": "איתור מנויים",
    "simulate_savings_plan": "בדיקת תוכנית חיסכון",
    "finish": "סיום",
}

EXAMPLES = [
    "איפה חרגתי מהתקציב החודש?",
    "על מה הלך הכי הרבה כסף במסעדות ובתי קפה?",
    "תבנה לי תוכנית מציאותית לחסוך 1,500 ש\"ח בחודש הבא",
]

st.set_page_config(page_title="אג'נט תקציב", page_icon="💰", layout="centered")
st.markdown(
    """<style>
    .stApp, .stMarkdown, .stChatMessage, .stTextInput, textarea, section[data-testid="stSidebar"] {
        direction: rtl; text-align: right;
    }
    code, pre { direction: ltr; text-align: left; }
    ul, ol { padding-right: 1.5em; padding-left: 0; }
    li, li p { direction: rtl; text-align: right !important; }
    </style>""",
    unsafe_allow_html=True,
)


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

st.title("💰 אג'נט לניהול תקציב")
st.caption("שאלו על ההוצאות שלכם, או בקשו תוכנית חיסכון. האג'נט מחליט בעצמו מה לבדוק.")

if not api_key:
    st.error("חסר מפתח API. צרו קובץ .env בתיקיית הפרויקט עם השורה GEMINI_API_KEY=המפתח שלכם, והפעילו מחדש.")
    st.stop()


def save_upload(uploaded, suffix: str) -> Path:
    tmp_dir = Path(tempfile.gettempdir()) / "budget_agent_uploads"
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / f"{uploaded.file_id}{suffix}"
    path.write_bytes(uploaded.getvalue())
    return path


with st.sidebar:
    st.header("הגדרות")
    st.write(f"מודל: `{model}`")
    exp_upload = st.file_uploader("קובץ הוצאות (CSV)", type=["csv"])
    bud_upload = st.file_uploader("קובץ תקציב (JSON)", type=["json"])
    st.caption("בלי העלאה - משתמשים בנתוני הדוגמה שבתיקייה data.")
    show_steps = st.toggle("להציג את צעדי האג'נט", value=True)
    new_chat = st.button("שיחה חדשה", use_container_width=True)

expenses_file = save_upload(exp_upload, ".csv") if exp_upload else DEFAULT_EXPENSES
budget_file = save_upload(bud_upload, ".json") if bud_upload else DEFAULT_BUDGET
files_key = (str(expenses_file), str(budget_file), model)

# יוצרים אג'נט חדש בתחילת השיחה, בלחיצה על "שיחה חדשה", או כשהוחלפו הקבצים
if new_chat or st.session_state.get("files_key") != files_key:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)
    st.session_state.agent = ChatAgent(client, model, expenses_file, budget_file)
    st.session_state.history = []
    st.session_state.files_key = files_key


# ---------------------------------------------------------------------------
# הצגת השיחה
# ---------------------------------------------------------------------------

def render_steps(steps: list[dict]) -> None:
    for s in steps:
        label = TOOL_LABELS.get(s["tool"], s["tool"])
        if "error" in s["result"]:
            st.markdown(f"⚠️ **{label}** - {s['result']['error']}")
        else:
            st.markdown(f"✔️ **{label}**")


for item in st.session_state.history:
    with st.chat_message(item["role"]):
        if item.get("steps") and show_steps:
            with st.expander(f"צעדי האג'נט ({len(item['steps'])})"):
                render_steps(item["steps"])
        st.markdown(item["content"])

prompt = st.chat_input("כתבו הודעה...") or st.session_state.pop("pending", None)

if not st.session_state.history and not prompt:
    st.write("אפשר להתחיל באחת מהשאלות האלה:")
    cols = st.columns(len(EXAMPLES))
    for col, text in zip(cols, EXAMPLES):
        if col.button(text, use_container_width=True):
            st.session_state.pending = text
            st.rerun()

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("האג'נט עובד...", expanded=show_steps)

        def on_step(name, args, result):
            label = TOOL_LABELS.get(name, name)
            if "error" in result:
                status.markdown(f"⚠️ **{label}** - {result['error']}")
            else:
                status.markdown(f"✔️ **{label}**")

        try:
            reply = st.session_state.agent.ask(prompt, on_step=on_step,
                                               on_log=lambda t="": status.markdown(f"⏳ {t.strip(' !')}"))
        except SystemExit as exc:          # מפתח או שם מודל שגויים
            status.update(label="שגיאה", state="error")
            st.error(str(exc))
            st.session_state.history.append({"role": "assistant", "content": f"⚠️ {exc}"})
            st.stop()
        except Exception as exc:           # שגיאה לא צפויה - מציגים ולא קורסים
            status.update(label="שגיאה", state="error")
            st.error(f"אירעה שגיאה: {type(exc).__name__}: {exc}")
            st.session_state.history.append(
                {"role": "assistant", "content": f"⚠️ אירעה שגיאה: {type(exc).__name__}: {exc}"})
            st.stop()

        status.update(label=f"סיים ({len(reply['steps'])} צעדים)", state="complete", expanded=False)
        if reply.get("finished") and reply.get("success") is False:
            st.warning("האג'נט לא הצליח להגיע ליעד במלואו - ראו הסבר בתשובה.")
        st.markdown(reply["answer"])

    st.session_state.history.append(
        {"role": "assistant", "content": reply["answer"], "steps": reply["steps"]}
    )
