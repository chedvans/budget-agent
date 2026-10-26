"""
אג'נט לניהול תקציב חודשי.

המבנה (לפי שלושת חלקי האג'נט):
  מוח   - מודל Gemini של Google שמחליט בכל צעד איזה כלי להפעיל.
          החיבור נעשה דרך הממשק התואם-OpenAI של Gemini, ולכן משתמשים בספריית openai
          עם base_url של Google ומפתח GEMINI_API_KEY.
  כלים  - פונקציות ב-tools.py: טעינת נתונים, השוואה לתקציב, פירוט עסקאות,
          איתור מנויים, ובדיקת תוכנית חיסכון.
  לולאה - run_agent(): שולחת למוח את המצב, מריצה את הכלי שבחר, מחזירה לו
          את התוצאה, וחוזר חלילה עד שהמוח קורא ל-finish או שנגמרו הצעדים.

הרצה:
    python agent.py
    python agent.py --goal "אני רוצה לחסוך 2000 ש\"ח בחודש" --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tools import TOOL_SCHEMAS, BudgetState, run_tool

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXPENSES = BASE_DIR / "data" / "expenses.csv"
DEFAULT_BUDGET = BASE_DIR / "data" / "budget.json"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT = 90  # שניות לכל בקשה למודל - כדי שהאג'נט לא ייתקע לנצח
# כמה "לחשוב" לפני כל תשובה. low = מהיר וחסכוני, מספיק לבחירת כלים.
REASONING_EFFORT = os.getenv("GEMINI_REASONING_EFFORT", "low")


def _is_daily_quota(exc) -> bool:
    text = str(exc)
    return "PerDay" in text or "per day" in text
DEFAULT_GOAL = "בדוק את ההוצאות שלי בחודש האחרון מול התקציב, ומצא תוכנית מציאותית לחסוך 1,500 ש\"ח בחודש הבא."

SYSTEM_PROMPT = """אתה אג'נט לניהול תקציב משפחתי. אתה פועל רק באמצעות הכלים שברשותך.

דרך העבודה:
1. טען את קובץ ההוצאות ואת קובץ התקציב.
2. השווה לתקציב וזהה היכן הייתה חריגה.
3. לכל קטגוריה בעייתית - בדוק את העסקאות עצמן כדי להבין ממה נבעה ההוצאה
   (למשל: הרבה משלוחי אוכל קטנים, או מנוי שלא נחוץ). אל תנחש בלי לבדוק.
4. בנה תוכנית קיצוצים ובדוק אותה עם simulate_savings_plan.
   - אם target_reached=false: קרא את problems ואת missing, תקן את התוכנית ובדוק שוב.
   - העדף לקצץ קודם בקטגוריות שאינן חיוניות ושבהן הייתה חריגה.
5. סיים עם finish:
   - success=true רק אחרי ש-simulate_savings_plan החזיר target_reached=true.
   - אם אחרי כמה ניסיונות אין דרך מציאותית להגיע ליעד - success=false,
     ציין כמה כן אפשר לחסוך ומדוע אי אפשר יותר.

הדוח הסופי בעברית, קצר ומעשי: מצב נוכחי, היכן החריגות ומה גרם להן,
התוכנית (קטגוריה, סכום, פעולה מעשית), וסך החיסכון.
אם כלי מחזיר error - קרא אותו והחלט: תקן את הארגומנטים, או דווח על הבעיה."""


# ---------------------------------------------------------------------------
# לוג: מדפיס למסך וגם שומר לקובץ (כדי שיהיה פלט ריצה להגשה)
# ---------------------------------------------------------------------------

class RunLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._fh = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        if self._fh:
            self._fh.write(text + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def _short(obj, limit: int = 600) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " ...[קוצר]"


# ---------------------------------------------------------------------------
# המוח: קריאה ל-Gemini (דרך ספריית openai) עם ניסיונות חוזרים
# ---------------------------------------------------------------------------

def call_llm(client, model: str, messages: list[dict], log: RunLog, retries: int = 3):
    """שולח את השיחה למודל. בשגיאת רשת/עומס מנסה שוב, בשגיאת הרשאה עוצר."""
    import openai  # מיובא כאן כדי שהבדיקות ירוצו גם בלי הספרייה

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",  # המוח מחליט בעצמו. אם לא בחר כלי - הלולאה מבקשת ממנו להמשיך
                **({"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}),
            )
            return response.choices[0].message
        except openai.AuthenticationError as exc:
            raise SystemExit("מפתח ה-API אינו תקין. בדקו את GEMINI_API_KEY בקובץ .env.") from exc
        except openai.BadRequestError as exc:
            # Gemini מחזיר 400 (ולא 401) כשהמפתח שגוי
            if "api key" in str(exc).lower():
                raise SystemExit("מפתח ה-API אינו תקין. בדקו את GEMINI_API_KEY בקובץ .env.") from exc
            raise
        except openai.PermissionDeniedError as exc:
            raise SystemExit(f"אין הרשאה למודל '{model}' עם המפתח הזה: {exc}") from exc
        except openai.NotFoundError as exc:
            raise SystemExit(f"המודל '{model}' לא נמצא. בדקו את GEMINI_MODEL בקובץ .env.") from exc
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as exc:
            if isinstance(exc, openai.RateLimitError) and _is_daily_quota(exc):
                # מכסה יומית - אין טעם לנסות שוב עכשיו
                raise SystemExit(f"נגמרה המכסה היומית החינמית של המודל '{model}'. "
                                 "אפשר לעבור למודל אחר בשורה GEMINI_MODEL בקובץ .env "
                                 "(לכל מודל מכסה נפרדת), או לחכות למחר.") from exc
            if attempt == retries:
                if isinstance(exc, openai.RateLimitError):
                    raise SystemExit("חרגתם ממכסת הבקשות של Gemini (429). חכו דקה ונסו שוב, "
                                     "או בדקו את המכסה ב-AI Studio.") from exc
                raise SystemExit(f"אין חיבור ל-Gemini ({type(exc).__name__}). "
                                 "בדקו את חיבור האינטרנט והריצו את check_connection.py.") from exc
            wait = 2 ** attempt
            log(f"   ! שגיאת תקשורת ({type(exc).__name__}), מנסה שוב בעוד {wait} שניות...")
            time.sleep(wait)


def _assistant_message(msg, tool_calls) -> dict:
    """
    שומר את תשובת המודל בהיסטוריה כפי שהתקבלה.
    חשוב ב-Gemini: לקריאות לכלים מצורפת "חתימת מחשבה" (thought signature)
    שחייבים להחזיר כמו שהיא בבקשה הבאה, ולכן לא בונים את ההודעה מחדש ידנית.
    """
    data: dict = {"role": "assistant", "content": msg.content or ""}
    if tool_calls:
        data["tool_calls"] = [
            # model_dump שומר גם שדות נוספים שהגיעו מ-Gemini (כמו extra_content עם החתימה)
            tc.model_dump(exclude_none=True) if hasattr(tc, "model_dump") else
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in tool_calls
        ]
    return data


# ---------------------------------------------------------------------------
# הלולאה
# ---------------------------------------------------------------------------

def run_agent(goal: str, expenses_file: Path, budget_file: Path, client, model: str,
              max_steps: int, log: RunLog) -> dict:
    state = BudgetState()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"המטרה: {goal}\n\n"
            f"קובץ ההוצאות: {expenses_file}\n"
            f"קובץ התקציב: {budget_file}"
        )},
    ]

    log(f"מטרה: {goal}")
    log(f"מודל: {model} | מקסימום צעדים: {max_steps}")
    log("=" * 70)

    for step in range(1, max_steps + 1):
        msg = call_llm(client, model, messages, log)
        tool_calls = msg.tool_calls or []

        # שומרים את ההחלטה של המודל בהיסטוריה
        messages.append(_assistant_message(msg, tool_calls))
        if msg.content:
            log(f"[צעד {step}] מחשבה: {msg.content.strip()}")

        if not tool_calls:
            # המודל ענה בטקסט בלי לבחור כלי - מזכירים לו לעבוד עם הכלים או לסיים
            log(f"[צעד {step}] המודל לא בחר כלי, מבקש ממנו להמשיך.")
            messages.append({"role": "user", "content": "יש להמשיך באמצעות כלי, או לסיים עם finish."})
            continue

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None

            log(f"[צעד {step}] כלי: {name}({_short(args, 300) if args else ''})")

            if args is None:
                result = {"error": "הארגומנטים שנשלחו אינם JSON תקין."}

            elif name == "finish":
                # תנאי סיום עם בדיקה: לא מאשרים "הצלחה" שלא נבדקה בפועל
                if args.get("success") and state.last_plan is None:
                    result = {"error": "אי אפשר לסיים בהצלחה לפני ש-simulate_savings_plan "
                                       "החזיר target_reached=true. בדוק את התוכנית."}
                else:
                    log("=" * 70)
                    log("הסתיים בהצלחה" if args.get("success") else "הסתיים ללא השגת היעד")
                    log("")
                    log(args.get("report", ""))
                    return {"success": bool(args.get("success")), "steps": step,
                            "report": args.get("report", ""), "plan": state.last_plan}
            else:
                result = run_tool(state, name, args)

            if "error" in result:
                log(f"         -> שגיאה: {result['error']}")
            else:
                log(f"         -> {_short(result)}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # תנאי עצירה: נגמרו הצעדים בלי שהמטרה הושגה
    log("=" * 70)
    log(f"האג'נט עצר אחרי {max_steps} צעדים בלי להשלים את המשימה.")
    return {"success": False, "steps": max_steps,
            "report": "המשימה לא הושלמה בתוך מספר הצעדים המותר.", "plan": state.last_plan}


# ---------------------------------------------------------------------------
# נקודת כניסה
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    # תמיכה בעברית בטרמינל של Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="אג'נט לניהול תקציב חודשי")
    parser.add_argument("--goal", default=DEFAULT_GOAL, help="המטרה של האג'נט, בשפה חופשית")
    parser.add_argument("--expenses", default=str(DEFAULT_EXPENSES), help="קובץ CSV של הוצאות")
    parser.add_argument("--budget", default=str(DEFAULT_BUDGET), help="קובץ JSON של תקציב")
    parser.add_argument("--model", default=None, help="שם מודל Gemini (ברירת מחדל: GEMINI_MODEL מקובץ .env)")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--no-log", action="store_true", help="לא לשמור קובץ לוג בתיקיית runs")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("חסר מפתח API. צרו קובץ .env לפי .env.example והכניסו בו את GEMINI_API_KEY.")
        return 2
    model = args.model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

    for label, p in (("ההוצאות", args.expenses), ("התקציב", args.budget)):
        if not Path(p).exists():
            print(f"קובץ {label} לא נמצא: {p}")
            return 2

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)

    log_path = None if args.no_log else BASE_DIR / "runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.txt"
    log = RunLog(log_path)
    try:
        result = run_agent(args.goal, Path(args.expenses), Path(args.budget),
                           client, model, args.max_steps, log)
    except Exception as exc:  # שגיאה לא צפויה - מדווחים בצורה ברורה
        log(f"שגיאה בהרצת האג'נט: {type(exc).__name__}: {exc}")
        return 1
    finally:
        log.close()

    if log_path:
        print(f"\nפלט הריצה נשמר ב: {log_path}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
