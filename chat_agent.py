"""
גרסת שיחה של האג'נט - משמשת את ממשק הצ'אט (app.py).

ההבדל מ-run_agent שב-agent.py:
  - השיחה נמשכת: כל הודעה חדשה של המשתמש מתווספת להיסטוריה, והאג'נט זוכר
    מה כבר נטען ונבדק.
  - בכל הודעה האג'נט מחליט בעצמו: לענות מיד, להפעיל כלים כדי לבדוק,
    לשאול שאלת הבהרה, או לבנות ולאמת תוכנית חיסכון ולסיים ב-finish.
  - תור (turn) מסתיים כשהמודל עונה בטקסט, או קורא ל-finish,
    או כשנגמרו הצעדים המותרים לתור אחד.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from agent import SYSTEM_PROMPT, _assistant_message, call_llm
from tools import BudgetState, run_tool

CHAT_ADDENDUM = """

אתה מדבר עם המשתמש בצ'אט, והשיחה נמשכת לאורך כמה הודעות.
- לשאלה על הנתונים (למשל "כמה הוצאתי על מזון?") - בדוק בכלים וענה בטקסט קצר. אין צורך ב-finish.
- לבקשה לתוכנית חיסכון - בצע את כל התהליך, אמת עם simulate_savings_plan וסיים ב-finish.
- אם חסר מידע הכרחי (למשל לא ברור כמה המשתמש רוצה לחסוך) - שאל אותו בטקסט, אל תנחש.
- אל תטען שוב קבצים שכבר נטענו בשיחה, אלא אם המשתמש ביקש.
- אם הבקשה לא קשורה לתקציב והוצאות - אמור בנימוס שזה התחום שלך."""

StepCallback = Callable[[str, dict, dict], None]


class ChatAgent:
    def __init__(self, client, model: str, expenses_file: Path, budget_file: Path,
                 max_steps_per_turn: int = 15):
        self.client = client
        self.model = model
        self.max_steps = max_steps_per_turn
        self.state = BudgetState()
        self.messages: list[dict] = [{
            "role": "system",
            "content": SYSTEM_PROMPT + CHAT_ADDENDUM + (
                f"\n\nקובץ ההוצאות: {expenses_file}\nקובץ התקציב: {budget_file}"
            ),
        }]

    def ask(self, user_text: str, on_step: StepCallback | None = None,
            on_log: Callable[[str], None] | None = None) -> dict:
        """
        מעביר הודעה אחת של המשתמש לאג'נט ומריץ את הלולאה עד שיש תשובה.
        on_step(name, args, result) נקרא אחרי כל כלי - כך הממשק מציג את הצעדים בזמן אמת.
        """
        self.messages.append({"role": "user", "content": user_text})
        self.state.last_plan = None  # תוכנית מאומתת שייכת לבקשה הנוכחית בלבד
        steps: list[dict] = []

        for _ in range(self.max_steps):
            msg = call_llm(self.client, self.model, self.messages, log=on_log or (lambda _t="": None))
            tool_calls = msg.tool_calls or []
            self.messages.append(_assistant_message(msg, tool_calls))

            if not tool_calls:
                # המודל החליט לענות (תשובה, או שאלת הבהרה) - התור הסתיים
                return {"answer": (msg.content or "").strip() or "(אין תשובה)",
                        "steps": steps, "finished": False, "success": None}

            final = None
            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = None

                if args is None:
                    result = {"error": "הארגומנטים שנשלחו אינם JSON תקין."}
                elif name == "finish":
                    if args.get("success") and self.state.last_plan is None:
                        result = {"error": "אי אפשר לסיים בהצלחה לפני ש-simulate_savings_plan "
                                           "החזיר target_reached=true. בדוק את התוכנית."}
                    else:
                        result = {"ok": True}
                        final = args
                else:
                    result = run_tool(self.state, name, args)

                steps.append({"tool": name, "args": args, "result": result})
                if on_step:
                    on_step(name, args or {}, result)

                # כל קריאה לכלי חייבת תשובה בהיסטוריה, גם finish, כדי שהשיחה תוכל להימשך
                self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                      "content": json.dumps(result, ensure_ascii=False)})

            if final is not None:
                return {"answer": final.get("report", ""), "steps": steps,
                        "finished": True, "success": bool(final.get("success")),
                        "plan": self.state.last_plan}

        return {"answer": "לא הצלחתי להשלים את הבקשה בתוך מספר הצעדים המותר. "
                          "נסו לנסח אותה בצורה ממוקדת יותר.",
                "steps": steps, "finished": False, "success": False}
