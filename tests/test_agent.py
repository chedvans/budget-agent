"""
בדיקות אוטומטיות - רצות בלי מפתח API.

במקום Gemini אמיתי משתמשים ב-FakeClient שמחזיר החלטות מוכנות מראש.
כך בודקים שהלולאה, הכלים, הטיפול בשגיאות ותנאי הסיום עובדים נכון.

הרצה:  python -m unittest discover tests -v
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import BudgetAgent  # noqa: E402
from tools import BudgetState, run_tool  # noqa: E402

EXPENSES = ROOT / "data" / "expenses.csv"
BUDGET = ROOT / "data" / "budget.json"


def call(name, **args):
    return SimpleNamespace(id=f"call_{name}_{id(args)}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)))


class FakeClient:
    """מחקה את client.chat.completions.create ומחזיר צעד מתוך תסריט."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_messages = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen_messages.append(list(kwargs["messages"]))
        step = self.script.pop(0) if self.script else [call("compare_to_budget")]
        if isinstance(step, str):   # תשובת טקסט בלי כלים
            msg = SimpleNamespace(content=step, tool_calls=None)
        else:
            msg = SimpleNamespace(content=None, tool_calls=step)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.s = BudgetState()

    def test_requires_loading_first(self):
        self.assertIn("error", run_tool(self.s, "compare_to_budget", {}))

    def test_bad_row_is_skipped_not_crashing(self):
        r = run_tool(self.s, "load_expenses", {"file_path": str(EXPENSES)})
        self.assertEqual(len(r["skipped_rows"]), 1)
        self.assertGreater(r["loaded_rows"], 0)

    def test_missing_file(self):
        self.assertIn("error", run_tool(self.s, "load_expenses", {"file_path": "nope.csv"}))

    def test_unknown_tool_and_bad_args(self):
        self.assertIn("error", run_tool(self.s, "delete_everything", {}))
        self.assertIn("error", run_tool(self.s, "compare_to_budget", {"x": 1}))

    def test_plan_rejects_fixed_and_unrealistic_cuts(self):
        run_tool(self.s, "load_expenses", {"file_path": str(EXPENSES)})
        run_tool(self.s, "load_budget", {"file_path": str(BUDGET)})
        r = run_tool(self.s, "simulate_savings_plan", {"target": 1500, "cuts": [
            {"category": "שכירות", "amount": 500},
            {"category": "מזון", "amount": 1000},
        ]})
        self.assertFalse(r["target_reached"])
        self.assertEqual(len(r["problems"]), 2)


class AgentLoopTests(unittest.TestCase):
    def test_full_run_with_correction(self):
        """המוח מנסה לסיים מוקדם, נדחה, מתקן תוכנית ומצליח."""
        script = [
            [call("load_expenses", file_path=str(EXPENSES)), call("load_budget", file_path=str(BUDGET))],
            [call("compare_to_budget")],
            [call("finish", success=True, report="מוקדם מדי")],          # חייב להידחות
            [call("simulate_savings_plan", target=1500, cuts=[
                {"category": "מסעדות ובתי קפה", "amount": 500}])],       # לא מספיק
            [call("simulate_savings_plan", target=1500, cuts=[
                {"category": "מסעדות ובתי קפה", "amount": 500},
                {"category": "ביגוד", "amount": 600},
                {"category": "מנויים", "amount": 170},
                {"category": "שונות", "amount": 110},
                {"category": "מזון", "amount": 150}])],
            [call("finish", success=True, report="דוח סופי")],
        ]
        client = FakeClient(script)
        result = BudgetAgent(client, "fake", EXPENSES, BUDGET, max_steps=10).ask("חסוך 1500")

        self.assertTrue(result["success"])
        self.assertEqual(result["rounds"], 6)
        self.assertTrue(result["plan"]["target_reached"])
        # בדיקה שהדחייה של הסיום המוקדם הגיעה למוח
        rejected = [m for m in client.seen_messages[3] if m["role"] == "tool" and "אי אפשר לסיים" in m["content"]]
        self.assertTrue(rejected)

    def test_stops_after_max_steps(self):
        client = FakeClient([])  # המוח "תקוע" וחוזר על אותו כלי
        result = BudgetAgent(client, "fake", EXPENSES, BUDGET, max_steps=3).ask("חסוך")
        self.assertFalse(result["success"])
        self.assertEqual(result["rounds"], 3)

    def test_honest_failure_is_allowed(self):
        client = FakeClient([[call("finish", success=False, report="היעד לא מציאותי")]])
        result = BudgetAgent(client, "fake", EXPENSES, BUDGET, max_steps=5).ask("חסוך 9000")
        self.assertFalse(result["success"])
        self.assertEqual(result["rounds"], 1)


class ConversationTests(unittest.TestCase):
    def test_multi_turn_conversation(self):
        """שאלה פשוטה נענית בטקסט, ובהודעה הבאה האג'נט זוכר את מה שכבר נטען."""
        client = FakeClient([
            [call("load_expenses", file_path=str(EXPENSES))],
            [call("get_transactions", category="מזון")],
            "על מזון הוצאת 3,028.90 ש\"ח.",
            [call("load_budget", file_path=str(BUDGET)), call("compare_to_budget")],
            "חרגת ב-7 קטגוריות.",
        ])
        agent = BudgetAgent(client, "fake", EXPENSES, BUDGET)
        steps_seen = []
        r1 = agent.ask("כמה הוצאתי על מזון?", on_step=lambda step, n, a, r: steps_seen.append(n))
        self.assertIn("3,028.90", r1["answer"])
        self.assertEqual(steps_seen, ["load_expenses", "get_transactions"])

        r2 = agent.ask("ואיפה חרגתי?")
        self.assertEqual(r2["answer"], "חרגת ב-7 קטגוריות.")
        self.assertNotIn("error", r2["steps"][-1]["result"])  # ההוצאות נשארו טעונות מהתור הקודם

    def test_finish_keeps_history_valid(self):
        """אחרי finish כל קריאה לכלי קיבלה תשובה, כך שאפשר להמשיך לשוחח."""
        client = FakeClient([[call("finish", success=False, report="אין מספיק נתונים")], "בבקשה"])
        agent = BudgetAgent(client, "fake", EXPENSES, BUDGET)
        r = agent.ask("תוכנית?")
        self.assertTrue(r["finished"])
        self.assertEqual(agent.messages[-1]["role"], "tool")
        self.assertEqual(agent.ask("תודה")["answer"], "בבקשה")


if __name__ == "__main__":
    unittest.main()
