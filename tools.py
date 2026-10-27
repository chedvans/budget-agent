"""
הכלים שהאג'נט יכול להפעיל.

כל כלי הוא פונקציה רגילה שמקבלת ארגומנטים ומחזירה dict.
הכלים לא "מחליטים" כלום - הם רק אוספים מידע, מחשבים או בודקים.
ההחלטה איזה כלי להפעיל ומתי שייכת למוח (LLM) שבקובץ agent.py.

כלל חשוב: כלי לעולם לא זורק חריגה החוצה. במקרה של שגיאה הוא מחזיר
{"error": "..."} - כך המוח רואה את השגיאה ויכול להחליט מה לעשות איתה
(לנסות קובץ אחר, לתקן ארגומנט, או לעצור ולדווח).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from pyluach import dates as hebdates, hebrewcal


class ToolError(Exception):
    """שגיאה צפויה בתוך כלי - תוחזר למוח כהודעה, לא תפיל את התוכנית."""


@dataclass
class BudgetState:
    """הזיכרון המשותף של הכלים בזמן ריצה אחת של האג'נט."""

    expenses: list[dict] = field(default_factory=list)
    budget: dict | None = None
    expenses_file: str | None = None
    last_plan: dict | None = None


# ---------------------------------------------------------------------------
# עזרים
# ---------------------------------------------------------------------------

def _require_expenses(state: BudgetState) -> None:
    if not state.expenses:
        raise ToolError("עדיין לא נטענו הוצאות. יש להפעיל קודם את load_expenses.")


def _require_budget(state: BudgetState) -> None:
    if state.budget is None:
        raise ToolError("עדיין לא נטען תקציב. יש להפעיל קודם את load_budget.")


def _totals_by_category(state: BudgetState) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for e in state.expenses:
        totals[e["category"]] += e["amount"]
    return {k: round(v, 2) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# הכלים
# ---------------------------------------------------------------------------

def load_expenses(state: BudgetState, file_path: str) -> dict:
    """טוען קובץ CSV של הוצאות (date, description, category, amount)."""
    path = Path(file_path)
    if not path.exists():
        raise ToolError(f"הקובץ '{file_path}' לא נמצא.")
    if path.suffix.lower() != ".csv":
        raise ToolError("קובץ ההוצאות חייב להיות בפורמט CSV.")

    required = {"date", "description", "category", "amount"}
    expenses, skipped = [], []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ToolError(f"חסרות עמודות בקובץ: {', '.join(sorted(missing))}")
        for line_no, row in enumerate(reader, start=2):
            try:
                try:
                    amount = float(row["amount"])
                except (ValueError, TypeError):
                    raise ValueError(f"סכום לא תקין: '{row['amount']}'")
                if amount < 0:
                    raise ValueError("סכום שלילי")
                expenses.append({
                    "date": row["date"].strip(),
                    "description": row["description"].strip(),
                    "category": row["category"].strip() or "ללא קטגוריה",
                    "amount": amount,
                })
            except (ValueError, TypeError) as exc:
                skipped.append({"line": line_no, "reason": str(exc)})

    if not expenses:
        raise ToolError("לא נמצאו שורות תקינות בקובץ.")

    state.expenses = expenses
    state.expenses_file = str(path)
    dates = sorted(e["date"] for e in expenses)
    return {
        "loaded_rows": len(expenses),
        "skipped_rows": skipped,
        "period": f"{dates[0]} עד {dates[-1]}",
        "total_spent": round(sum(e["amount"] for e in expenses), 2),
        "categories": sorted({e["category"] for e in expenses}),
    }


def load_budget(state: BudgetState, file_path: str) -> dict:
    """טוען קובץ JSON עם הכנסה חודשית ותקרת הוצאה לכל קטגוריה."""
    path = Path(file_path)
    if not path.exists():
        raise ToolError(f"הקובץ '{file_path}' לא נמצא.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolError(f"קובץ התקציב אינו JSON תקין: {exc}") from exc
    if "categories" not in data or not isinstance(data["categories"], dict):
        raise ToolError("בקובץ התקציב חסר המפתח 'categories'.")

    state.budget = data
    return {
        "monthly_income": data.get("monthly_income"),
        "total_budget": sum(c["limit"] for c in data["categories"].values()),
        "categories": {
            name: {"limit": c["limit"], "essential": c.get("essential", False),
                   "fixed": c.get("fixed", False)}
            for name, c in data["categories"].items()
        },
    }


def compare_to_budget(state: BudgetState) -> dict:
    """משווה את ההוצאה בפועל לתקציב בכל קטגוריה."""
    _require_expenses(state)
    _require_budget(state)

    totals = _totals_by_category(state)
    cats = state.budget["categories"]
    rows = []
    for name in sorted(set(totals) | set(cats)):
        spent = totals.get(name, 0.0)
        limit = cats.get(name, {}).get("limit")
        row = {
            "category": name,
            "spent": spent,
            "limit": limit,
            "essential": cats.get(name, {}).get("essential", False),
        }
        if limit is None:
            row["status"] = "אין תקציב לקטגוריה"
        else:
            row["difference"] = round(spent - limit, 2)
            row["status"] = "חריגה" if spent > limit else "בתוך התקציב"
        rows.append(row)

    total_spent = round(sum(totals.values()), 2)
    income = state.budget.get("monthly_income")
    return {
        "rows": sorted(rows, key=lambda r: r.get("difference", 0), reverse=True),
        "total_spent": total_spent,
        "monthly_income": income,
        "balance": round(income - total_spent, 2) if income else None,
        "over_budget_categories": [r["category"] for r in rows if r["status"] == "חריגה"],
    }


def get_transactions(state: BudgetState, category: str, top_n: int = 5) -> dict:
    """מחזיר את העסקאות הגדולות בקטגוריה + קיבוץ לפי תיאור (כדי לזהות הרגלים)."""
    _require_expenses(state)
    items = [e for e in state.expenses if e["category"] == category]
    if not items:
        known = sorted({e["category"] for e in state.expenses})
        raise ToolError(f"אין עסקאות בקטגוריה '{category}'. קטגוריות קיימות: {', '.join(known)}")

    top_n = max(1, min(int(top_n), 20))
    by_desc: dict[str, list[float]] = defaultdict(list)
    for e in items:
        by_desc[e["description"]].append(e["amount"])

    return {
        "category": category,
        "count": len(items),
        "total": round(sum(e["amount"] for e in items), 2),
        "largest": sorted(items, key=lambda e: e["amount"], reverse=True)[:top_n],
        "grouped_by_description": sorted(
            (
                {"description": d, "times": len(a), "total": round(sum(a), 2),
                 "average": round(sum(a) / len(a), 2)}
                for d, a in by_desc.items()
            ),
            key=lambda g: g["total"],
            reverse=True,
        ),
    }


def find_recurring(state: BudgetState) -> dict:
    """מאתר מנויים / תשלומים חוזרים שכדאי לבחון אם עדיין נחוצים."""
    _require_expenses(state)
    subs = [e for e in state.expenses
            if e["category"] == "מנויים" or e["description"].startswith("מנוי")]
    counts: dict[str, int] = defaultdict(int)
    for e in state.expenses:
        counts[e["description"]] += 1
    frequent = sorted(
        ({"description": d, "times": n} for d, n in counts.items() if n >= 3),
        key=lambda x: x["times"], reverse=True,
    )
    return {
        "subscriptions": subs,
        "subscriptions_total": round(sum(s["amount"] for s in subs), 2),
        "frequent_purchases": frequent,
    }


def simulate_savings_plan(state: BudgetState, target: float, cuts: list[dict]) -> dict:
    """
    בודק תוכנית חיסכון: האם הקיצוצים המוצעים מגיעים ליעד, והאם הם סבירים.
    זה הכלי שבו המוח 'בוחן את התוצאה' - אם היעד לא הושג, עליו לשנות את התוכנית.
    """
    _require_expenses(state)
    _require_budget(state)
    if target <= 0:
        raise ToolError("יעד החיסכון חייב להיות מספר חיובי.")
    if not isinstance(cuts, list) or not cuts:
        raise ToolError("יש לספק רשימת קיצוצים (cuts) לא ריקה.")

    totals = _totals_by_category(state)
    cats = state.budget["categories"]
    accepted, problems = [], []
    total_saving = 0.0

    for cut in cuts:
        name = cut.get("category")
        try:
            amount = float(cut.get("amount", 0))
        except (TypeError, ValueError):
            problems.append(f"סכום לא תקין לקטגוריה '{name}'.")
            continue
        spent = totals.get(name)
        if spent is None:
            problems.append(f"הקטגוריה '{name}' לא קיימת בהוצאות.")
            continue
        if amount <= 0:
            problems.append(f"קיצוץ ב-'{name}' חייב להיות חיובי.")
            continue
        if amount > spent:
            problems.append(f"אי אפשר לקצץ {amount} ב-'{name}' - הוצאו בה רק {spent}.")
            continue
        if cats.get(name, {}).get("fixed", False):
            problems.append(f"'{name}' היא הוצאה קבועה (למשל חוזה) ואי אפשר לקצץ בה בחודש הבא.")
            continue
        essential = cats.get(name, {}).get("essential", False)
        share = amount / spent
        # הגנה מפני תוכנית לא מציאותית: לא לחתוך יותר מ-10% מהוצאה חיונית
        # ולא יותר מ-60% מהוצאה שאינה חיונית.
        max_share = 0.10 if essential else 0.60
        if share > max_share:
            problems.append(
                f"קיצוץ של {share:.0%} ב-'{name}' לא מציאותי "
                f"(מקסימום {max_share:.0%} ל{'הוצאה חיונית' if essential else 'קטגוריה זו'}). "
                f"מקסימום אפשרי: {round(spent * max_share, 2)}."
            )
            continue
        accepted.append({
            "category": name,
            "amount": round(amount, 2),
            "reason": cut.get("reason", ""),
            "new_monthly_spend": round(spent - amount, 2),
        })
        total_saving += amount

    total_saving = round(total_saving, 2)
    reached = total_saving >= target and not problems
    result = {
        "target": target,
        "total_saving": total_saving,
        "target_reached": reached,
        "missing": round(max(0.0, target - total_saving), 2),
        "accepted_cuts": accepted,
        "problems": problems,
    }
    if reached:
        state.last_plan = result
    return result


# ---------------------------------------------------------------------------
# חגים - לפי הלוח העברי
# ---------------------------------------------------------------------------

# חודשים לפי הספרייה pyluach: ניסן=1 ... תשרי=7, כסלו=9, אדר=12 (בשנה מעוברת: אדר ב'=13)
HOLIDAYS = {
    "ראש השנה": (7, 1),
    "סוכות": (7, 15),
    "חנוכה": (9, 25),
    "פורים": ("adar", 14),
    "פסח": (1, 15),
    "שבועות": (3, 6),
}
HOLIDAY_ALIASES = {"חגי תשרי": "ראש השנה", "תשרי": "ראש השנה", "סוכות ושמחת תורה": "סוכות"}


def _holiday_date(name: str, hebrew_year: int) -> hebdates.HebrewDate:
    month, day = HOLIDAYS[name]
    if month == "adar":  # פורים: בשנה מעוברת חוגגים באדר ב'
        month = 13 if hebrewcal.Year(hebrew_year).leap else 12
    return hebdates.HebrewDate(hebrew_year, month, day)


def _next_occurrence(name: str, today: date) -> hebdates.HebrewDate:
    """המועד הקרוב של החג, מהיום והלאה."""
    today_heb = hebdates.GregorianDate(today.year, today.month, today.day).to_heb()
    for year in (today_heb.year, today_heb.year + 1):
        d = _holiday_date(name, year)
        if d >= today_heb:
            return d
    raise ToolError(f"לא נמצא מועד ל{name}.")  # לא אמור לקרות


def upcoming_holidays(today: date | None = None) -> list[dict]:
    """רשימת החגים הקרובים, מהקרוב לרחוק. משמש את הבאנר בממשק (לא כלי של המודל)."""
    today = today or date.today()
    result = []
    for name in HOLIDAYS:
        heb = _next_occurrence(name, today)
        greg = heb.to_pydate()
        result.append({"holiday": name, "date": greg.isoformat(), "days_left": (greg - today).days,
                       "hebrew_date": heb.hebrew_date_string()})
    return sorted(result, key=lambda h: h["days_left"])


def holiday_countdown(state: BudgetState, holiday: str, amount: float | None = None,
                      today: str | None = None) -> dict:
    """
    כמה זמן נשאר עד החג וכמה חודשי חיסכון יש עד אליו.
    אם ניתן סכום - מחשב כמה צריך לחסוך בכל חודש (החישוב כאן, לא אצל המודל).
    """
    name = HOLIDAY_ALIASES.get(holiday.strip(), holiday.strip())
    if name not in HOLIDAYS:
        raise ToolError(f"החג '{holiday}' לא מוכר. חגים נתמכים: {', '.join(HOLIDAYS)}.")
    try:
        start = date.fromisoformat(today) if today else date.today()
    except ValueError as exc:
        raise ToolError("התאריך today צריך להיות בפורמט YYYY-MM-DD.") from exc

    heb = _next_occurrence(name, start)
    greg = heb.to_pydate()
    # חודשי חיסכון: מהחודש הבא ועד חודש החג (כולל). למשל מאוקטובר לאפריל = 6 חודשים
    months = (greg.year - start.year) * 12 + (greg.month - start.month)
    result = {
        "holiday": name,
        "hebrew_date": heb.hebrew_date_string(),
        "date": greg.isoformat(),
        "days_left": (greg - start).days,
        "saving_months": max(months, 0),
    }
    if months <= 0:
        result["note"] = "החג חל החודש - אין חודשים מלאים לחיסכון לפניו. כדאי לתכנן כבר את החג הבא."
    if amount is not None:
        try:
            amount = float(amount)
        except (TypeError, ValueError) as exc:
            raise ToolError("הסכום חייב להיות מספר.") from exc
        if amount <= 0:
            raise ToolError("הסכום חייב להיות חיובי.")
        result["amount"] = round(amount, 2)
        result["monthly_saving_needed"] = round(amount / max(months, 1), 2)
    return result


# ---------------------------------------------------------------------------
# הגדרת הכלים עבור OpenAI (function calling)
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "load_expenses": load_expenses,
    "load_budget": load_budget,
    "compare_to_budget": compare_to_budget,
    "get_transactions": get_transactions,
    "find_recurring": find_recurring,
    "simulate_savings_plan": simulate_savings_plan,
    "holiday_countdown": holiday_countdown,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "load_expenses",
            "description": "טוען קובץ CSV של הוצאות החודש. יש להפעיל לפני כל ניתוח.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_budget",
            "description": "טוען קובץ JSON עם ההכנסה החודשית ותקרת הוצאה לכל קטגוריה.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_budget",
            "description": "משווה הוצאה בפועל לתקציב בכל קטגוריה ומחזיר היכן הייתה חריגה.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transactions",
            "description": "מחזיר את העסקאות הגדולות בקטגוריה וקיבוץ לפי תיאור, כדי להבין ממה נבעה ההוצאה.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "top_n": {"type": "integer", "default": 5},
                },
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_recurring",
            "description": "מאתר מנויים ורכישות שחוזרות על עצמן לעיתים קרובות.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_savings_plan",
            "description": (
                "בודק תוכנית קיצוצים: מחשב את סך החיסכון, האם היעד הושג, "
                "ומחזיר בעיות (קיצוץ לא מציאותי, קטגוריה לא קיימת וכו'). "
                "אם target_reached=false יש לתקן את התוכנית ולבדוק שוב."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "number", "description": "יעד החיסכון החודשי בש\"ח"},
                    "cuts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "amount": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["category", "amount"],
                        },
                    },
                },
                "required": ["target", "cuts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "holiday_countdown",
            "description": (
                "מחשב לפי הלוח העברי כמה זמן נשאר עד חג וכמה חודשי חיסכון יש עד אליו. "
                "אם ניתן amount - מחזיר גם כמה צריך לחסוך בכל חודש (monthly_saving_needed). "
                "חגים: ראש השנה (או חגי תשרי), סוכות, חנוכה, פורים, פסח, שבועות."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "holiday": {"type": "string", "description": "שם החג בעברית"},
                    "amount": {"type": "number", "description": "הסכום שצריך לחג, בש\"ח"},
                },
                "required": ["holiday"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "מסיים את המשימה. success=true רק אם simulate_savings_plan החזיר target_reached=true. "
                "אם אין דרך מציאותית להגיע ליעד - success=false והסבר מדוע."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "report": {"type": "string", "description": "דוח סיכום למשתמש בעברית"},
                },
                "required": ["success", "report"],
            },
        },
    },
]


def run_tool(state: BudgetState, name: str, args: dict) -> dict:
    """מפעיל כלי לפי שם ותמיד מחזיר dict (גם במקרה של שגיאה)."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"כלי לא מוכר: {name}"}
    try:
        return func(state, **args)
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:  # ארגומנטים שגויים מהמודל
        return {"error": f"ארגומנטים לא תקינים ל-{name}: {exc}"}
