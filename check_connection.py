"""
בדיקת חיבור מהירה ל-Gemini.

הרצה:  python check_connection.py
הסקריפט בודק שלב אחרי שלב: שיש מפתח, שהמודל קיים, ושהמודל עונה ויודע להפעיל כלי.
אם משהו נכשל - הוא מדפיס מה בדיוק הבעיה ומה לעשות.
"""

import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from agent import DEFAULT_MODEL, GEMINI_BASE_URL, REASONING_EFFORT, _is_daily_quota  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    print("! הספרייה python-dotenv לא מותקנת. הריצו: pip install -r requirements.txt")

import openai  # noqa: E402
from openai import OpenAI  # noqa: E402


def fail(text: str) -> None:
    print(f"\n✗ {text}")
    sys.exit(1)


# 1. מפתח
key = os.getenv("GEMINI_API_KEY", "").strip()
if not (BASE_DIR / ".env").exists():
    fail("לא נמצא קובץ .env בתיקיית הפרויקט (שימו לב שהוא לא נשמר בטעות בשם .env.txt).")
if not key:
    fail("בקובץ .env אין ערך ל-GEMINI_API_KEY.")
if key.startswith(("\"", "'")) or " " in key:
    print("! במפתח יש מרכאות או רווחים - כדאי להסיר אותם.")
print(f"✓ נמצא מפתח: {key[:6]}...{key[-4:]} (אורך {len(key)})")

model = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
print(f"  מודל: {model}")

client = OpenAI(api_key=key, base_url=GEMINI_BASE_URL, timeout=60, max_retries=0)
extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}


def quota_fail(exc) -> None:
    if _is_daily_quota(exc):
        fail(f"נגמרה המכסה היומית החינמית של '{model}'. לכל מודל מכסה נפרדת - "
             "החליפו את GEMINI_MODEL בקובץ .env למודל אחר מהרשימה למעלה, או חכו למחר.")
    fail("יותר מדי בקשות בדקה האחרונה (429). חכו דקה ונסו שוב.")

# 2. רשימת מודלים (בודק גם שהמפתח תקין)
try:
    names = sorted(m.id.replace("models/", "") for m in client.models.list())
    print(f"✓ המפתח תקין, נמצאו {len(names)} מודלים.")
    skip = ("image", "tts", "audio", "live", "embedding", "veo", "imagen", "omni")
    flash = [n for n in names if "flash" in n and not any(s in n for s in skip)]
    print("  מודלי flash שאפשר לכתוב ב-GEMINI_MODEL:")
    for n in flash[:20]:
        print(f"     {n}")
    if model not in names:
        print(f"✗ המודל '{model}' לא נמצא ברשימה.")
        fail("העתיקו אחד מהשמות האלה לשורה GEMINI_MODEL בקובץ .env והריצו שוב.")
    print(f"✓ המודל '{model}' זמין.")
except openai.APITimeoutError:
    fail("אין תשובה מהשרת של Google תוך 30 שניות. בדקו את חיבור האינטרנט, חומת אש או VPN.")
except openai.APIConnectionError as exc:
    fail(f"אין חיבור לשרת של Google: {exc}. בדקו אינטרנט, חומת אש או VPN.")
except (openai.AuthenticationError, openai.BadRequestError, openai.PermissionDeniedError) as exc:
    fail(f"המפתח נדחה: {exc}")

# 3. שאלה פשוטה
try:
    t = time.time()
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "ענה במילה אחת: שלום"}],
                                       **extra)
    print(f"✓ המודל ענה תוך {time.time() - t:.1f} שניות: {r.choices[0].message.content!r}")
except openai.RateLimitError as exc:
    quota_fail(exc)
except openai.APITimeoutError:
    fail(f"המודל '{model}' לא ענה תוך 60 שניות. נסו מודל מהיר יותר מהרשימה למעלה (למשל עם lite בשם).")
except openai.APIError as exc:
    fail(f"שגיאה בשאלה פשוטה: {type(exc).__name__}: {exc}")

# 4. הפעלת כלי
try:
    tool = {"type": "function", "function": {
        "name": "add", "description": "מחבר שני מספרים",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"]}}}
    r = client.chat.completions.create(model=model, tools=[tool], tool_choice="auto", **extra,
                                       messages=[{"role": "user", "content": "כמה זה 2 ועוד 3? השתמש בכלי."}])
    calls = r.choices[0].message.tool_calls or []
    if calls:
        print(f"✓ המודל יודע להפעיל כלים: {calls[0].function.name}({calls[0].function.arguments})")
    else:
        print("! המודל ענה בלי להפעיל כלי. האג'נט עדיין יעבוד, אבל כדאי לנסות מודל אחר.")
except openai.RateLimitError as exc:
    quota_fail(exc)
except openai.APIError as exc:
    fail(f"שגיאה בהפעלת כלי: {type(exc).__name__}: {exc}")

print("\nהכול תקין. אפשר להריץ: python -m streamlit run app.py")
