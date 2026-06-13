import os
import ollama
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict
import sqlite3
from datetime import datetime

import pandas as pd
from core.ebom_loader import load_ebom
from core.engine import generate_mbom
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from federated.local_trainer import export_local_updates, log_human_feedback
from ocr.ebom_from_image import ebom_from_image
from pydantic import BaseModel
from repo.DB import init_db, save_mbom
from fastapi import HTTPException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(BASE_DIR, "frontend")
DATABASE = os.path.join(BASE_DIR, "bomgenius.db")

c_id = -1

# C:\Users\Raja\OneDrive\Desktop\BOM_Project\BOMGenius2.0\federated\local_trainer.py

app = FastAPI(title="BOMGenius API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown (optional cleanup)
    print("API shutting down...")


# basemodels for request bodies
class LoginRequest(BaseModel):
    company_name: str
    company_id: str
    password: str


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    company_address: str
    password: str
    confirm_password: str


class SettingsRequest(BaseModel):
    company: str
    email: str


class SettingsData(BaseModel):
    company_name: str
    company_email: str
    password: str | None = None


@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_path, "index.html"))


# ================= GET SETTINGS =================
@app.get("/get-settings")
def get_settings():

    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT company_name, email
            FROM companies
            ORDER BY id DESC
            LIMIT 1
        """
        )

        row = cursor.fetchone()

        if not row:
            return {"company_name": "", "company_email": ""}

        return {"company_name": row["company_name"], "company_email": row["email"]}


# ================= SAVE SETTINGS =================
@app.post("/save-settings")
def save_settings(data: SettingsData):

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        if data.password:
            cursor.execute(
                """
                UPDATE companies
                SET company_name=?, email=?, password=?
                WHERE id = (SELECT id FROM companies ORDER BY id DESC LIMIT 1)
            """,
                (data.company_name, data.company_email, data.password),
            )
        else:
            cursor.execute(
                """
                UPDATE companies
                SET company_name=?, email=?
                WHERE id = (SELECT id FROM companies ORDER BY id DESC LIMIT 1)
            """,
                (data.company_name, data.company_email),
            )

        conn.commit()

    return {"message": "Settings updated successfully"}


@app.post("/userlogin")
def login(data: LoginRequest):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, company_name
        FROM companies
        WHERE email=? AND company_name=? AND password=?
    """,
        (data.company_id, data.company_name, data.password),
    )

    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    global c_id
    c_id = user[0]

    return {
        "message": "Login successful",
        "company_id": user[0],
        "company_name": user[1],
    }


@app.post("/register")
def register(data: RegisterRequest):

    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()

        # Check if email already exists
        cursor.execute("SELECT id FROM companies WHERE email=?", (data.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")

        cursor.execute(
            """
            INSERT INTO companies (company_name, email, password)
            VALUES (?, ?, ?)
        """,
            (
                data.full_name,
                data.email,
                data.password,
            ),
        )

        conn.commit()

    return {"message": "Registered successfully"}


@app.post("/forgot-password")
def forgot_password(email: str):
    return {"message": "Reset link sent"}


@app.post("/settings")
def save_settings(data: SettingsRequest):
    return {"message": "Settings saved"}


@app.post("/fullbomconverter")
async def fullbomconverter(
    ebom: UploadFile = File(...),
    inventory: Optional[UploadFile] = File(None),
):

    ebom_bytes = await ebom.read()
    ext = Path(ebom.filename).suffix.lower() if ebom.filename else ""

    if ext in [".png", ".jpg", ".jpeg"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(ebom_bytes)
            tmp_path = tmp.name
        ebom_df = ebom_from_image(tmp_path)
    else:
        ebom_df = load_ebom(ebom_bytes, ebom.filename)

    if inventory:
        inv_bytes = await inventory.read()
        inv_df = load_ebom(inv_bytes, inventory.filename)
    else:
        inv_df = pd.DataFrame()

    df_final = generate_mbom(ebom_df, inv_df)

    # -------------------------
    # CLEAN DATA BEFORE SAVE
    # -------------------------
    df_final = df_final.fillna("")

    for c in ["Parent Part Number", "Parent Description"]:
        if c in df_final.columns:
            df_final[c] = (
                df_final[c].astype(str).replace({"nan": "", "None": ""}).fillna("")
            )

    # Keep Confidence_Score numeric for DB + dashboard analytics
    if "Confidence_Score" in df_final.columns:
        df_final["Confidence_Score"] = pd.to_numeric(df_final["Confidence_Score"], errors="coerce").fillna(0.0)

    if "timestamp" in df_final.columns:
        df_final = df_final.drop(columns=["timestamp"])

    print(c_id)

    # -------------------------
    # SAVE TO DB
    # -------------------------
    save_mbom(df_final, c_id)

    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records"),
        "mode": (
            "WITH_INVENTORY"
            if (inventory and not inv_df.empty)
            else "WITHOUT_INVENTORY"
        ),
    }


@app.get("/dashboard/analytics")
def dashboard_analytics():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT MAX(timestamp) AS ts FROM mbom")
    ts = cur.fetchone()["ts"]

    if not ts:
        return {
            "composition": [],
            "confidence": {"high": 0, "medium": 0, "low": 0},
            "timestamp": None,
        }

    # Pie: Item type composition
    cur.execute(
        """
        SELECT COALESCE(Item_Type, 'Standard Parts') AS item_type, COUNT(*) AS cnt
        FROM mbom
        WHERE timestamp = ?
        GROUP BY COALESCE(Item_Type, 'Standard Parts')
        ORDER BY cnt DESC
    """,
        (ts,),
    )
    composition = [dict(r) for r in cur.fetchall()]

    # Bar: confidence buckets
    cur.execute(
        """
        SELECT
          SUM(CASE WHEN Confidence_Score >= 0.90 THEN 1 ELSE 0 END) AS high,
          SUM(CASE WHEN Confidence_Score >= 0.70 AND Confidence_Score < 0.90 THEN 1 ELSE 0 END) AS medium,
          SUM(CASE WHEN Confidence_Score < 0.70 THEN 1 ELSE 0 END) AS low
        FROM mbom
        WHERE timestamp = ?
    """,
        (ts,),
    )
    conf = dict(cur.fetchone())
    # Accuracy

    # Confidence + Accuracy Calculation
    cur.execute(
        """
    SELECT
      SUM(CASE WHEN Confidence_Score >= 0.90 THEN 1 ELSE 0 END) AS high,
      SUM(CASE WHEN Confidence_Score >= 0.70 AND Confidence_Score < 0.90 THEN 1 ELSE 0 END) AS medium,
      SUM(CASE WHEN Confidence_Score < 0.70 THEN 1 ELSE 0 END) AS low,
      COUNT(*) as total
    FROM mbom
    WHERE timestamp = ?
""",
        (ts,),
    )

    conf = dict(cur.fetchone())

    total = conf.get("total", 0) or 0
    high = conf.get("high", 0) or 0

    # Accuracy = High confidence matches / Total records
    accuracy = round((high / total) * 100, 2) if total > 0 else 0

    conn.close()
    return {
        "composition": composition,
        "confidence": conf,
        "timestamp": ts,
        "accuracy": accuracy,
    }


@app.post("/ebom/ocr")
async def ebom_from_image_api(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower() if file.filename else ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    df_ebom = ebom_from_image(tmp_path)

    return {"columns": list(df_ebom.columns), "rows": df_ebom.to_dict(orient="records")}


# (Imports and DATABASE definition cleaned up / moved to top)


@app.get("/mbom/history", response_model=List[Dict])
def get_mbom_history():
    """
    Returns MBOM history grouped by timestamp.
    Includes formatted date, time, and row count.
    """

    try:
        # Connect to SQLite database
        with sqlite3.connect(DATABASE) as conn:

            # Optional: return rows as dict-like objects
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT timestamp, COUNT(*) as total_rows
                FROM mbom
                GROUP BY timestamp
                ORDER BY timestamp DESC
                """
            )

            rows = cursor.fetchall()

        history = []

        for row in rows:

            ts = row["timestamp"]
            count = row["total_rows"]

            # Skip invalid timestamps
            if ts is None:
                continue

            # Convert timestamp safely
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts)
                else:
                    dt = datetime.fromisoformat(str(ts))
            except Exception:
                # fallback if format is different
                dt = datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")

            history.append(
                {
                    "timestamp": str(ts),
                    "date": dt.strftime("%d-%m-%Y"),
                    "time": dt.strftime("%I:%M %p"),
                    "rows": count,
                }
            )

        return history

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/mbom/by-timestamp/{ts}")
def get_mbom_by_timestamp(ts: str):

    import pandas as pd

    with sqlite3.connect(DATABASE) as conn:

        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()

        cursor.execute("SELECT * FROM mbom WHERE timestamp = ?", (ts,))

        rows = cursor.fetchall()

        if not rows:
            return {"columns": [], "rows": []}

        # Automatically get column names
        columns = [col[0] for col in cursor.description]

        # Convert to DataFrame safely
        df = pd.DataFrame(rows, columns=columns)

    return {"columns": columns, "rows": df.to_dict(orient="records")}


@app.get("/dashboard")
def get_dashboard():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # -------------------------
    # Total BOMs
    # -------------------------
    cursor.execute(
        """
        SELECT COUNT(*) AS total_boms
        FROM mbom
        """
        )
    total_boms = round(cursor.fetchone()["total_boms"]/2) or 0

    # -------------------------
    # Total Components
    # -------------------------
    cursor.execute(
        """
        SELECT COUNT(*) AS total_components
        FROM mbom
    """
    )
    total_components = cursor.fetchone()["total_components"]

    # -------------------------
    # Latest Upload
    # -------------------------
    cursor.execute(
        """
        SELECT MAX(timestamp) AS last_uploaded
        FROM mbom
    """
    )
    last_uploaded = cursor.fetchone()["last_uploaded"]

    # -------------------------
    # Avg Components per BOM
    # -------------------------
    cursor.execute(
        """
        SELECT ROUND(AVG(component_count), 2) AS avg_components
        FROM (
        SELECT Parent_Part_No, COUNT(*) AS component_count
        FROM mbom
        WHERE company_id = ?
        GROUP BY Parent_Part_No
        ) t
        """
    , (c_id,))
    avg_components = cursor.fetchone()["avg_components"] or 0

    # -------------------------
    # Accuracy Calculation
    # -------------------------
    cursor.execute(
        """
    SELECT COUNT(*) as total,
           SUM(CASE WHEN Confidence_Score >= 0.90 THEN 1 ELSE 0 END) as high
    FROM mbom
    WHERE Confidence_Score IS NOT NULL
"""
    )

    row = cursor.fetchone()
    total = row["total"] or 0
    high = row["high"] or 0

    accuracy = round((high / total) * 100, 2) if total > 0 else 0

    # -------------------------
    # Consumable Breakdown
    # -------------------------
    cursor.execute(
        """
        SELECT Consumables, COUNT(*) as count
        FROM mbom
        WHERE Consumables IS NOT NULL
              AND TRIM(Consumables) != ''
              AND LOWER(Consumables) != 'na'
        GROUP BY Consumables
    """
    )

    rows = cursor.fetchall()

    consumable_breakdown = [
        {"type": row["Consumables"], "count": row["count"]} for row in rows
    ]

    # -------------------------
    # Avg Confidence Score
    # -------------------------
    cursor.execute(
        """
        SELECT ROUND(AVG(Confidence_Score), 2) AS avg_confidence
        FROM mbom
        WHERE Confidence_Score IS NOT NULL
    """
    )

    avg_confidence = cursor.fetchone()["avg_confidence"]

    conn.close()
    return {
        "total_boms": total_boms or 0,
        "total_components": total_components or 0,
        "last_uploaded": last_uploaded,
        "avg_components": avg_components or 0,
        "avg_confidence": avg_confidence or 0,
        "accuracy": accuracy,  # 👈 IMPORTANT
        "consumable_breakdown": consumable_breakdown,
    }


from pydantic import BaseModel
from typing import Optional


class Feedback(BaseModel):
    ebom_part: str
    ai_matched_part: str
    correct_part: str


@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    log_human_feedback(
        {"ebom_part": feedback.ebom_part, "correct_part": feedback.correct_part}
    )
    return {"status": "feedback recorded"}


@app.get("/federated/export")
def federated_export():
    return export_local_updates()


@app.post("/federated/import")
def federated_import(global_rules: dict):
    import json

    os.makedirs("federated", exist_ok=True)
    with open("federated/global_rules.json", "w") as f:
        json.dump(global_rules, f, indent=2)
    return {"status": "global rules updated"}


class CompanyCreate(BaseModel):
    name: str


@app.post("/company")
def create_company(data: CompanyCreate):

    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            "INSERT INTO companies (company_name, created_at) VALUES (?, ?)",
            (data.name, datetime.now().isoformat()),
        )

    return {"status": "company created"}


@app.get("/company")
def get_companies():
    with sqlite3.connect(DATABASE) as conn:
        rows = conn.execute(
            "SELECT id, company_name, created_at, last_login, is_active FROM companies"
        ).fetchall()

    return [
        {
            "id": r[0],
            "name": r[1],
            "created_at": r[2],
            "last_login": r[3],
            "is_active": r[4],
        }
        for r in rows
    ]


class CompanyUpdate(BaseModel):
    name: str


@app.put("/company/{cid}")
def update_company(cid: int, data: CompanyUpdate):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("UPDATE companies SET company_name=? WHERE id=?", (data.name, cid))

    return {"status": "updated"}


@app.delete("/company/{cid}")
def delete_company(cid: int):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("DELETE FROM companies WHERE id=?", (cid,))

    return {"status": "deleted"}


@app.patch("/company/{cid}/status")
def toggle_company_status(cid: int):
    with sqlite3.connect(DATABASE) as conn:
        cur = conn.cursor()

        cur.execute("SELECT is_active FROM companies WHERE id=?", (cid,))
        row = cur.fetchone()

        if not row:
            return {"error": "company not found"}

        current = row[0]
        new_status = 0 if current == 1 else 1

        cur.execute("UPDATE companies SET is_active=? WHERE id=?", (new_status, cid))

    return {"status": "changed", "is_active": new_status}