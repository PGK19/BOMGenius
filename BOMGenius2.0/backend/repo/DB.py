import sqlite3
import pandas as pd
from datetime import datetime

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATABASE = os.path.join(BASE_DIR, "bomgenius.db")


# -------------------------
# Initialize DB
# -------------------------
def init_db():

    conn = sqlite3.connect(DATABASE)
    print("[init_db] DB:", os.path.abspath(DATABASE))
    cursor = conn.cursor()

    # ---- Companies Table (LOGIN TABLE) ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            created_at TEXT,
            last_login TEXT,
            is_active INTEGER DEFAULT 1
        )
    """)

    # Ensure missing columns exist in case the table was created before
    cursor.execute("PRAGMA table_info(companies)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    if "last_login" not in existing_columns:
        cursor.execute("ALTER TABLE companies ADD COLUMN last_login TEXT")
    if "is_active" not in existing_columns:
        cursor.execute("ALTER TABLE companies ADD COLUMN is_active INTEGER DEFAULT 1")

    # ---- MBOM Table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mbom (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            Parent_Part_No TEXT,
            Child_Part_No TEXT,
            Description TEXT,
            Qty_Per REAL,
            UOM TEXT,
            BOM_Level INTEGER,
            Op_Sequence INTEGER,
            Work_Center TEXT,
            Make_Buy TEXT,
            Backflush_Ind TEXT,
            Scrap_Pct REAL,
            Plant TEXT,
            Bin_Location TEXT,
            Item_Type TEXT,
            Confidence_Score REAL,
            timestamp TEXT,
            FOREIGN KEY (company_id) REFERENCES companies(id)
        )
    """)

    conn.commit()
    conn.close()


# -------------------------
# Add missing columns dynamically
# -------------------------
def ensure_columns(conn, df):

    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(mbom)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    for column in df.columns:
        if column not in existing_columns:
            print(f"Adding missing column: {column}")
            cursor.execute(
                f'ALTER TABLE mbom ADD COLUMN "{column}" TEXT'
            )

    conn.commit()


# -------------------------
# Save MBOM safely
# -------------------------
def save_mbom(df, company_id):

    conn = sqlite3.connect(DATABASE)

    # Add timestamp
    timestamp = datetime.now().isoformat(timespec="seconds") 
    df["timestamp"] = timestamp

        # --- FIX: normalize Item_Type for dashboard pie ---
    # If engine gives "Node Type", copy it into Item_Type so pie chart works
    if "Item_Type" not in df.columns:
        if "Node Type" in df.columns:
            df["Item_Type"] = df["Node Type"]
        elif "Node_Type" in df.columns:
            df["Item_Type"] = df["Node_Type"]

    # Add company id
    df["company_id"] = company_id

    # Ensure columns exist
    ensure_columns(conn, df)

    # Save to DB
    df.to_sql(
        "mbom",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()
    conn.close()

    print("MBOM saved successfully with timestamp:", timestamp)
