"""Read generation records from Supabase or the local JSON fallback."""
import json
import os

from .auth import LOCAL_USER
from typing import List

from .common import DATA_DIR

GEN_FILE = os.path.join(DATA_DIR, "generations.json")


def load_generations(since_iso: str, user_id: str) -> List[dict]:
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        try:
            from supabase import create_client
            res = (create_client(url, key).table("generations").select("*")
                   .eq("user_id", user_id).gte("created_at", since_iso).limit(10000).execute())
            return res.data or []
        except Exception:
            pass
    if os.path.exists(GEN_FILE):
        try:
            with open(GEN_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [r for r in data if str(r.get("created_at", "")) >= since_iso
                    and (r.get("user_id") == user_id
                         or (not r.get("user_id") and user_id == LOCAL_USER))]
        except Exception:
            return []
    return []
