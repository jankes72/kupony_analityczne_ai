from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import psutil
import os
from typing import Any, Dict, Optional
from fastapi.responses import FileResponse

from .generator_synthetic_data import SyntheticMatchGeneratorV2
import tempfile
import json

app = FastAPI(
    title="Kupony Analityczne AI",
    description="API do analizy kuponów",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Stats(BaseModel):
    total_requests: int
    active_sessions: int
    avg_response_time: float
    last_update: datetime

class Monitor(BaseModel):
    cpu_usage: float
    memory_usage: float
    memory_available: int
    uptime: float
    timestamp: datetime

class Settings(BaseModel):
    debug: bool
    host: str
    port: int
    cors_enabled: bool
    version: str


# Contract for /collect-world-data
class SportsContract(BaseModel):
    api_key: str
    action: str
    params: Dict[str, Any] = {}


class CollectRequest(BaseModel):
    # special key in contract that contains wrapper config and params
    sports: SportsContract

class FetchSeasonRequest(BaseModel):
    api_key: str
    league: int
    season: int
    db_path: Optional[str] = None


class BuildDatasetRequest(BaseModel):
    league: int
    season: int
    db_path: Optional[str] = None
    output_path: Optional[str] = None
    return_file: bool = False  # jak True -> zwróci plik Parquet


@app.get("/")
async def root():
    """🏠 Root endpoint

    Zwraca krótkie info o API i listę podstawowych endpointów.
    """
    return {
        "message": "Kupony Analityczne AI API",
        "version": "1.0.0",
        "endpoints": ["/", "/health", "/stats", "/monitor", "/settings"]
    }

@app.get("/health")
async def health_check():
    """✅ Health check

    Prosty endpoint do sprawdzenia dostępności serwisu.
    Zwraca `status: ok` jeśli aplikacja działa.
    """
    return {"status": "ok"}

@app.get("/stats", response_model=Stats)
async def get_stats():
    """📈 Stats

    Zwraca przykładowe statystyki aplikacji:
    - `total_requests`, `active_sessions`, `avg_response_time`, `last_update`.
    (🔧 można rozszerzyć o rzeczywiste liczniki)
    """
    return {
        "total_requests": 42,
        "active_sessions": 3,
        "avg_response_time": 0.125,
        "last_update": datetime.now()
    }

@app.get("/monitor", response_model=Monitor)
async def get_monitor():
    """🖥️ Monitor systemu

    Zwraca metryki systemowe: `cpu_usage`, `memory_usage`, `memory_available`, `uptime`, `timestamp`.
    Używane do szybkiego monitoringu środowiska aplikacji.
    """
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu_usage": cpu,
        "memory_usage": memory.percent,
        "memory_available": memory.available,
        "uptime": os.popen("uptime -p").read().strip(),
        "timestamp": datetime.now()
    }

@app.get("/settings", response_model=Settings)
async def get_settings():
    """⚙️ Ustawienia aplikacji (statyczne)

    Zwraca przykładowe ustawienia: `debug`, `host`, `port`, `cors_enabled`, `version`.
    """
    return {
        "debug": True,
        "host": "0.0.0.0",
        "port": 8000,
        "cors_enabled": True,
        "version": "1.0.0"
    }


@app.post("/collect-world-data")
async def collect_world_data(payload: CollectRequest):
    sc = payload.sports

    """🌐 Endpoint integrujący ApiSportsHockey

    Kontrakt (JSON):
    {
        "sports": {
            "api_key": "API_KEY_HERE",
            "action": "leagues|games|game|games_events|team_statistics",
            "params": { ... }
        }
    }
    """
    # init wrapper (will raise ValueError when api_key is missing/empty)
    try:
        from .sport_wrapper import ApiSportsHockey, ApiSportsError
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing or broken sport_wrapper module")

    try:
        api = ApiSportsHockey(api_key=sc.api_key, rate_limit_sleep=0.15)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    action = sc.action
    params = sc.params or {}

    try:
        if action == "leagues":
            result = api.leagues(**params)
        elif action == "games":
            result = api.games(**params)
        elif action == "game":
            game_id = params.get("game_id") or params.get("id")
            if not game_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing game_id/id in params")
            result = api.game(int(game_id))
        elif action == "games_events":
            game = params.get("game") or params.get("game_id") or params.get("id")
            if not game:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing game in params")
            result = api.games_events(game=int(game))
        elif action == "team_statistics":
            result = api.team_statistics(**params)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown action: {action}")
    except ApiSportsError as e:
        msg = str(e)
        if "HTTP 401" in msg or "401" in msg or "Unauthorized" in msg or "invalid" in msg.lower():
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"External API auth error: {msg}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=msg)

    return {"result": result}

@app.post("/fetch-and-store-season")
async def fetch_and_store_season(payload: FetchSeasonRequest):
    """
    Zasysa wszystkie mecze dla (league, season) i zapisuje do SQLite.
    """
    try:
        from .sport_wrapper import ApiSportsHockey, ApiSportsError
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing or broken sport_wrapper module")

    try:
        api = ApiSportsHockey(api_key=payload.api_key, rate_limit_sleep=0.15, db_path=payload.db_path)
        summary = api.fetch_and_store_season(league=payload.league, season=payload.season)
        return {"ok": True, "summary": summary}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ApiSportsError as e:
        msg = str(e)

        # auth problems
        if "HTTP 401" in msg or "Unauthorized" in msg or "invalid" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"External API auth error: {msg}"
            )

        # plan limits / forbidden seasons etc.
        if "Free plans do not have access" in msg or "'plan'" in msg or "plan" in msg.lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Plan limit: {msg}"
            )

        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=msg)



@app.post("/build-dataset")
async def build_dataset(payload: BuildDatasetRequest):
    """
    Buduje dataset z DB i zapisuje Parquet.
    """
    try:
        # potrzebujemy pandas do operacji na DF / Parquet
        try:
            import pandas as pd #type: ignore
        except Exception:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Missing dependency: pandas. Install with 'pip install pandas pyarrow' to enable Parquet export.")

        # wczytaj/zbuduj dataset z DB przy użyciu ApiSportsHockey
        try:
            from .sport_wrapper import ApiSportsHockey
        except Exception:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing or broken sport_wrapper module")

        if not payload.db_path:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="`db_path` is required to build dataset from DB")

        api = ApiSportsHockey(api_key="DUMMY_KEY_NOT_USED_FOR_DATASET", rate_limit_sleep=0.0, db_path=payload.db_path)

        # najpierw stwórz bazowy parquet z DB do tymczasowej ścieżki
        fd, tmp_base = tempfile.mkstemp(suffix=".parquet")
        os.close(fd)
        try:
            base_path = api.build_dataset(league=payload.league, season=payload.season, output_path=tmp_base)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to build base dataset: {e}")

        # wczytaj dataframe z DB
        try:
            base_df = pd.read_parquet(base_path)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Can't read base parquet: {e}")

        augmented = []

        def _to_base_record(row: dict) -> dict:
            # mapujemy dostępne pola do schematu generatora, z bezpiecznymi defaultami
            return {
                "league": row.get("league", "DB League"),
                "season": str(row.get("season", payload.season)),
                "home_team": row.get("home_team") or f"team_{row.get('home_team_id', '')}",
                "away_team": row.get("away_team") or f"team_{row.get('away_team_id', '')}",
                "closing_odds": {"home": row.get("closing_odds_home", 2.0), "draw": row.get("closing_odds_draw", 10.0), "away": row.get("closing_odds_away", 3.0)},
                "final_result": row.get("final_result", "draw"),
                "form_home": float(row.get("form_home", 0.6)),
                "form_away": float(row.get("form_away", 0.6)),
                "sentiment": float(row.get("sentiment", 0.5)),
                "faceoff_win_home": int(row.get("faceoff_win_home", 50)),
                "shots_home": int(row.get("shots_home", max(0, int(row.get("shots_home", 25))))),
                "shots_away": int(row.get("shots_away", max(0, int(row.get("shots_away", 25))))),
                "powerplays_home": int(row.get("powerplays_home", 0)),
                "powerplays_away": int(row.get("powerplays_away", 0)),
                "penalty_minutes_home": int(row.get("penalty_minutes_home", 0)),
                "penalty_minutes_away": int(row.get("penalty_minutes_away", 0)),
                "goalie_sv_pct_home": float(row.get("goalie_sv_pct_home", 0.91)),
                "goalie_sv_pct_away": float(row.get("goalie_sv_pct_away", 0.91)),
                "injuries_home": int(row.get("injuries_home", 0)),
                "injuries_away": int(row.get("injuries_away", 0)),
            }

        records = base_df.to_dict(orient="records")

        for rec in records:
            # dodajemy oryginalny rekord jako pierwszy
            augmented.append(rec)

            base = _to_base_record(rec)
            gen = SyntheticMatchGeneratorV2(base)
            synths = gen.generate()
            augmented.extend(synths)

        out_df = pd.DataFrame(augmented)

        # zapisz wynikowy parquet
        if payload.output_path:
            out_path = payload.output_path
        else:
            fd, tmp_out = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            out_path = tmp_out

        try:
            out_df.to_parquet(out_path, index=False)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to write output parquet: {e}")

        if payload.return_file:
            return FileResponse(path=out_path, media_type="application/octet-stream", filename=os.path.basename(out_path))

        return {"ok": True, "parquet_path": out_path, "base_rows": len(records), "rows": len(out_df)}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

