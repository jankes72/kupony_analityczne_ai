from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union, Tuple
import requests
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from .features_helpers import (
    GameRow,
    GameStore,
    parse_dt,
    scores_side,
    make_training_record,
)

class SQLiteGameStore(GameStore):
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def team_games_before(self, *, team_id: int, league_id: int, season: int, before_date_utc: datetime) -> List[GameRow]:
        before_iso = before_date_utc.isoformat().replace("+00:00", "Z")
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT game_id, league_id, season, date_utc, status,
                       home_team_id, away_team_id, home_goals, away_goals
                FROM games
                WHERE league_id=? AND season=? AND date_utc < ?
                  AND (home_team_id=? OR away_team_id=?)
                ORDER BY date_utc ASC
            """, (league_id, season, before_iso, team_id, team_id)).fetchall()

        return [
            GameRow(
                game_id=r[0],
                league_id=r[1],
                season=r[2],
                date_utc=parse_dt(r[3]),
                status=r[4] or "NA",
                home_team_id=r[5],
                away_team_id=r[6],
                home_goals=r[7],
                away_goals=r[8],
            )
            for r in rows
        ]

    def h2h_games_before(self, *, team_a: int, team_b: int, league_id: int, season: int, before_date_utc: datetime) -> List[GameRow]:
        before_iso = before_date_utc.isoformat().replace("+00:00", "Z")
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT game_id, league_id, season, date_utc, status,
                       home_team_id, away_team_id, home_goals, away_goals
                FROM games
                WHERE league_id=? AND season=? AND date_utc < ?
                  AND (
                        (home_team_id=? AND away_team_id=?) OR
                        (home_team_id=? AND away_team_id=?)
                  )
                ORDER BY date_utc ASC
            """, (league_id, season, before_iso, team_a, team_b, team_b, team_a)).fetchall()

        return [
            GameRow(
                game_id=r[0],
                league_id=r[1],
                season=r[2],
                date_utc=parse_dt(r[3]),
                status=r[4] or "NA",
                home_team_id=r[5],
                away_team_id=r[6],
                home_goals=r[7],
                away_goals=r[8],
            )
            for r in rows
        ]


class ApiSportsError(RuntimeError):
    pass


@dataclass
class ApiSportsHockey:
    api_key: str
    base_url: str = "https://v1.hockey.api-sports.io"
    timeout: int = 30
    rate_limit_sleep: float = 0.0  # np. 0.25 jeśli trafiasz limity
    session: Optional[requests.Session] = None
    db_path: Optional[str] = None

    def __post_init__(self):
        """🔑 Inicjalizacja wrappera

        - Sprawdza czy podano `api_key` (z ENV lub argumentu).
        - Tworzy `requests.Session()` jeśli nie podano.
        """
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("Brak API key. Ustaw API_SPORTS_KEY w ENV lub przekaż w konstruktorze.")
        if self.session is None:
            self.session = requests.Session()
        if self.db_path:
            self._init_db()

    # ---------------------------
    # Core request
    # ---------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """🔎 Wykonuje GET do API

        Parametry:
        - `path`: ścieżka końcówki (np. `/leagues`)
        - `params`: słownik parametrów zapytania

        Zwraca: zdeserializowane JSON -> słownik z kluczami takimi jak `response`/`errors`.
        Podnosi `ApiSportsError` przy HTTP >=400 lub gdy API zwróci błąd.
        """
        if self.rate_limit_sleep:
            time.sleep(self.rate_limit_sleep)

        url = f"{self.base_url}{path}"
        headers = {"x-apisports-key": self.api_key}

        r = self.session.get(url, headers=headers, params=params or {}, timeout=self.timeout)
        # czasem API zwraca 4xx/5xx w JSONie, ale lepiej obsłużyć status
        if r.status_code >= 400:
            raise ApiSportsError(f"HTTP {r.status_code}: {r.text[:500]}")

        data = r.json()

        # typowa struktura: errors, response, results
        errors = data.get("errors") or []
        if isinstance(errors, dict) and errors:
            raise ApiSportsError(f"API errors: {errors}")
        if isinstance(errors, list) and len(errors) > 0:
            raise ApiSportsError(f"API errors: {errors}")

        return data

    # ---------------------------
    # Meta
    # ---------------------------
    def seasons(self) -> List[int]:
        """📅 Pobierz dostępne sezony

        Opcje: brak parametrów.
        Zwraca listę numerów sezonów (np. [2021, 2022, 2023]).
        """
        data = self._get("/seasons")
        return data.get("response", [])

    def countries(self, *, id: int = None, name: str = None, code: str = None, search: str = None) -> List[dict]:
        """🌍 Pobierz kraje

        Filtry (opcjonalne): `id`, `name`, `code`, `search`.
        Zwraca listę obiektów opisujących kraj.
        """
        params = self._clean_params(locals())
        data = self._get("/countries", params=params)
        return data.get("response", [])

    def leagues(
        self,
        *,
        id: int = None,
        name: str = None,
        country_id: int = None,
        country: str = None,
        type: str = None,   # "league" | "cup"
        season: int = None,
        search: str = None,
    ) -> List[dict]:
        """🏆 Pobierz ligi

        Filtry: `id`, `name`, `country_id`, `country`, `type` ("league"|"cup"), `season`, `search`.
        Przykład: `leagues(season=2024, search="NHL")` zwraca listę lig dla sezonu.
        """
        params = self._clean_params(locals())
        data = self._get("/leagues", params=params)
        return data.get("response", [])

    def teams(
        self,
        *,
        id: int = None,
        name: str = None,
        country_id: int = None,
        country: str = None,
        league: int = None,
        season: int = None,
        search: str = None,
    ) -> List[dict]:
        """👥 Pobierz drużyny

        Wymaga co najmniej jednego filtra: `id` lub `league`+`season`.
        Opcje: `id`, `name`, `country_id`, `country`, `league`, `season`, `search`.
        Zwraca listę obiektów drużyn.
        """
        params = self._clean_params(locals())
        if not params:
            raise ValueError("Endpoint /teams wymaga co najmniej jednego parametru (np. id albo league+season).")
        data = self._get("/teams", params=params)
        return data.get("response", [])

    # ---------------------------
    # Games
    # ---------------------------
    def games(
        self,
        *,
        id: int = None,
        date: str = None,      # "YYYY-MM-DD"
        league: int = None,
        season: int = None,
        team: int = None,
        timezone: str = None,
    ) -> List[dict]:
        """
        Pobierz mecze po filtrach. API wymaga co najmniej jednego parametru.
        Najczęściej: league+season albo date.
        """
        """🏒 Pobierz mecze

        Filtry: `id`, `date` ("YYYY-MM-DD"), `league`, `season`, `team`, `timezone`.
        Uwaga: API wymaga co najmniej jednego parametru (np. `league+season` albo `date`).
        Zwraca listę meczów.
        """
        params = self._clean_params(locals())
        if not params:
            raise ValueError("Endpoint /games wymaga co najmniej jednego parametru (np. id/date/league+season).")
        data = self._get("/games", params=params)
        return data.get("response", [])

    def game(self, game_id: int) -> dict:
        """🎯 Pobierz pojedynczy mecz po `game_id`

        Zwraca słownik z danymi meczu lub rzuca `ApiSportsError` jeśli brak.
        """
        items = self.games(id=game_id)
        if not items:
            raise ApiSportsError(f"Nie znaleziono meczu id={game_id}")
        return items[0]

    def games_h2h(
        self,
        *,
        team1_id: int,
        team2_id: int,
        date: str = None,      # "YYYY-MM-DD"
        league: int = None,
        season: int = None,
        timezone: str = None,
    ) -> List[dict]:
        """⚔️ Pobierz head-to-head (H2H) między dwoma drużynami

        Parametry: `team1_id`, `team2_id`, opcjonalnie `date`, `league`, `season`, `timezone`.
        Zwraca listę spotkań w formacie H2H.
        """
        params = self._clean_params(locals())
        # API chce h2h="id-id"
        h2h = f"{team1_id}-{team2_id}"
        params.pop("team1_id", None)
        params.pop("team2_id", None)
        params["h2h"] = h2h

        data = self._get("/games/h2h", params=params)
        return data.get("response", [])

    def games_events(self, *, game: int) -> List[dict]:
        """⚽ Pobierz wydarzenia meczu (gole, kary itp.)

        Parametr: `game` (id meczu).
        Zwraca listę eventów dla meczu.
        """
        data = self._get("/games/events", params={"game": game})
        return data.get("response", [])

    # ---------------------------
    # Standings / Stats
    # ---------------------------
    def standings(
        self,
        *,
        league: int,
        season: int,
        team: int = None,
        stage: str = None,
        group: str = None,
    ) -> List[list]:
        """📊 Pobierz tabelę/standingi

        Wymagane: `league`, `season`. Opcjonalnie `team`, `stage`, `group`.
        Zwraca strukturę tabeli dla ligi/sezonu.
        """
        params = self._clean_params(locals())
        data = self._get("/standings", params=params)
        return data.get("response", [])

    def standings_stages(self, *, league: int, season: int) -> List[str]:
        """🔢 Pobierz dostępne etapy (stages) w standings dla ligi i sezonu.

        Parametry: `league`, `season`.
        Zwraca listę identyfikatorów etapów/roundów.
        """
        data = self._get("/standings/stages", params={"league": league, "season": season})
        return data.get("response", [])

    def standings_groups(self, *, league: int, season: int) -> List[str]:
        """🔢 Pobierz dostępne grupy w standings dla ligi i sezonu.

        Parametry: `league`, `season`.
        Zwraca listę grup (np. "Group A").
        """
        data = self._get("/standings/groups", params={"league": league, "season": season})
        return data.get("response", [])

    def team_statistics(
        self,
        *,
        league: int,
        season: int,
        team: int,
        date: str = None,
    ) -> dict:
        """📈 Statystyki drużyny w sezonie

        Parametry: `league`, `season`, `team`, opcjonalnie `date`.
        Zwraca słownik ze statystykami drużyny.
        """
        params = self._clean_params(locals())
        data = self._get("/teams/statistics", params=params)
        return data.get("response", {})

    # ---------------------------
    # Utils
    # ---------------------------
    @staticmethod
    def _clean_params(d: Dict[str, Any]) -> Dict[str, Any]:
        """🧹 Usuń z lokalnego `locals()` niepotrzebne pola.

        - Usuwa `self` i wartości `None`.
        - Używane wewnętrznie przed przekazaniem `params` do `_get`.
        """
        out = {}
        for k, v in d.items():
            if k == "self":
                continue
            if v is None:
                continue
            out[k] = v
        return out
    
    # ===========================
    # DB (SQLite) helpers
    # ===========================

    def _conn(self) -> sqlite3.Connection:
        if not self.db_path:
            raise ValueError("db_path jest wymagany dla operacji DB (np. fetch_and_store_season/build_dataset).")
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id INTEGER PRIMARY KEY,
                    league_id INTEGER NOT NULL,
                    season INTEGER NOT NULL,
                    date_utc TEXT NOT NULL,
                    status TEXT,
                    home_team_id INTEGER NOT NULL,
                    away_team_id INTEGER NOT NULL,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    raw_json TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_league_season_date ON games(league_id, season, date_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_home ON games(home_team_id, date_utc);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_away ON games(away_team_id, date_utc);")

    @staticmethod
    def _parse_dt_utc(value: str) -> str:
        if not value:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _extract_score(scores: Any, side: str) -> Optional[int]:
        if not scores:
            return None
        v = scores.get(side) if isinstance(scores, dict) else None
        if v is None:
            return None
        if isinstance(v, int):
            return int(v)
        if isinstance(v, dict):
            if "total" in v and v["total"] is not None:
                return int(v["total"])
            for k in ("goals", "score", "points"):
                if k in v and v[k] is not None:
                    return int(v[k])
        return None

    def _normalize_game(self, game_json: Dict[str, Any], league_id: int, season: int) -> GameRow:
        game_id = int(game_json.get("id") or game_json.get("game", {}).get("id") or 0)
        if not game_id:
            raise ValueError("Nie udało się odczytać game_id z /games response")

        date_raw = (
            game_json.get("date")
            or game_json.get("time")
            or game_json.get("game", {}).get("date")
            or game_json.get("game", {}).get("time")
        )
        date_utc = parse_dt(date_raw)

        status = (game_json.get("status") or {}).get("short") or game_json.get("status") or "NA"
        status = str(status)

        teams = game_json.get("teams") or {}
        home_team_id = int((teams.get("home") or {}).get("id") or 0)
        away_team_id = int((teams.get("away") or {}).get("id") or 0)
        if not home_team_id or not away_team_id:
            raise ValueError("Brak home/away team_id w /games response")

        scores = game_json.get("scores") or game_json.get("score") or {}
        home_goals = scores_side(scores, "home")
        away_goals = scores_side(scores, "away")

        return GameRow(
            game_id=game_id,
            league_id=int(league_id),
            season=int(season),
            date_utc=date_utc,
            status=status,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
        )


    def fetch_and_store_season(self, *, league: int, season: int) -> Dict[str, Any]:
        if not self.db_path:
            raise ValueError("Ustaw db_path, np. ApiSportsHockey(..., db_path='./hockey.sqlite')")

        games = self.games(league=league, season=season)

        with self._conn() as conn:
            for g in games:
                gr = self._normalize_game(g, league_id=league, season=season)
                conn.execute("""
                    INSERT INTO games(
                        game_id, league_id, season, date_utc, status,
                        home_team_id, away_team_id, home_goals, away_goals, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(game_id) DO UPDATE SET
                        league_id=excluded.league_id,
                        season=excluded.season,
                        date_utc=excluded.date_utc,
                        status=excluded.status,
                        home_team_id=excluded.home_team_id,
                        away_team_id=excluded.away_team_id,
                        home_goals=excluded.home_goals,
                        away_goals=excluded.away_goals,
                        raw_json=excluded.raw_json
                """, (
                    gr.game_id,
                    gr.league_id,
                    gr.season,
                    gr.date_utc.isoformat().replace("+00:00", "Z"),
                    gr.status,
                    gr.home_team_id,
                    gr.away_team_id,
                    gr.home_goals,
                    gr.away_goals,
                    json.dumps(g, ensure_ascii=False),
                ))

        return {
            "league": league,
            "season": season,
            "fetched": len(games),
            "db_path": self.db_path,
        }


    # ===========================
    # Dataset / Features -> Parquet
    # ===========================

    def _load_games_df(self, *, league: int, season: int) -> "pd.DataFrame":
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT game_id, league_id, season, date_utc, status,
                       home_team_id, away_team_id, home_goals, away_goals
                FROM games
                WHERE league_id=? AND season=?
                ORDER BY date_utc ASC
            """, (league, season)).fetchall()

        df = pd.DataFrame(rows, columns=[
            "game_id","league_id","season","date_utc","status",
            "home_team_id","away_team_id","home_goals","away_goals"
        ])
        if df.empty:
            return df
        df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True, errors="coerce")
        return df

    @staticmethod
    def _winrate(goals_for: List[int], goals_against: List[int]) -> Optional[float]:
        if not goals_for:
            return None
        wins = sum(1 for gf, ga in zip(goals_for, goals_against) if gf > ga)
        return wins / len(goals_for)

    def build_dataset(self, *, league: int, season: int, output_path: Optional[str] = None) -> str:
        if not self.db_path:
            raise ValueError("Ustaw db_path, żeby budować dataset.")

        store = SQLiteGameStore(self.db_path)

        # wczytaj wszystkie mecze sezonu jako GameRow (tylko core pola)
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT game_id, league_id, season, date_utc, status,
                    home_team_id, away_team_id, home_goals, away_goals
                FROM games
                WHERE league_id=? AND season=?
                ORDER BY date_utc ASC
            """, (league, season)).fetchall()

        if not rows:
            raise ValueError(f"Brak danych w DB dla league={league}, season={season}. Najpierw fetch_and_store_season().")

        games: List[GameRow] = [
            GameRow(
                game_id=r[0],
                league_id=r[1],
                season=r[2],
                date_utc=parse_dt(r[3]),
                status=r[4] or "NA",
                home_team_id=r[5],
                away_team_id=r[6],
                home_goals=r[7],
                away_goals=r[8],
            )
            for r in rows
        ]

        # build record per game using single source of truth: make_training_record(...)
        records: List[Dict[str, Any]] = []
        for g in games:
            rec = make_training_record(g, store)

            # flatten do kolumn
            flat: Dict[str, Any] = {}
            flat.update(rec["meta"])
            flat.update(rec["teams"])
            for k, v in rec["features"].items():
                flat[k] = v
            for k, v in rec["target"].items():
                flat[k] = v

            records.append(flat)

        df = pd.DataFrame(records)

        if not output_path:
            output_path = f"dataset_hockey_league{league}_season{season}.parquet"

        try:
            df.to_parquet(output_path, index=False)
        except Exception as e:
            raise ValueError(
                f"Nie mogę zapisać Parquet. Zainstaluj 'pyarrow' lub 'fastparquet'. "
                f"Original error: {e}"
            )

        return output_path


# ---------------------------
# Usage examples (CLI / local)
# ---------------------------
if __name__ == "__main__":
    """
    Szybkie testy wrappera bez FastAPI.

    Wymagane:
    - export API_SPORTS_KEY="..."
    - (opcjonalnie) pip install pandas pyarrow

    Przykład:
      python -m app.sport_wrapper
    """

    import os
    import argparse

    parser = argparse.ArgumentParser(description="API-Sports Hockey wrapper - CLI")
    parser.add_argument("--key", default=os.getenv("API_SPORTS_KEY", ""), help="API-Sports key (lub ENV API_SPORTS_KEY)")
    parser.add_argument("--db", default="./hockey.sqlite", help="Ścieżka do SQLite (default: ./hockey.sqlite)")
    parser.add_argument("--league", type=int, required=False, help="League ID (np. NHL)")
    parser.add_argument("--season", type=int, required=False, help="Season (4 cyfry, np. 2024)")
    parser.add_argument("--fetch", action="store_true", help="Zasysanie sezonu do DB (fetch_and_store_season)")
    parser.add_argument("--dataset", action="store_true", help="Budowa datasetu Parquet (build_dataset)")
    parser.add_argument("--out", default=None, help="Ścieżka wyjściowa Parquet (opcjonalnie)")
    parser.add_argument("--quick-leagues", action="store_true", help="Szybki test: pobierz ligi NHL dla sezonu 2024")
    args = parser.parse_args()

    if args.quick_leagues:
        if not args.key:
            raise SystemExit("Brak API key. Ustaw ENV API_SPORTS_KEY albo podaj --key.")
        api = ApiSportsHockey(api_key=args.key, rate_limit_sleep=0.15)
        leagues = api.leagues(season=2024, search="NHL")
        print("Leagues sample:", leagues[:1])
        raise SystemExit(0)

    # Operacje DB/dataset
    if args.fetch or args.dataset:
        if not args.league or not args.season:
            raise SystemExit("Dla --fetch/--dataset wymagane są --league oraz --season.")

    if args.fetch:
        if not args.key:
            raise SystemExit("Brak API key. Ustaw ENV API_SPORTS_KEY albo podaj --key.")

        api = ApiSportsHockey(api_key=args.key, rate_limit_sleep=0.15, db_path=args.db)
        summary = api.fetch_and_store_season(league=args.league, season=args.season)
        print("FETCH OK:", summary)

    if args.dataset:
        # build_dataset nie potrzebuje key do samego odczytu DB,
        # ale konstruktor wymaga niepustego api_key -> dajemy placeholder.
        api = ApiSportsHockey(api_key=args.key or "DUMMY_KEY", rate_limit_sleep=0.0, db_path=args.db)
        out_path = api.build_dataset(league=args.league, season=args.season, output_path=args.out)
        print("DATASET OK:", out_path)

    if not (args.quick_leagues or args.fetch or args.dataset):
        print("""
Brak akcji. Przykłady:

1) Test lig:
   python -m app.sport_wrapper --quick-leagues

2) Zasysanie sezonu do SQLite:
   python -m app.sport_wrapper --fetch --league 57 --season 2024 --db ./hockey.sqlite

3) Budowa datasetu do Parquet:
   python -m app.sport_wrapper --dataset --league 57 --season 2024 --db ./hockey.sqlite --out ./nhl_2024.parquet

Jeśli chcesz uruchomić serwer FastAPI, użyj pliku run.py
""")

