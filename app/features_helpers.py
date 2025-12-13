from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Core datamodel
# =========================

@dataclass(frozen=True)
class GameRow:
    game_id: int
    league_id: int
    season: int
    date_utc: datetime
    status: str

    home_team_id: int
    away_team_id: int

    home_goals: Optional[int]
    away_goals: Optional[int]


# =========================
# Parse helpers
# =========================

def parse_dt(value: str) -> datetime:
    """
    API-Sports zwykle ISO, czasem z 'Z'. Zwraca UTC datetime aware.
    """
    if not value:
        raise ValueError("Brak daty w rekordzie meczu.")
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def scores_side(scores: Any, side: str) -> Optional[int]:
    if not scores or not isinstance(scores, dict):
        return None
    v = scores.get(side)
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


# =========================
# Targets / labels
# =========================

def build_target(game: GameRow) -> Dict[str, Any]:
    went_ot = 1 if game.status in {"AOT", "AP"} else 0

    if game.home_goals is None or game.away_goals is None:
        return {
            "home_goals": None,
            "away_goals": None,
            "home_win": None,
            "home_win_regulation": None,
            "total_goals": None,
            "went_ot": None,
        }

    home_win = 1 if game.home_goals > game.away_goals else 0
    home_win_reg = 1 if (home_win == 1 and game.status == "FT") else 0
    total_goals = int(game.home_goals + game.away_goals)

    return {
        "home_goals": int(game.home_goals),
        "away_goals": int(game.away_goals),
        "home_win": int(home_win),
        "home_win_regulation": int(home_win_reg),
        "total_goals": int(total_goals),
        "went_ot": int(went_ot),
    }


# =========================
# Odds -> implied/fair
# =========================

def normalize_2way_odds(odds_home: float, odds_away: float) -> Dict[str, float]:
    if odds_home <= 1.0 or odds_away <= 1.0:
        raise ValueError("Kursy muszą być > 1.0")

    implied_home = 1.0 / float(odds_home)
    implied_away = 1.0 / float(odds_away)
    s = implied_home + implied_away
    overround = s - 1.0

    fair_p_home = implied_home / s
    fair_p_away = implied_away / s

    return {
        "odds_home": float(odds_home),
        "odds_away": float(odds_away),
        "implied_p_home": implied_home,
        "implied_p_away": implied_away,
        "overround": overround,
        "fair_p_home": fair_p_home,
        "fair_p_away": fair_p_away,
        "fair_odds_home": 1.0 / fair_p_home,
        "fair_odds_away": 1.0 / fair_p_away,
    }


# =========================
# Feature engineering
# =========================

def _team_goals(g: GameRow, team_id: int) -> Tuple[Optional[int], Optional[int]]:
    if g.home_goals is None or g.away_goals is None:
        return None, None
    if team_id == g.home_team_id:
        return g.home_goals, g.away_goals
    if team_id == g.away_team_id:
        return g.away_goals, g.home_goals
    return None, None


def compute_rest_days(team_games_before: List[GameRow], current_date_utc: datetime) -> Optional[int]:
    if not team_games_before:
        return None
    last_game = max(team_games_before, key=lambda x: x.date_utc)
    return int((current_date_utc.date() - last_game.date_utc.date()).days)


def compute_form_features(team_id: int, team_games_before: List[GameRow], window: int) -> Dict[str, Optional[float]]:
    games = [
        g for g in sorted(team_games_before, key=lambda x: x.date_utc, reverse=True)
        if g.home_goals is not None and g.away_goals is not None
    ][:window]

    if not games:
        return {
            f"form_last{window}_winrate": None,
            f"avg_goals_for_last{window}": None,
            f"avg_goals_against_last{window}": None,
            f"avg_goal_diff_last{window}": None,
        }

    wins = 0
    gf_sum = 0
    ga_sum = 0

    for g in games:
        gf, ga = _team_goals(g, team_id)
        if gf is None or ga is None:
            continue
        gf_sum += gf
        ga_sum += ga
        if gf > ga:
            wins += 1

    n = len(games)
    return {
        f"form_last{window}_winrate": wins / n,
        f"avg_goals_for_last{window}": gf_sum / n,
        f"avg_goals_against_last{window}": ga_sum / n,
        f"avg_goal_diff_last{window}": (gf_sum - ga_sum) / n,
    }


def compute_h2h_features(home_team_id: int, away_team_id: int, h2h_games_before: List[GameRow], last_n: int = 10) -> Dict[str, Optional[float]]:
    games = [
        g for g in sorted(h2h_games_before, key=lambda x: x.date_utc, reverse=True)
        if g.home_goals is not None and g.away_goals is not None
    ][:last_n]

    if not games:
        return {
            f"h2h_last{last_n}_home_winrate": None,
            f"h2h_last{last_n}_avg_goal_diff_home": None,
        }

    home_wins = 0
    diff_sum = 0.0

    for g in games:
        gf, ga = _team_goals(g, home_team_id)
        if gf is None or ga is None:
            continue
        diff_sum += (gf - ga)
        if gf > ga:
            home_wins += 1

    n = len(games)
    return {
        f"h2h_last{last_n}_home_winrate": home_wins / n,
        f"h2h_last{last_n}_avg_goal_diff_home": diff_sum / n,
    }


# =========================
# Store interface
# =========================

class GameStore:
    def team_games_before(self, *, team_id: int, league_id: int, season: int, before_date_utc: datetime) -> List[GameRow]:
        raise NotImplementedError

    def h2h_games_before(self, *, team_a: int, team_b: int, league_id: int, season: int, before_date_utc: datetime) -> List[GameRow]:
        raise NotImplementedError


# =========================
# Record builder (single source of truth)
# =========================

def make_training_record(
    game: GameRow,
    store: GameStore,
    *,
    timezone_name: str = "Europe/Warsaw",
    odds_2way: Optional[Dict[str, Any]] = None,
    standings_snapshot: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:

    home_hist = store.team_games_before(
        team_id=game.home_team_id, league_id=game.league_id, season=game.season, before_date_utc=game.date_utc
    )
    away_hist = store.team_games_before(
        team_id=game.away_team_id, league_id=game.league_id, season=game.season, before_date_utc=game.date_utc
    )
    h2h_hist = store.h2h_games_before(
        team_a=game.home_team_id, team_b=game.away_team_id, league_id=game.league_id, season=game.season, before_date_utc=game.date_utc
    )

    feats: Dict[str, Any] = {}
    feats["rest_days_home"] = compute_rest_days(home_hist, game.date_utc)
    feats["rest_days_away"] = compute_rest_days(away_hist, game.date_utc)

    feats.update({f"{k}_home": v for k, v in compute_form_features(game.home_team_id, home_hist, window=5).items()})
    feats.update({f"{k}_away": v for k, v in compute_form_features(game.away_team_id, away_hist, window=5).items()})

    feats.update({f"{k}_home": v for k, v in compute_form_features(game.home_team_id, home_hist, window=10).items()})
    feats.update({f"{k}_away": v for k, v in compute_form_features(game.away_team_id, away_hist, window=10).items()})

    feats.update(compute_h2h_features(game.home_team_id, game.away_team_id, h2h_hist, last_n=10))
    feats["is_home_advantage"] = 1

    if standings_snapshot:
        sh = standings_snapshot.get(game.home_team_id, {})
        sa = standings_snapshot.get(game.away_team_id, {})
        feats["season_rank_home"] = sh.get("rank")
        feats["season_rank_away"] = sa.get("rank")
        feats["season_points_home"] = sh.get("points")
        feats["season_points_away"] = sa.get("points")
    else:
        feats["season_rank_home"] = None
        feats["season_rank_away"] = None
        feats["season_points_home"] = None
        feats["season_points_away"] = None

    market: Optional[Dict[str, Any]] = None
    if odds_2way:
        market = {
            "bookmaker_id": odds_2way.get("bookmaker_id"),
            "market_type": "winner_2way",
            **normalize_2way_odds(float(odds_2way["odds_home"]), float(odds_2way["odds_away"]))
        }

    return {
        "meta": {
            "game_id": game.game_id,
            "league_id": game.league_id,
            "season": game.season,
            "date_utc": game.date_utc.isoformat().replace("+00:00", "Z"),
            "timezone": timezone_name,
            "status": game.status,
        },
        "teams": {"home_team_id": game.home_team_id, "away_team_id": game.away_team_id},
        "market": market,
        "features": feats,
        "target": build_target(game),
    }
