import copy
from itertools import product


class SyntheticMatchGeneratorV2:
    """Generator syntetycznych danych meczów hokejowych."""

    HARD_FEATURES = {
        "league",
        "season",
        "home_team",
        "away_team",
        "closing_odds",
        "final_result",
    }

    SOFT_FEATURES_DEVIATION = {
        "form_home": [0.0, 0.02, -0.02],
        "form_away": [0.0, 0.02, -0.02],
        "sentiment": [0.0, 0.01, -0.01],
        "faceoff_win_home": [0, 1, -1],
        "shots_home": [0, 1, 2],
        "shots_away": [0, 1, 2],
        "powerplays_home": [0, 1],
        "powerplays_away": [0, 1],
        "penalty_minutes_home": [0, 2],
        "penalty_minutes_away": [0, 2],
        "goalie_sv_pct_home": [0.0, 0.005, -0.005],
        "goalie_sv_pct_away": [0.0, 0.005, -0.005],
        "injuries_home": [0, 1],
        "injuries_away": [0, 1],
    }

    BASE_RECORD_SCHEMA = {
        "league": str,
        "season": str,
        "home_team": str,
        "away_team": str,
        "closing_odds": dict,
        "final_result": str,
        "form_home": float,
        "form_away": float,
        "sentiment": float,
        "faceoff_win_home": int,
        "shots_home": int,
        "shots_away": int,
        "powerplays_home": int,
        "powerplays_away": int,
        "penalty_minutes_home": int,
        "penalty_minutes_away": int,
        "goalie_sv_pct_home": float,
        "goalie_sv_pct_away": float,
        "injuries_home": int,
        "injuries_away": int,
    }

    def __init__(self, base_record: dict):
        self.base = base_record
        self._validate_base_record(self.base)

    def _validate_base_record(self, record: dict) -> None:
        missing = [k for k in self.BASE_RECORD_SCHEMA.keys() if k not in record]
        if missing:
            raise ValueError(f"Missing fields in base_record: {missing}")

        for k, expected in self.BASE_RECORD_SCHEMA.items():
            val = record[k]
            if expected is float:
                if not isinstance(val, (float, int)):
                    raise ValueError(f"Field '{k}' should be float-like")
            elif expected is int:
                if not isinstance(val, int):
                    raise ValueError(f"Field '{k}' should be int")
            elif expected is dict:
                if not isinstance(val, dict):
                    raise ValueError(f"Field '{k}' should be dict")
            elif expected is str:
                if not isinstance(val, str):
                    raise ValueError(f"Field '{k}' should be str")

        if not (0 <= record["form_home"] <= 1):
            raise ValueError("'form_home' must be in [0,1]")
        if not (0 <= record["form_away"] <= 1):
            raise ValueError("'form_away' must be in [0,1]")
        if not (0 <= record["sentiment"] <= 1):
            raise ValueError("'sentiment' must be in [0,1]")
        if not (0 <= record["faceoff_win_home"] <= 100):
            raise ValueError("'faceoff_win_home' must be in [0,100]")
        if not (0 <= record["goalie_sv_pct_home"] <= 1):
            raise ValueError("'goalie_sv_pct_home' must be in [0,1]")
        if not (0 <= record["goalie_sv_pct_away"] <= 1):
            raise ValueError("'goalie_sv_pct_away' must be in [0,1]")
        if record["injuries_home"] < 0 or record["injuries_away"] < 0:
            raise ValueError("'injuries_*' must be >= 0")

        co = record.get("closing_odds", {})
        for ck in ("home", "draw", "away"):
            if ck not in co or not isinstance(co[ck], (int, float)):
                raise ValueError("'closing_odds' must contain numeric keys: home, draw, away")

        return None

    def _apply_correlations(self, record: dict) -> dict:
        """Proste korelacje hokejowe."""
        # kontuzje obniżają formę
        record["form_home"] = max(0.0, record["form_home"] - 0.05 * record["injuries_home"])
        record["form_away"] = max(0.0, record["form_away"] - 0.05 * record["injuries_away"])

        # faceoffs: bazowo 50, przesunięcie zależne od różnicy form
        record["faceoff_win_home"] = int(min(100, max(0, 50 + 20 * (record["form_home"] - record["form_away"]))))

        # strzały zależne od formy i faceoffów
        record["shots_home"] = max(0, int(25 + 15 * record["form_home"] + 0.1 * (record["faceoff_win_home"] - 50)))
        record["shots_away"] = max(0, int(25 + 15 * record["form_away"] - 0.1 * (record["faceoff_win_home"] - 50)))

        # powerplays i penalty minutes zachowujemy liczbę z rekordu
        record["powerplays_home"] = max(0, int(record.get("powerplays_home", 0)))
        record["powerplays_away"] = max(0, int(record.get("powerplays_away", 0)))

        # różnica w SV% wpływa na sentyment
        sv_adv = float(record["goalie_sv_pct_home"]) - float(record["goalie_sv_pct_away"]) 
        record["sentiment"] = min(1.0, max(0.0, 0.5 + 0.5 * (record["form_home"] - record["form_away"] + 0.5 * sv_adv)))

        return record

    def generate(self) -> list:
        synthetic = []

        deviation_keys = list(self.SOFT_FEATURES_DEVIATION.keys())
        deviation_values = [self.SOFT_FEATURES_DEVIATION[k] for k in deviation_keys]

        for deltas in product(*deviation_values):
            rec = copy.deepcopy(self.base)
            for key, delta in zip(deviation_keys, deltas):
                if key not in rec:
                    continue

                expected = self.BASE_RECORD_SCHEMA.get(key)
                if expected is float:
                    # stosujemy odchylenie i trzymamy w [0,1]
                    rec[key] = float(rec[key]) + float(delta)
                    rec[key] = min(max(0.0, rec[key]), 1.0)
                elif expected is int:
                    # dodajemy i zaokrąglamy w dół
                    rec[key] = int(max(0, int(rec[key]) + int(delta)))
                else:
                    # fallback: spróbuj dodać jeśli to liczba
                    try:
                        rec[key] = rec[key] + delta
                    except Exception:
                        pass

            rec = self._apply_correlations(rec)
            synthetic.append(rec)

        return synthetic
