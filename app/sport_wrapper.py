from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union, Tuple
import requests


class ApiSportsError(RuntimeError):
    pass


@dataclass
class ApiSportsHockey:
    api_key: str
    base_url: str = "https://v1.hockey.api-sports.io"
    timeout: int = 30
    rate_limit_sleep: float = 0.0  # np. 0.25 jeśli trafiasz limity
    session: Optional[requests.Session] = None

    def __post_init__(self):
        """🔑 Inicjalizacja wrappera

        - Sprawdza czy podano `api_key` (z ENV lub argumentu).
        - Tworzy `requests.Session()` jeśli nie podano.
        """
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("Brak API key. Ustaw API_SPORTS_KEY w ENV lub przekaż w konstruktorze.")
        if self.session is None:
            self.session = requests.Session()

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


# ---------------------------
# Usage examples
# ---------------------------
if __name__ == "__main__":
    key = os.getenv("API_SPORTS_KEY", "WKLEJ_TUTAJ_KLUCZ")
    api = ApiSportsHockey(api_key=key, rate_limit_sleep=0.15)

    # 1) ligi dla sezonu 2024
    leagues = api.leagues(season=2024, search="NHL")
    print("Leagues:", leagues[:1])

    # 2) pobierz mecze ligi+sezon (historycznie)
    # Uwaga: league_id musisz wziąć z /leagues
    # games = api.games(league=57, season=2024)

    # 3) pobierz jeden mecz po ID
    # g = api.game(8279)

    # 4) wydarzenia meczu (gole/kary)
    # ev = api.games_events(game=8279)

    # 5) statystyki drużyny w sezonie
    # stats = api.team_statistics(league=57, season=2024, team=29)
