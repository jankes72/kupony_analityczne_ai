# app — moduły i szybka nawigacja

Ten plik opisuje moduły znajdujące się w katalogu `app/` oraz ich najważniejsze klasy, funkcje i przykłady użycia, żeby łatwiej poruszać się po kodzie.

- `app/__init__.py` — moduł inicjalizacyjny (pusty).

- `app/main.py` — serwer FastAPI z publicznymi endpointami i modelami Pydantic:
  - Endpointy:
    - `GET /` — informacje o API i lista endpointów.
    - `GET /health` — prosty health-check.
    - `GET /stats` — przykładowe statystyki (model `Stats`).
    - `GET /monitor` — metryki systemowe (model `Monitor`).
    - `GET /settings` — zwraca statyczne ustawienia (model `Settings`).
    - `POST /collect-world-data` — przyjmuje payload z kontraktem `CollectRequest` i wywołuje wrapper `ApiSportsHockey`.
    - `POST /fetch-and-store-season` — pobiera sezon przez wrapper i zapisuje do SQLite.
    - `POST /build-dataset` — buduje Parquet z DB i opcjonalnie zwraca plik.
  - Główne klasy / modele w pliku:
    - `Stats`, `Monitor`, `Settings` — modele Pydantic odpowiedzi.
    - `SportsContract`, `CollectRequest`, `FetchSeasonRequest`, `BuildDatasetRequest` — modele wejściowe.
  - Przykład użycia (lokalny): uruchom serwer przez `python run.py`, potem wywołaj endpointy lub sprawdź `/docs`.

- `app/sport_wrapper.py` — wrapper do API-Sports (Hockey) i narzędzia do DB / datasetów:
  - Główne klasy i funkcje:
    - `ApiSportsError` — wyjątek wrappera.
    - `ApiSportsHockey` — główny wrapper:
      - Inicjalizacja: `ApiSportsHockey(api_key, base_url=..., db_path=..., rate_limit_sleep=...)`.
      - Metody HTTP/zasoby: `seasons()`, `countries()`, `leagues()`, `teams()`, `games(...)`, `game(game_id)`, `games_h2h(...)`, `games_events(game)`, `standings(...)`, `team_statistics(...)`.
      - DB helpers: `_init_db()`, `_conn()`, `fetch_and_store_season(league, season)` — zapisuje tabele `games` w SQLite.
      - Dataset: `build_dataset(league, season, output_path)` — buduje rekordy feature-engineering i zapisuje Parquet.
      - CLI pod koniec pliku: szybkotesty (`--quick-leagues`, `--fetch`, `--dataset`).
    - `SQLiteGameStore` — implementacja `GameStore` oparta o SQLite używana przy budowie datasetu.
  - Wymagania/uwagi:
    - Wymaga `requests` oraz `pandas` (+ `pyarrow` lub `fastparquet` do zapisu Parquet).
    - Dla operacji DB należy podać `db_path` przy konstrukcji wrappera.
    - Metody normalizują dane do `GameRow` (z `features_helpers`).

- `app/features_helpers.py` — helpery do feature engineering i reprezentacja meczów:
  - `GameRow` (dataclass) — podstawowy model rekordu meczu (id, liga, sezon, daty, drużyny, bramki, status).
  - Parsowanie / pomocnicze:
    - `parse_dt(value)` — parsuje datę ISO -> aware UTC.
    - `scores_side(scores, side)` — bezpieczne wyciąganie wyniku dla danej strony.
  - Targety i normalizacja kursów:
    - `build_target(game)` — buduje etykiety/targety (home_win, total_goals, went_ot itp.).
    - `normalize_2way_odds(odds_home, odds_away)` — oblicza implied/fair odds i overround.
  - Feature engineering:
    - `compute_rest_days`, `compute_form_features`, `compute_h2h_features` — generowanie cech opartych na historii meczów.
    - `make_training_record(game, store, ...)` — "single source of truth" budujący kompletny rekord (meta, teams, features, target) używany potem do eksportu do Parquet.
  - Interfejs magazynu:
    - `GameStore` — abstrakcja (metody `team_games_before`, `h2h_games_before`) — implementowana przez `SQLiteGameStore`.

## Pliki danych (w katalogu root projektu)
- `hockey.sqlite` — przykładowa baza SQLite (jeśli jest dodana do repozytorium). Służy do testów local fetch/store.
- `nhl_2023.parquet` — przykładowy dataset Parquet wygenerowany przez `build_dataset`.

## Jak szybko znaleźć funkcję/metodę
- Szukaj po nazwie klasy/funkcji w repo: np. `ApiSportsHockey`, `make_training_record`, `build_dataset`.
- Przykładowe komendy:

```bash
# listuj pliki w app
ls -la app
# przeszukaj definicje
grep -n "class ApiSportsHockey" -n app/*.py
# otwórz fragment pliku
sed -n '1,200p' app/sport_wrapper.py
```

## Uruchamianie i testowanie
- Uruchom API lokalnie:

```bash
pip install -r requirements.txt
python run.py
# lub równolegle
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Szybkie sprawdzenie wrappera (lokalnie, bez FastAPI):

```bash
python -m app.sport_wrapper --quick-leagues --key "$API_SPORTS_KEY"
```

- Zasysanie sezonu do SQLite:

```bash
python -m app.sport_wrapper --fetch --league 57 --season 2024 --db ./hockey.sqlite --key "$API_SPORTS_KEY"
```

- Budowa datasetu Parquet:

```bash
python -m app.sport_wrapper --dataset --league 57 --season 2024 --db ./hockey.sqlite --out ./nhl_2024.parquet
```

---

Jeśli chcesz, mogę:
- dodać docstringi/komentarze w konkretnych funkcjach,
- zautomatyzować testy jednostkowe dla kluczowych funkcji,
- albo od razu zatwierdzić (`git add`/`commit`/`push`) nowy `app/README.md` do repo — daj znać, czy chcesz, żebym to zrobił automatycznie.
