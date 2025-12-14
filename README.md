# Kupony Analityczne AI

## Opis projektu

Projekt jest eksperymentem oraz jednocześnie badaniem różnych struktur sieci neuronowych do predykcji i analizy danych predykcji i dynamicznego uczenia się na własnych błędach w czasie rzeczywistym.

## Charakterystyka

- 🧠 Eksperymenty z różnymi architekturami sieci neuronowych
- 📊 Analiza i predykcja danych w czasie rzeczywistym
- 🔄 Dynamiczne uczenie się na własnych błędach
- ⚡ API FastAPI z monitoringiem i statystykami
- 🔍 Endpoints do analizy i monitorowania systemu

## Wymagania

- Python 3.8+
- pip

## Instalacja

1. Klonuj repozytorium
```bash
git clone https://github.com/jankes72/kupony_analityczne_ai.git
cd kupony_analityczne_ai
```

2. Zainstaluj zależności
```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python run.py
```

Serwer uruchomi się na `http://localhost:8000`

## API Endpoints

- `GET /` - Informacje o API
- `GET /health` - Status zdrowia aplikacji
- `GET /stats` - Statystyki (requesty, sesje, czas odpowiedzi)
- `GET /monitor` - Monitorowanie zasobów (CPU, pamięć, uptime)
- `GET /settings` - Ustawienia aplikacji

## Kontrakty endpointów (request / response)

- `POST /collect-world-data` — uniwersalny endpoint do wywołań wrappera `ApiSportsHockey`.
	- Request JSON (przykład):

```json
{
	"sports": {
		"api_key": "API_SPORTS_KEY",
		"action": "leagues|games|game|games_events|team_statistics",
		"params": { "league": 57, "season": 2024 }
	}
}
```

	- Response (przykład, zależy od akcji):

```json
{
	"result": [ /* array lub obiekt zwrócony przez ApiSports */ ]
}
```

- `POST /fetch-and-store-season` — pobiera mecze z API i zapisuje do SQLite.
	- Request JSON:

```json
{
	"api_key": "API_SPORTS_KEY",
	"league": 57,
	"season": 2024,
	"db_path": "./hockey.sqlite"        
}
```

	- Response (przykład):

```json
{
	"ok": true,
	"summary": {
		"league": 57,
		"season": 2024,
		"fetched": 123,
		"db_path": "./hockey.sqlite"
	}
}
```

- `POST /build-dataset` — buduje dataset (feature engineering) z DB i zapisuje Parquet.
	- Request JSON:

```json
{
	"league": 57,
	"season": 2024,
	"db_path": "./hockey.sqlite",
	"output_path": "./nhl_2024.parquet",
	"return_file": false
}
```

	- Response (przykłady):

		- Gdy `return_file` = `false`:

```json
{
	"ok": true,
	"parquet_path": "dataset_hockey_league57_season2024.parquet"
}
```

		- Gdy `return_file` = `true` — endpoint zwraca plik Parquet jako download (`Content-Disposition`):
			bez JSON, bezpośrednio plik binarny.

Uwagi:
- Wszystkie POSTy zwracają odpowiednie kody HTTP w przypadku błędów (400/401/403/500) wraz z polem `detail` w treści odpowiedzi.
- Interaktywna specyfikacja (openapi) dostępna jest pod `/docs` i `/redoc` — tam znajdziesz dokładne schematy Pydantic.

## Dokumentacja API

Interaktywna dokumentacja dostępna jest na:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Struktura projektu

```
kupony_analityczne_ai/
├── app/
│   ├── __init__.py
│   └── main.py          # Główna aplikacja FastAPI
├── run.py               # Entry point
├── requirements.txt     # Zależności
├── .env.example        # Przykład zmiennych środowiskowych
└── README.md           # Ten plik
```

## Zmienne środowiskowe

Skopiuj `.env.example` na `.env` i dostosuj wartości:

```
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

## Licencja

MIT
