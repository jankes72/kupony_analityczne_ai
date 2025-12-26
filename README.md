# Kupony Analityczne AI

## Opis projektu

Projekt to eksperyment badawczy nad modelami predykcyjnymi dla zdarzeń sportowych zintegrowanymi z
instalacją danych i API. System łączy:

- eksperymenty z różnymi architekturami sieci neuronowych do predykcji wyników,
- moduł integrujący z zewnętrznym API (API-Sports) do zbierania danych meczowych (`app/sport_wrapper.py`),
- warstwę przechowywania i transformacji danych (SQLite -> Parquet) oraz narzędzia do feature engineering,
- serwer HTTP oparty na FastAPI z endpointami do monitoringu, pobierania danych, budowania datasetów i testów.

Całość pozwala na: zasysanie danych, tworzenie feature'ów, budowę datasetów Parquet i szybką ocenę modeli.

## Charakterystyka

- 🧭 Integracja z API-Sports (wrapper `ApiSportsHockey`) — pobieranie lig, meczów, eventów, standings,
- 🗄️ Lokalna baza SQLite do przechowywania zasysanych meczów (`hockey.sqlite`),
- 📦 Eksport datasetów do Parquet z gotowymi cechami (`nhl_2023.parquet` przykładowy plik),
- ⚡ FastAPI z endpointami: zdrowie, statystyki, monitoring systemu, endpointy operacyjne (`collect-world-data`, `fetch-and-store-season`, `build-dataset`),
- 🧩 Helpery do feature-engineering (`app/features_helpers.py`) — budowa targetów, normalizacja kursów, cechy formy i H2H,
- 🛠️ CLI w `app/sport_wrapper.py` do szybkich testów (pobieranie lig, fetch, budowa datasetu),
- 📚 Interaktywna dokumentacja OpenAPI dostępna pod `/docs` i `/redoc`.

## Wymagania

- Python 3.8+
- pip
- Zależności w `requirements.txt` (FastAPI, Uvicorn, pandas, requests, psutil, pydantic, itp.).

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

Serwer uruchomi się na `http://localhost:8000`.

## API Endpoints (szybki przegląd)

- `GET /` — informacje o API i lista endpointów
- `GET /health` — health-check
- `GET /stats` — przykładowe statystyki aplikacji
- `GET /monitor` — metryki systemowe (CPU, pamięć, uptime)
- `GET /settings` — zwraca statyczne ustawienia
- `POST /collect-world-data` — uniwersalny proxy do `ApiSportsHockey`
- `POST /fetch-and-store-season` — zasysa mecze dla podanego `league`+`season` i zapisuje do SQLite
- `POST /build-dataset` — buduje dataset Parquet z danych w SQLite (feature engineering)

Szczegółowe kontrakty request/response znajdują się niżej oraz w automatycznie wygenerowanej dokumentacji OpenAPI.

## Kontrakty endpointów (request / response)

### POST /collect-world-data

Request JSON (przykład):

```json
{
	"sports": {
		"api_key": "API_SPORTS_KEY",
		"action": "leagues|games|game|games_events|team_statistics",
		"params": { "league": 57, "season": 2024 }
	}
}
```

Response (przykład, zależy od akcji):

```json
{
	"result": [ /* array lub obiekt zwrócony przez ApiSports */ ]
}
```

### POST /fetch-and-store-season

Request JSON:

```json
{
	"api_key": "API_SPORTS_KEY",
	"league": 57,
	"season": 2024,
	"db_path": "./hockey.sqlite"
}
```

Response (przykład):

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

### POST /build-dataset

Request JSON:

```json
{
	"league": 57,
	"season": 2024,
	"db_path": "./hockey.sqlite",
	"output_path": "./nhl_2024.parquet",
	"return_file": false
}
```

Response (przykłady):

- Gdy `return_file` = `false`:

```json
{
	"ok": true,
	"parquet_path": "dataset_hockey_league57_season2024.parquet"
}
```

- Gdy `return_file` = `true` — endpoint zwraca plik Parquet jako download (`Content-Disposition`).

Uwagi: w przypadku błędów endpointy zwracają odpowiednie kody HTTP i pole `detail` z opisem.

## Struktura projektu

```
kupony_analityczne_ai/
├── app/
│   ├── __init__.py
│   ├── main.py              # Główna aplikacja FastAPI (endpointy i modele Pydantic)
│   ├── sport_wrapper.py     # Wrapper ApiSportsHockey, DB helpers, dataset builder i CLI
+│   ├── features_helpers.py  # Feature engineering helpers i GameRow dataclass
+│   └── README.md            # Dokumentacja modułów wewnątrz `app/`
├── run.py                   # Entry point uruchamiający Uvicorn
├── requirements.txt         # Zależności
├── .env.example             # Przykład zmiennych środowiskowych
├── hockey.sqlite            # (opcjonalnie) przykładowa baza danych SQLite
└── nhl_2023.parquet         # (opcjonalnie) przykładowy dataset Parquet
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

## Użycie wytrenowanego modelu (szybka ściąga)

Po uruchomieniu `app/example.py` w katalogu `models/` pojawią się przykładowe zapisy modeli:

- TensorFlow: `models/tf_demo_model/` (katalog z zapisanym modelem Keras)
- PyTorch: `models/torch_demo_model.pt` (pliki wag sieci)

Krótki przykład użycia (TensorFlow):

```python
import tensorflow as tf
model = tf.keras.models.load_model("models/tf_demo_model")
X_new = ...  # numpy array z cechami w tej samej kolejności co FEATURE_COLS
pred = model.predict(X_new)
```

Dla PyTorch:

```python
import torch
from app.example import build_torch_model
model = build_torch_model(input_dim=3, n_classes=1)
model.load_state_dict(torch.load("models/torch_demo_model.pt"))
model.eval()
X_new = torch.tensor([[...]], dtype=torch.float32)
with torch.no_grad():
	out = model(X_new)
	prob = torch.sigmoid(out).item()
```

Uwaga: wejście musi mieć tę samą kolejność i skalowanie cech co podczas treningu. Zapisz `FEATURE_COLS` i używaj go wszędzie.
