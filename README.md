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
