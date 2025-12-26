"""Przykładowy plik demonstracyjny.

Zawiera proste helpery do konwersji DataFrame -> tensory (TensorFlow / PyTorch),
oraz przykład wczytania Parquet i przygotowania datasetu do trenowania.

Plik ma działać samodzielnie: dynamicznie importuje pandas / tensorflow / torch
jeżeli są dostępne. W przeciwnym wypadku zwraca instrukcję instalacji.

Krótkie instrukcje — użycie modelu po treningu
1) Co masz po treningu?
- TensorFlow: models/tf_demo_model/  (katalog z zapisanym modelem Keras)
- PyTorch: models/torch_demo_model.pt  (ważenie sieci)

2) UŻYCIE MODELU – TensorFlow (najprościej)
- Wczytanie modelu:
    import tensorflow as tf
    model = tf.keras.models.load_model("models/tf_demo_model")
- Przygotowanie danych (muszą mieć ten sam układ kolumn co przy treningu):
    import numpy as np
    X_new = np.array([[12, 8, 0.55]], dtype="float32")  # np. [shots_home, shots_away, faceoff_win_home]
- Predykcja:
    pred = model.predict(X_new)
    prob = float(pred[0][0])  # jeśli to model binarny

3) UŻYCIE MODELU – PyTorch
- Wczytanie modelu:
    import torch
    model = build_torch_model(input_dim=3, n_classes=1)  # zaimportuj/definiuj funkcję
    model.load_state_dict(torch.load("models/torch_demo_model.pt"))
    model.eval()
- Dane wejściowe:
    X_new = torch.tensor([[12, 8, 0.55]], dtype=torch.float32)
- Predykcja:
    with torch.no_grad():
            out = model(X_new)
            prob = torch.sigmoid(out).item()

4) Najczęstszy błąd (ważne)
- Nie: inna kolejność kolumn, inne skalowanie, inne typy danych.
- Tak: ta sama kolejność cech, to samo przetwarzanie, ten sam format.
    Sugerowane: zapisz listę cech i używaj jej wszędzie:
    FEATURE_COLS = ["shots_home", "shots_away", "faceoff_win_home"]

Te instrukcje są krótkie i praktyczne — wklej je do skryptu, który będzie używał
wytrenowanego modelu, aby zapewnić zgodność danych wejściowych.
"""

from typing import Tuple, List, Optional


def read_parquet(path: str):
    """Wczytuje Parquet do pandas.DataFrame. Podnosi czytelny błąd przy braku pandas."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise RuntimeError("Brakuje biblioteki pandas. Zainstaluj: pip install pandas pyarrow") from e
    return pd.read_parquet(path)


def df_to_tf_tensors(df, feature_cols: List[str], label_col: Optional[str] = None):
    """Konwertuje DataFrame do TensorFlow tensors (X, y).

    Zwraca: (X_tensor, y_tensor) — jeśli label_col jest None zwraca tylko X_tensor w krotce.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise RuntimeError("Brakuje TensorFlow. Zainstaluj: pip install tensorflow") from e

    X = df[feature_cols].values.astype("float32")
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

    if label_col is None:
        return (X_tensor,)

    y = df[label_col].values
    # automat: jeśli binarna -> float32, jeśli wieloklasowa -> int32
    if y.dtype.kind in "biu":
        y_tensor = tf.convert_to_tensor(y.astype("float32"), dtype=tf.float32)
    else:
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    return X_tensor, y_tensor


def df_to_torch_tensors(df, feature_cols: List[str], label_col: Optional[str] = None):
    """Konwertuje DataFrame do PyTorch tensors (X, y).

    Wymaga `torch` zainstalowanego.
    """
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise RuntimeError("Brakuje PyTorch. Zainstaluj: pip install torch") from e

    X = df[feature_cols].values.astype("float32")
    X_tensor = torch.from_numpy(X)

    if label_col is None:
        return (X_tensor,)

    y = df[label_col].values
    # domyślnie float tensor dla regresji/binarnej
    y_tensor = torch.from_numpy(y.astype("float32"))
    return X_tensor, y_tensor


def prepare_dataset(path: str, feature_cols: List[str], label_col: str, framework: str = "tf"):
    """Wczytuje Parquet i zwraca tensory dla wybranego frameworku ('tf' lub 'torch')."""
    df = read_parquet(path)
    if framework == "tf":
        return df_to_tf_tensors(df, feature_cols, label_col)
    elif framework == "torch":
        return df_to_torch_tensors(df, feature_cols, label_col)
    else:
        raise ValueError("framework musi być 'tf' lub 'torch'")


def build_tf_model(input_dim: int, n_classes: int = 1):
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise RuntimeError("Brakuje TensorFlow. Zainstaluj: pip install tensorflow") from e

    if n_classes == 1:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def train_tf_model(X_tensor, y_tensor, epochs: int = 5, batch_size: int = 32, save_path: Optional[str] = None):
    import tensorflow as tf

    # wykryj czy binarny czy wieloklasowy
    y_np = y_tensor.numpy() if hasattr(y_tensor, "numpy") else None
    n_classes = 1
    if y_np is not None:
        uniques = set(y_np.flatten().tolist())
        if len(uniques) > 2:
            n_classes = max(uniques) + 1 if all(isinstance(u, int) for u in uniques) else len(uniques)

    model = build_tf_model(X_tensor.shape[1], n_classes=n_classes)
    model.fit(X_tensor, y_tensor, epochs=epochs, batch_size=batch_size)

    if save_path:
        model.save(save_path)
    return model


def build_torch_model(input_dim: int, n_classes: int = 1):
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError as e:
        raise RuntimeError("Brakuje PyTorch. Zainstaluj: pip install torch") from e

    if n_classes == 1:
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    else:
        model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
            # softmax will be applied in loss (CrossEntropy)
        )
    return model


def train_torch_model(X_tensor, y_tensor, epochs: int = 5, batch_size: int = 32, save_path: Optional[str] = None):
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import TensorDataset, DataLoader  # type: ignore

    X = X_tensor
    y = y_tensor
    if isinstance(X, torch.Tensor) is False:
        X = torch.from_numpy(X)
    if isinstance(y, torch.Tensor) is False:
        y = torch.from_numpy(y)

    # determine classes
    uniques = torch.unique(y)
    n_classes = 1 if uniques.numel() <= 2 else int(uniques.max().item()) + 1

    model = build_torch_model(X.shape[1], n_classes=n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float() if n_classes == 1 else yb.to(device).long()
            optimizer.zero_grad()
            out = model(xb)
            if n_classes == 1:
                out = out.view(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(ds):.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == "__main__":
    # Rozszerzone demo:
    # - jeśli brak `out.parquet` spróbuj utworzyć mały przykładowy plik (wymaga pandas)
    # - preferuj TensorFlow, jeśli brak, spróbuj PyTorch
    # - wykonaj szybki trening (2 epoki) i zapisz model
    import os
    from pprint import pprint

    FEATURE_COLS = ["shots_home", "shots_away", "faceoff_win_home"]
    LABEL_COL = "home_win"
    sample_path = "out.parquet"

    def create_sample_parquet(path: str):
        try:
            import pandas as pd  # type: ignore
        except Exception:
            print("Nie można utworzyć przykładowego parquet — brak biblioteki pandas.")
            return False

        # prosty, mały dataset do szybkiego demo
        df = pd.DataFrame([
            {"shots_home": 12, "shots_away": 8, "faceoff_win_home": 0.55, "home_win": 1},
            {"shots_home": 7, "shots_away": 15, "faceoff_win_home": 0.40, "home_win": 0},
            {"shots_home": 10, "shots_away": 10, "faceoff_win_home": 0.50, "home_win": 1},
            {"shots_home": 5, "shots_away": 20, "faceoff_win_home": 0.35, "home_win": 0},
        ])
        df.to_parquet(path)
        print(f"Utworzono przykładowy plik: {path} (rows={len(df)})")
        return True

    # jeśli brak pliku, spróbuj utworzyć przykładowy
    if not os.path.exists(sample_path):
        created = create_sample_parquet(sample_path)
        if not created:
            print("Brak 'out.parquet' i nie udało się utworzyć przykładu. Uruchom endpoint build-dataset lub zainstaluj pandas.")
            raise SystemExit(0)

    # wczytaj kolumny i zweryfikuj cechy
    try:
        import pandas as pd  # type: ignore
        cols = pd.read_parquet(sample_path).columns
    except Exception as e:
        print("Nie można wczytać parquet:", e)
        raise SystemExit(1)

    feature_cols = [c for c in FEATURE_COLS if c in cols]
    label_col = LABEL_COL if LABEL_COL in cols else None

    if not feature_cols:
        print("Nie znaleziono znanych kolumn cech w parquet — podaj FEATURE_COLS w skrypcie.")
        raise SystemExit(1)

    # wybierz framework: preferuj TF
    framework = None
    try:
        import tensorflow as _tf  # type: ignore
        framework = "tf"
    except Exception:
        try:
            import torch  # type: ignore
            framework = "torch"
        except Exception:
            framework = None

    if framework is None:
        print("Brak TensorFlow i PyTorch — nie można trenować. Możesz tylko przygotować dane.")
        print("FEATURE_COLS:")
        pprint(feature_cols)
        raise SystemExit(0)

    print(f"Wybrany framework: {framework}")

    try:
        tensors = prepare_dataset(sample_path, feature_cols, label_col, framework=framework)
    except Exception as e:
        print("Błąd przy przygotowywaniu tensorów:", e)
        raise SystemExit(1)

    print("Przygotowano tensory:", [getattr(t, 'shape', None) for t in tensors])

    if label_col is None:
        print("Brak etykiety w danych — pomijam trening.")
        raise SystemExit(0)

    # krótki trening i zapis
    os.makedirs("models", exist_ok=True)
    if framework == "tf":
        X, y = tensors
        save_path = "models/tf_demo_model"
        try:
            model = train_tf_model(X, y, epochs=2, batch_size=16, save_path=save_path)
            print("Zapisano model TensorFlow:", save_path)

            # demonstracja ładowania i predykcji
            try:
                import numpy as np  # type: ignore
                import tensorflow as tf  # type: ignore
                loaded = tf.keras.models.load_model(save_path)
                X_new = np.array([[12, 8, 0.55]], dtype="float32")
                pred = loaded.predict(X_new)
                prob = float(pred[0][0])
                print("Przykładowa predykcja (TF) prob(home_win):", prob)
            except Exception as e:
                print("Nie udało się załadować/model predykcja po zapisie:", e)
        except Exception as e:
            print("Trening TF nie powiódł się:", e)
    else:
        X, y = tensors
        save_path = "models/torch_demo_model.pt"
        try:
            model = train_torch_model(X, y, epochs=2, batch_size=16, save_path=save_path)
            print("Zapisano model PyTorch:", save_path)

            # demonstracja ładowania i predykcji
            try:
                import torch  # type: ignore
                m = build_torch_model(input_dim=len(feature_cols), n_classes=1)
                m.load_state_dict(torch.load(save_path))
                m.eval()
                X_new = torch.tensor([[12, 8, 0.55]], dtype=torch.float32)
                with torch.no_grad():
                    out = m(X_new)
                    prob = torch.sigmoid(out).item()
                print("Przykładowa predykcja (Torch) prob(home_win):", prob)
            except Exception as e:
                print("Nie udało się załadować/model predykcja PyTorch:", e)
        except Exception as e:
            print("Trening PyTorch nie powiódł się:", e)