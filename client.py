#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path

import requests


def _pretty(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


class KuponyAIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def health(self) -> dict:
        r = requests.get(self._url("/health"), timeout=30)
        r.raise_for_status()
        return r.json()

    def stats(self) -> dict:
        r = requests.get(self._url("/stats"), timeout=30)
        r.raise_for_status()
        return r.json()

    def monitor(self) -> dict:
        r = requests.get(self._url("/monitor"), timeout=30)
        r.raise_for_status()
        return r.json()

    def settings(self) -> dict:
        r = requests.get(self._url("/settings"), timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_and_store_season(self, api_key: str, league: int, season: int, db_path: str) -> dict:
        payload = {
            "api_key": api_key,
            "league": league,
            "season": season,
            "db_path": db_path,
        }
        r = requests.post(self._url("/fetch-and-store-season"), json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    def build_dataset(
        self,
        league: int,
        season: int,
        db_path: str,
        output_path: str,
        return_file: bool = False,
        download_to: str | None = None,
    ):
        payload = {
            "league": league,
            "season": season,
            "db_path": db_path,
            "output_path": output_path,
            "return_file": return_file,
        }

        # Jeśli return_file = True, API może zwrócić binarny plik parquet
        r = requests.post(self._url("/build-dataset"), json=payload, timeout=600, stream=return_file)
        r.raise_for_status()

        if not return_file:
            return r.json()

        # return_file=True => zapisujemy plik
        if not download_to:
            download_to = Path(output_path).name  # np. nhl_2024.parquet
        download_path = Path(download_to)

        with download_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        return {"ok": True, "downloaded_to": str(download_path.resolve())}

    def collect_world_data(self, api_key: str, action: str, params: dict) -> dict:
        payload = {
            "sports": {
                "api_key": api_key,
                "action": action,
                "params": params,
            }
        }
        r = requests.post(self._url("/collect-world-data"), json=payload, timeout=600)
        r.raise_for_status()
        return r.json()


def main():
    p = argparse.ArgumentParser(description="Kupony Analityczne AI - API client")
    p.add_argument("--base-url", default=os.getenv("KUPO_URL", "http://localhost:8000"), help="np. http://localhost:8000")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    sub.add_parser("stats")
    sub.add_parser("monitor")
    sub.add_parser("settings")

    f = sub.add_parser("fetch-season")
    f.add_argument("--api-key", default=os.getenv("API_SPORTS_KEY"), required=False)
    f.add_argument("--league", type=int, required=True)
    f.add_argument("--season", type=int, required=True)
    f.add_argument("--db-path", default="./hockey.sqlite")

    b = sub.add_parser("build-dataset")
    b.add_argument("--league", type=int, required=True)
    b.add_argument("--season", type=int, required=True)
    b.add_argument("--db-path", default="./hockey.sqlite")
    b.add_argument("--output-path", default="./out.parquet")
    b.add_argument("--return-file", action="store_true", help="jeśli ustawisz, API zwróci plik (download)")
    b.add_argument("--download-to", default=None, help="gdzie zapisać pobrany plik, gdy --return-file")

    c = sub.add_parser("collect")
    c.add_argument("--api-key", default=os.getenv("API_SPORTS_KEY"), required=False)
    c.add_argument("--action", required=True, help="leagues|games|game|games_events|team_statistics")
    c.add_argument("--params", default="{}", help='JSON string, np. \'{"league":57,"season":2024}\'')

    args = p.parse_args()
    cli = KuponyAIClient(args.base_url)

    try:
        if args.cmd == "health":
            print(_pretty(cli.health()))
        elif args.cmd == "stats":
            print(_pretty(cli.stats()))
        elif args.cmd == "monitor":
            print(_pretty(cli.monitor()))
        elif args.cmd == "settings":
            print(_pretty(cli.settings()))
        elif args.cmd == "fetch-season":
            api_key = args.api_key
            if not api_key:
                raise SystemExit("Brak --api-key i brak API_SPORTS_KEY w env.")
            out = cli.fetch_and_store_season(api_key, args.league, args.season, args.db_path)
            print(_pretty(out))
        elif args.cmd == "build-dataset":
            out = cli.build_dataset(
                league=args.league,
                season=args.season,
                db_path=args.db_path,
                output_path=args.output_path,
                return_file=args.return_file,
                download_to=args.download_to,
            )
            print(_pretty(out))
        elif args.cmd == "collect":
            api_key = args.api_key
            if not api_key:
                raise SystemExit("Brak --api-key i brak API_SPORTS_KEY w env.")
            params = json.loads(args.params)
            out = cli.collect_world_data(api_key, args.action, params)
            print(_pretty(out))
        else:
            raise SystemExit(f"Nieznana komenda: {args.cmd}")
    except requests.HTTPError as e:
        # lepszy debug: pokaż body błędu, jeśli jest JSON
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                print("HTTP ERROR:", resp.status_code)
                print(_pretty(resp.json()))
            except Exception:
                print("HTTP ERROR:", resp.status_code)
                print(resp.text)
        else:
            print("HTTP ERROR:", str(e))
        sys.exit(2)


if __name__ == "__main__":
    main()
