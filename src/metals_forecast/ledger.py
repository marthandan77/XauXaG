"""Append-only forecast and outcome ledgers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .models import score_forecast


class ForecastLedger:
    def __init__(self, directory: str | Path = "data/ledger") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.forecasts_path = self.directory / "forecasts.jsonl"
        self.outcomes_path = self.directory / "outcomes.jsonl"

    def _read(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def forecasts(self) -> list[dict[str, Any]]:
        return self._read(self.forecasts_path)

    def outcomes(self) -> list[dict[str, Any]]:
        return self._read(self.outcomes_path)

    def append_forecast(self, record: dict[str, Any]) -> bool:
        """Append once by forecast_id; return False for an existing vintage."""
        existing = {str(row["forecast_id"]) for row in self.forecasts()}
        if str(record["forecast_id"]) in existing:
            return False
        with self.forecasts_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return True

    def score_matured(self, bars: pd.DataFrame) -> int:
        """Score each forecast once when its target session exists in completed bars."""
        existing = {str(row["forecast_id"]) for row in self.outcomes()}
        rows: list[dict[str, Any]] = []
        for forecast in self.forecasts():
            forecast_id = str(forecast["forecast_id"])
            target = pd.Timestamp(forecast["target_session"]).normalize()
            if forecast_id in existing or target not in bars.index:
                continue
            actual = bars.loc[target]
            result = score_forecast(forecast, float(actual.low), float(actual.close), float(actual.high))
            rows.append({
                "forecast_id": forecast_id,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "target_session": target.date().isoformat(),
                "model_name": forecast["model_name"],
                "horizon": forecast["horizon"],
                "orderflow_features_enabled": bool(forecast.get("orderflow_features_enabled", False)),
                "orderflow_rows_used": int(forecast.get("orderflow_rows_used", 0)),
                **result,
            })
        if rows:
            with self.outcomes_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        return len(rows)
