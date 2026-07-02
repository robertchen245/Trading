from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trading.data import DataFetchConfig, DataFetchError, fetch_close_prices


class DataSourceTests(unittest.TestCase):
    def test_local_csv_source_loads_close_prices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "AAA.csv"
            path.write_text(
                "Date,Close\n"
                "2020-01-02,10\n"
                "2020-01-03,11\n",
                encoding="utf-8",
            )

            close = fetch_close_prices(
                ["AAA"],
                "2020-01-01",
                "2020-01-10",
                use_cache=False,
                data_source="local",
                local_data_dir=tmp,
            )

        self.assertEqual(list(close.columns), ["AAA"])
        self.assertEqual(float(close.iloc[-1]["AAA"]), 11.0)

    @patch("trading.data._fetch_symbol_from_stooq")
    @patch("trading.data._fetch_symbol_from_yfinance")
    def test_auto_falls_back_after_yfinance_failure(self, mock_yf, mock_stooq) -> None:
        mock_yf.side_effect = DataFetchError("rate limited")
        mock_stooq.return_value = pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
            name="SPY",
        )

        close = fetch_close_prices(
            ["SPY"],
            "2020-01-01",
            "2020-01-10",
            use_cache=False,
            data_source="auto",
        )

        self.assertEqual(float(close.iloc[-1]["SPY"]), 101.0)
        self.assertTrue(mock_yf.called)
        self.assertTrue(mock_stooq.called)

    def test_ibkr_source_is_valid_config(self) -> None:
        cfg = DataFetchConfig.from_env(source="ibkr")
        self.assertEqual(cfg.source, "ibkr")


if __name__ == "__main__":
    unittest.main()
