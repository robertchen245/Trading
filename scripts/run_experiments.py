#!/usr/bin/env python3
"""从项目根目录运行: python scripts/run_experiments.py [--spec-file specs.json]"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from trading.experiment import rank_experiments, run_experiments
from trading.specs import StrategySpec, nl_to_strategy_spec, preset_strategy_specs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行策略实验并输出汇总")
    parser.add_argument("--spec-file", type=str, default=None, help="JSON 文件，格式为 StrategySpec 列表")
    parser.add_argument("--prompt", type=str, default=None, help="自然语言描述策略（MVP 解析）")
    args = parser.parse_args()

    specs = _load_specs(args.spec_file, args.prompt)
    runs, summary = run_experiments(specs)
    ranking = rank_experiments(summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = REPORTS_DIR / f"experiments_{ts}.summary.csv"
    ranking_path = REPORTS_DIR / f"experiments_{ts}.ranking.csv"
    summary.to_csv(summary_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    print(f"实验策略数: {len(runs)}")
    print(f"汇总 CSV: {summary_path}")
    print(f"排名 CSV: {ranking_path}")


def _load_specs(spec_file: str | None, prompt: str | None) -> list[StrategySpec]:
    if prompt:
        return [nl_to_strategy_spec(prompt, name="nl_prompt_strategy")]
    if spec_file:
        path = Path(spec_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("spec-file JSON 须为列表。")
        return [StrategySpec.from_dict(item) for item in payload]
    return preset_strategy_specs()


if __name__ == "__main__":
    main()
