from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading.engine import BacktestResult
from trading.metrics import equity_curve, excess_equity_vs_baseline, infer_monthly_full_baseline_key


def fig_equity_comparison(results: dict[str, BacktestResult], *, title: str = "组合净值对比") -> go.Figure:
    fig = go.Figure()
    for _, r in results.items():
        v = equity_curve(r)
        fig.add_trace(go.Scatter(x=v.index, y=v.values, mode="lines", name=r.name))
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="净值",
        legend_title="场景",
        hovermode="x unified",
    )
    return fig


def fig_drawdown(results: dict[str, BacktestResult], *, title: str = "回撤（水下曲线）") -> go.Figure:
    fig = go.Figure()
    for _, r in results.items():
        v = equity_curve(r)
        rm = v.cummax()
        dd = v / rm - 1.0
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", name=r.name))
    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="回撤",
        legend_title="场景",
        hovermode="x unified",
    )
    return fig


def fig_excess_vs_baseline(
    strategy: BacktestResult,
    baseline: BacktestResult,
    *,
    title: str | None = None,
) -> go.Figure:
    ex = excess_equity_vs_baseline(strategy, baseline)
    t = title or f"{strategy.name} 相对 {baseline.name} 超额（净值比-1）"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex.index, y=ex.values, mode="lines", name="excess", fill="tozeroy"))
    fig.update_layout(title=t, xaxis_title="日期", yaxis_title="相对强弱")
    return fig


def fig_yearly_weights_stacked(yearly_weights: pd.DataFrame, *, title: str = "年度目标权重（阶跃）") -> go.Figure:
    cols = [c for c in yearly_weights.columns if c.endswith("_weight")]
    if not cols:
        fig = go.Figure()
        fig.update_layout(title=title + "（无数据）")
        return fig

    years = yearly_weights.index
    fig = go.Figure()
    for c in cols:
        fig.add_trace(
            go.Bar(x=years.astype(str), y=yearly_weights[c].values, name=c.replace("_weight", "")),
        )
    fig.update_layout(barmode="stack", title=title, xaxis_title="年", yaxis_title="权重")
    return fig


def fig_monthly_returns_heatmap(equity: pd.Series, *, title: str = "月度收益率热力图") -> go.Figure:
    daily = equity.pct_change().dropna()
    monthly = (1.0 + daily).resample("M").prod() - 1.0
    if monthly.empty:
        return go.Figure().update_layout(title=title + "（无数据）")
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    month_labels = ["M" + str(m) for m in pivot.columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="月收益"),
        ),
    )
    fig.update_layout(title=title, xaxis_title="月", yaxis_title="年")
    return fig


def fig_rolling_sharpe(
    result: BacktestResult,
    *,
    window: int = 252,
    title: str | None = None,
) -> go.Figure:
    v = equity_curve(result)
    daily = v.pct_change().dropna()
    rolling_mean = daily.rolling(window).mean()
    rolling_std = daily.rolling(window).std()
    rs = np.sqrt(252) * rolling_mean / rolling_std.replace(0, np.nan)
    t = title or f"{result.name} 滚动夏普（{window} 日）"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values, mode="lines", name="rolling_sharpe"))
    fig.update_layout(title=t, xaxis_title="日期", yaxis_title="夏普")
    return fig


def fig_summary_dashboard(
    results: dict[str, BacktestResult],
    strategy_key: str = "strategy",
    baseline_key: str | None = None,
) -> go.Figure:
    """多子图：净值 + 回撤 +（可选）超额。`baseline_key` 为 None 时用 `infer_monthly_full_baseline_key`。"""
    strat = results.get(strategy_key)
    bkey = baseline_key if baseline_key is not None else infer_monthly_full_baseline_key(results)
    base = results.get(bkey) if bkey is not None else None
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("净值", "回撤", "相对基准超额"),
        row_heights=[0.4, 0.3, 0.3],
    )
    for _, r in results.items():
        v = equity_curve(r)
        fig.add_trace(
            go.Scatter(x=v.index, y=v.values, mode="lines", name=r.name, legendgroup=r.name),
            row=1,
            col=1,
        )
    for _, r in results.items():
        v = equity_curve(r)
        dd = v / v.cummax() - 1.0
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd.values, mode="lines", name=r.name, showlegend=False, legendgroup=r.name),
            row=2,
            col=1,
        )
    if strat is not None and base is not None:
        ex = excess_equity_vs_baseline(strat, base)
        fig.add_trace(
            go.Scatter(x=ex.index, y=ex.values, mode="lines", name="excess", line=dict(color="black")),
            row=3,
            col=1,
        )
    fig.update_layout(height=900, title_text="回测总览", hovermode="x unified")
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", row=2, col=1)
    fig.update_yaxes(title_text="超额", row=3, col=1)
    return fig


def write_report_html(
    figures: list[tuple[str, go.Figure]],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = [
        "<html><head><meta charset='utf-8'><title>回测报告</title></head><body>",
        "<h1>回测报告</h1>",
    ]
    for i, (title, fig) in enumerate(figures):
        parts.append(f"<h2>{title}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_figure_image(fig: go.Figure, path: str | Path, *, width: int = 1200, height: int = 600) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=width, height=height)
