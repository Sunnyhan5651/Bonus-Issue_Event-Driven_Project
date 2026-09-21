"""
무상증자 이벤트 스터디 백테스트
────────────────────────────────
events.csv / prices.csv / market.csv 를 읽어
AAR·CAAR·CAR 윈도우 분석 및 차트를 생성합니다.

설치: pip install pandas numpy statsmodels scipy matplotlib
실행: python backtest.py
"""

# ═══════════════════════════════════════════════
#  ⚙ 설정
# ═══════════════════════════════════════════════
EVENTS_PATH = "events.csv"
PRICES_PATH = "prices.csv"
MARKET_PATH = "market.csv"

# 컬럼명 매핑 — 보유 데이터 컬럼명과 다를 경우 우측 값만 수정하세요
COL_DATE         = "date"         # prices / market 날짜 컬럼
COL_STOCK_CODE   = "stock_code"   # prices 종목코드 컬럼
COL_CLOSE        = "close"        # prices 종가 컬럼
COL_MARKET_CLOSE = "close"        # market 종가 컬럼

EVENT_WINDOW      = (-10, 20)    # 이벤트 윈도우
ESTIMATION_WINDOW = (-120, -21)  # 추정 윈도우
MIN_EST_OBS       = 60           # 최소 추정 관측치
USE_LOG_RETURN    = True         # 로그수익률 사용

CAR_WINDOWS = [(-1, 1), (0, 1), (0, 3), (0, 5), (0, 10), (0, 20)]

OUTPUT_DIR  = "."   # 결과 파일 저장 경로
# ═══════════════════════════════════════════════

import os
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
def load_data():
    events = pd.read_csv(EVENTS_PATH)
    prices = pd.read_csv(PRICES_PATH)
    market = pd.read_csv(MARKET_PATH)

    events["event_date"]    = pd.to_datetime(events["event_date"])
    prices[COL_DATE]        = pd.to_datetime(prices[COL_DATE])
    market[COL_DATE]        = pd.to_datetime(market[COL_DATE])

    # 컬럼명 정규화 (내부 처리용)
    prices = prices.rename(columns={COL_DATE: "date", COL_STOCK_CODE: "stock_code", COL_CLOSE: "close"})
    market = market.rename(columns={COL_DATE: "date", COL_MARKET_CLOSE: "market_close"})

    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    events["stock_code"] = events["stock_code"].astype(str).str.zfill(6)

    logger.info(f"로드 완료 | 이벤트 {len(events)}건 | 주가 {len(prices)}행 | 시장 {len(market)}행")
    return events, prices, market


# ─────────────────────────────────────────
# 2. 수익률 계산
# ─────────────────────────────────────────
def compute_returns(prices: pd.DataFrame, market: pd.DataFrame):
    prices = prices.sort_values(["stock_code", "date"]).copy()
    market = market.sort_values("date").copy()

    if USE_LOG_RETURN:
        prices["ret"]     = prices.groupby("stock_code")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        market["mkt_ret"] = np.log(market["market_close"] / market["market_close"].shift(1))
    else:
        prices["ret"]     = prices.groupby("stock_code")["close"].pct_change()
        market["mkt_ret"] = market["market_close"].pct_change()

    return prices, market


# ─────────────────────────────────────────
# 3. 거래일 캘린더 + 이벤트 인덱스 부착
# ─────────────────────────────────────────
def build_calendar_and_attach(prices, market, events):
    common = sorted(set(prices["date"]).intersection(market["date"]))
    td = pd.DataFrame({"date": common})
    td["t_index"] = np.arange(len(td))

    prices = prices.merge(td, on="date", how="inner")
    market = market.merge(td, on="date", how="inner")

    em = td.rename(columns={"date": "event_date", "t_index": "event_t_index"})
    events = events.merge(em, on="event_date", how="left")

    missing = events["event_t_index"].isna().sum()
    if missing:
        logger.warning(f"거래일 캘린더 미매칭 {missing}건 → 제외")

    return prices, market, events


# ─────────────────────────────────────────
# 4. 마켓모델 추정 및 AR 계산 (이벤트 1건)
# ─────────────────────────────────────────
def _estimate_one(stock_df, market_df, event_t_index):
    df = stock_df.merge(
        market_df[["date", "t_index", "mkt_ret"]],
        on=["date", "t_index"], how="inner"
    ).copy()
    df["rel_day"] = df["t_index"] - event_t_index

    est = df[df["rel_day"].between(*ESTIMATION_WINDOW)].dropna(subset=["ret", "mkt_ret"])
    if len(est) < MIN_EST_OBS:
        return None

    model = sm.OLS(est["ret"], sm.add_constant(est["mkt_ret"])).fit()

    ev = df[df["rel_day"].between(*EVENT_WINDOW)].dropna(subset=["ret", "mkt_ret"]).copy()
    if ev.empty:
        return None

    ev["expected_ret"] = model.predict(sm.add_constant(ev["mkt_ret"]))
    ev["ar"] = ev["ret"] - ev["expected_ret"]

    return {
        "alpha":    model.params["const"],
        "beta":     model.params["mkt_ret"],
        "n_est":    len(est),
        "event_df": ev[["date", "t_index", "rel_day", "ret", "mkt_ret", "expected_ret", "ar"]].copy(),
    }


# ─────────────────────────────────────────
# 5. 전체 이벤트 처리
# ─────────────────────────────────────────
def run_event_study(events, prices, market):
    results, meta_rows = [], []
    skipped = 0

    for _, ev in events.iterrows():
        if pd.isna(ev.get("event_t_index")):
            skipped += 1
            continue
        sdf = prices[prices["stock_code"] == ev["stock_code"]].copy()
        if sdf.empty:
            skipped += 1
            continue

        out = _estimate_one(sdf, market, ev["event_t_index"])
        if out is None:
            skipped += 1
            continue

        edf = out["event_df"].copy()
        edf["stock_code"] = ev["stock_code"]
        edf["event_date"]  = ev["event_date"]
        results.append(edf)

        meta_rows.append({
            "stock_code": ev["stock_code"],
            "event_date": ev["event_date"],
            "alpha":      out["alpha"],
            "beta":       out["beta"],
            "n_est":      out["n_est"],
        })

    logger.info(f"이벤트 처리 완료: 유효 {len(results)}건 | 제외 {skipped}건")

    if not results:
        return None, None
    return pd.concat(results, ignore_index=True), pd.DataFrame(meta_rows)


# ─────────────────────────────────────────
# 6. AAR / CAAR 요약
# ─────────────────────────────────────────
def summarize_aar(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in panel.groupby("rel_day"):
        arr  = g["ar"].dropna().values
        t, p = stats.ttest_1samp(arr, 0.0) if len(arr) >= 2 else (np.nan, np.nan)
        rows.append({
            "rel_day": d,
            "N":       len(arr),
            "AAR":     np.mean(arr) if len(arr) else np.nan,
            "AAR_t":   t,
            "AAR_p":   p,
        })
    result = pd.DataFrame(rows).sort_values("rel_day").reset_index(drop=True)
    result["CAAR"] = result["AAR"].cumsum()
    return result


# ─────────────────────────────────────────
# 7. CAR 윈도우별 요약
# ─────────────────────────────────────────
def summarize_car(panel: pd.DataFrame, windows=CAR_WINDOWS) -> pd.DataFrame:
    rows = []
    for a, b in windows:
        car  = (panel[panel["rel_day"].between(a, b)]
                .groupby(["stock_code", "event_date"])["ar"]
                .sum().dropna().values)
        t, p = stats.ttest_1samp(car, 0.0) if len(car) >= 2 else (np.nan, np.nan)
        rows.append({
            "window":     f"[{a},{b}]",
            "N":          len(car),
            "mean_CAR":   np.mean(car)        if len(car) else np.nan,
            "median_CAR": np.median(car)      if len(car) else np.nan,
            "std_CAR":    np.std(car, ddof=1) if len(car) > 1 else np.nan,
            "t_stat":     t,
            "p_value":    p,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 8. 전략 백테스트 요약
# ─────────────────────────────────────────
def backtest_summary(panel: pd.DataFrame, entry: int = 0, exit_: int = 5):
    tmp = panel[panel["rel_day"].between(entry, exit_)].copy()
    out = (tmp.groupby(["stock_code", "event_date"])
              .agg(raw_cum=("ret", "sum"), ab_cum=("ar", "sum"), n_days=("rel_day", "count"))
              .reset_index())
    out["raw_return"]      = np.exp(out["raw_cum"]) - 1
    out["abnormal_return"] = np.exp(out["ab_cum"])  - 1

    arr  = out["abnormal_return"].dropna().values
    t, p = stats.ttest_1samp(arr, 0.0) if len(arr) >= 2 else (np.nan, np.nan)

    return {
        "window":                   f"[{entry},{exit_}]",
        "N":                        len(out),
        "mean_raw_return":          out["raw_return"].mean(),
        "median_raw_return":        out["raw_return"].median(),
        "mean_abnormal_return":     out["abnormal_return"].mean(),
        "median_abnormal_return":   out["abnormal_return"].median(),
        "t_stat":                   t,
        "p_value":                  p,
    }


# ─────────────────────────────────────────
# 9. 시각화
# ─────────────────────────────────────────
def save_charts(aar_test: pd.DataFrame, out_dir: str):
    # AAR 막대차트
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["tomato" if v < 0 else "steelblue" for v in aar_test["AAR"]]
    ax.bar(aar_test["rel_day"], aar_test["AAR"] * 100, color=colors, width=0.7)
    ax.axvline(0, linestyle="--", color="black", linewidth=1, label="Event Day")
    ax.axhline(0, linestyle="-",  color="black", linewidth=0.5)
    ax.set_xlabel("Relative Day")
    ax.set_ylabel("AAR (%)")
    ax.set_title("Average Abnormal Return (AAR) — Bonus Issue")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aar.png"), dpi=150)
    plt.close()

    # CAAR 선 차트
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aar_test["rel_day"], aar_test["CAAR"] * 100,
            marker="o", markersize=3, color="steelblue", linewidth=2)
    ax.fill_between(aar_test["rel_day"], aar_test["CAAR"] * 100, alpha=0.15, color="steelblue")
    ax.axvline(0, linestyle="--", color="black", linewidth=1, label="Event Day")
    ax.axhline(0, linestyle="-",  color="black", linewidth=0.5)
    ax.set_xlabel("Relative Day")
    ax.set_ylabel("CAAR (%)")
    ax.set_title("Cumulative Average Abnormal Return (CAAR) — Bonus Issue")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "caar.png"), dpi=150)
    plt.close()

    logger.info(f"차트 저장: {out_dir}/aar.png, caar.png")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 로드
    events, prices, market = load_data()

    # 2. 수익률
    prices, market = compute_returns(prices, market)

    # 3. 캘린더 + 인덱스
    prices, market, events = build_calendar_and_attach(prices, market, events)

    # 4. 이벤트 스터디
    panel, meta = run_event_study(events, prices, market)
    if panel is None:
        logger.error("유효한 이벤트가 없습니다. 종료.")
        return

    # 5. 요약
    aar_test    = summarize_aar(panel)
    car_summary = summarize_car(panel)

    print("\n" + "═" * 60)
    print("  Daily AAR / CAAR")
    print("═" * 60)
    print(aar_test.to_string(index=False))

    print("\n" + "═" * 60)
    print("  CAR Window Summary")
    print("═" * 60)
    print(car_summary.to_string(index=False))

    print("\n" + "═" * 60)
    print("  Strategy Backtest")
    print("═" * 60)
    for entry, exit_ in [(0, 5), (0, 10), (0, 20)]:
        s = backtest_summary(panel, entry, exit_)
        print(f"\n  윈도우 {s['window']}  N={s['N']}")
        print(f"  Raw Return     : mean={s['mean_raw_return']:.4f}  median={s['median_raw_return']:.4f}")
        print(f"  Abnormal Return: mean={s['mean_abnormal_return']:.4f}  median={s['median_abnormal_return']:.4f}"
              f"  t={s['t_stat']:.3f}  p={s['p_value']:.4f}")

    # 6. 저장
    panel.to_csv(      os.path.join(OUTPUT_DIR, "event_panel_ar.csv"),    index=False, encoding="utf-8-sig")
    meta.to_csv(       os.path.join(OUTPUT_DIR, "event_meta.csv"),        index=False, encoding="utf-8-sig")
    aar_test.to_csv(   os.path.join(OUTPUT_DIR, "aar_caar_summary.csv"),  index=False, encoding="utf-8-sig")
    car_summary.to_csv(os.path.join(OUTPUT_DIR, "car_window_summary.csv"),index=False, encoding="utf-8-sig")

    # 7. 차트
    save_charts(aar_test, OUTPUT_DIR)

    logger.info("=== 백테스트 완료 ===")


if __name__ == "__main__":
    main()
