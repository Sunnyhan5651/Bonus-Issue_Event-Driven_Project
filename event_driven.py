import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
EVENT_WINDOW      = (-10, 20)   # 이벤트 윈도우: 공시일 기준 -10 ~ +20 거래일
ESTIMATION_WINDOW = (-120, -21) # 추정 윈도우: -120 ~ -21 거래일
MIN_EST_OBS       = 60          # alpha, beta 추정에 필요한 최소 관측치
USE_LOG_RETURN    = True        # 로그수익률 사용 여부


# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
def load_data(events_path: str, prices_path: str, market_path: str):
    events = pd.read_csv(events_path)
    prices = pd.read_csv(prices_path)
    market = pd.read_csv(market_path)

    events["event_date"] = pd.to_datetime(events["event_date"])
    for col in ("rights_date", "list_date"):
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors="coerce")

    prices["date"] = pd.to_datetime(prices["date"])
    market["date"] = pd.to_datetime(market["date"])

    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    events["stock_code"]  = events["stock_code"].astype(str).str.zfill(6)

    return events, prices, market


# ─────────────────────────────────────────
# 2. 수익률 계산
# ─────────────────────────────────────────
def compute_returns(prices: pd.DataFrame, market: pd.DataFrame, use_log_return: bool = True):
    prices = prices.sort_values(["stock_code", "date"]).copy()
    market = market.sort_values("date").copy()

    if use_log_return:
        prices["ret"]    = prices.groupby("stock_code")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        market["mkt_ret"] = np.log(market["market_close"] / market["market_close"].shift(1))
    else:
        prices["ret"]    = prices.groupby("stock_code")["close"].pct_change()
        market["mkt_ret"] = market["market_close"].pct_change()

    return prices, market


# ─────────────────────────────────────────
# 3. 거래일 캘린더 생성
# ─────────────────────────────────────────
def build_trading_calendar(prices: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    common_dates = sorted(set(prices["date"]).intersection(market["date"]))
    trading_days = pd.DataFrame({"date": common_dates})
    trading_days["t_index"] = np.arange(len(trading_days))
    return trading_days


# ─────────────────────────────────────────
# 4. 이벤트 시간 인덱스 부착
# ─────────────────────────────────────────
def attach_event_time(prices, market, events, trading_days):
    prices = prices.merge(trading_days, on="date", how="inner")
    market = market.merge(trading_days, on="date", how="inner")

    event_map = trading_days.rename(columns={"date": "event_date", "t_index": "event_t_index"})
    events = events.merge(event_map, on="event_date", how="left")

    missing = events[events["event_t_index"].isna()][["stock_code", "event_date"]]
    if not missing.empty:
        print("주의: 거래일 캘린더에서 찾지 못한 event_date가 있음")
        print(missing.head())

    return prices, market, events


# ─────────────────────────────────────────
# 5. 마켓모델 추정 및 AR 계산 (이벤트 1건)
# ─────────────────────────────────────────
def estimate_market_model(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    event_t_index: int,
    estimation_window: tuple = ESTIMATION_WINDOW,
    event_window: tuple = EVENT_WINDOW,
    min_est_obs: int = MIN_EST_OBS,
):
    df = stock_df.merge(
        market_df[["date", "t_index", "mkt_ret"]],
        on=["date", "t_index"],
        how="inner",
    ).copy()
    df["rel_day"] = df["t_index"] - event_t_index

    # 추정 구간
    est = df[df["rel_day"].between(*estimation_window)].dropna(subset=["ret", "mkt_ret"])
    if len(est) < min_est_obs:
        return None

    model = sm.OLS(est["ret"], sm.add_constant(est["mkt_ret"])).fit()

    # 이벤트 구간
    ev = df[df["rel_day"].between(*event_window)].dropna(subset=["ret", "mkt_ret"]).copy()
    if ev.empty:
        return None

    ev["expected_ret"] = model.predict(sm.add_constant(ev["mkt_ret"]))
    ev["ar"] = ev["ret"] - ev["expected_ret"]

    return {
        "alpha":    model.params["const"],
        "beta":     model.params["mkt_ret"],
        "n_est":    len(est),
        "event_df": ev[["date", "t_index", "rel_day", "ret", "mkt_ret", "expected_ret", "ar"]].copy(),
        "model":    model,
    }


# ─────────────────────────────────────────
# 6. 전체 이벤트 처리
# ─────────────────────────────────────────
def run_event_study(
    events, prices, market,
    estimation_window: tuple = ESTIMATION_WINDOW,
    event_window: tuple = EVENT_WINDOW,
    min_est_obs: int = MIN_EST_OBS,
):
    results, meta_rows = [], []

    for _, ev in events.iterrows():
        if pd.isna(ev["event_t_index"]):
            continue

        stock_df = prices[prices["stock_code"] == ev["stock_code"]].copy()
        if stock_df.empty:
            continue

        out = estimate_market_model(
            stock_df=stock_df,
            market_df=market,
            event_t_index=ev["event_t_index"],
            estimation_window=estimation_window,
            event_window=event_window,
            min_est_obs=min_est_obs,
        )
        if out is None:
            continue

        evdf = out["event_df"].copy()
        evdf["stock_code"] = ev["stock_code"]
        evdf["event_date"]  = ev["event_date"]

        for col in events.columns:
            if col not in evdf.columns:
                evdf[col] = ev[col]

        results.append(evdf)
        meta_rows.append({
            "stock_code": ev["stock_code"],
            "event_date": ev["event_date"],
            "alpha":      out["alpha"],
            "beta":       out["beta"],
            "n_est":      out["n_est"],
        })

    if not results:
        return None, None

    return pd.concat(results, ignore_index=True), pd.DataFrame(meta_rows)


# ─────────────────────────────────────────
# 7. AAR / CAAR 요약
# ─────────────────────────────────────────
def summarize_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in panel.groupby("rel_day"):
        arr = g["ar"].dropna().values
        t_stat, p_val = (
            stats.ttest_1samp(arr, 0.0) if len(arr) >= 2 else (np.nan, np.nan)
        )
        rows.append({
            "rel_day": d,
            "N":       len(arr),
            "AAR":     np.mean(arr) if len(arr) else np.nan,
            "AAR_t":   t_stat,
            "AAR_p":   p_val,
        })

    result = pd.DataFrame(rows).sort_values("rel_day")
    result["CAAR"] = result["AAR"].cumsum()
    return result


# ─────────────────────────────────────────
# 8. CAR 윈도우별 요약
# ─────────────────────────────────────────
def compute_car_by_window(
    panel: pd.DataFrame,
    windows: list = [(0, 1), (0, 3), (0, 5), (0, 10), (-1, 1), (-3, 3)],
) -> pd.DataFrame:
    rows = []
    for a, b in windows:
        car = (
            panel[panel["rel_day"].between(a, b)]
            .groupby(["stock_code", "event_date"])["ar"]
            .sum()
            .dropna()
            .values
        )
        t_stat, p_val = (
            stats.ttest_1samp(car, 0.0) if len(car) >= 2 else (np.nan, np.nan)
        )
        rows.append({
            "window":     f"[{a},{b}]",
            "N":          len(car),
            "mean_CAR":   np.mean(car)   if len(car) else np.nan,
            "median_CAR": np.median(car) if len(car) else np.nan,
            "std_CAR":    np.std(car, ddof=1) if len(car) > 1 else np.nan,
            "t_stat":     t_stat,
            "p_value":    p_val,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 9. 전략 수익률 백테스트
# ─────────────────────────────────────────
def strategy_backtest_from_event(panel: pd.DataFrame, entry_day: int = 0, exit_day: int = 5):
    tmp = panel[panel["rel_day"].between(entry_day, exit_day)].copy()
    out = (
        tmp.groupby(["stock_code", "event_date"])
        .agg(raw_cum_ret=("ret", "sum"), abnormal_cum_ret=("ar", "sum"), n_days=("rel_day", "count"))
        .reset_index()
    )
    # 로그수익률 → 단순수익률 변환
    out["raw_total_return"]      = np.exp(out["raw_cum_ret"]) - 1
    out["abnormal_total_return"] = np.exp(out["abnormal_cum_ret"]) - 1

    arr = out["abnormal_total_return"].dropna().values
    t_stat, p_val = (
        stats.ttest_1samp(arr, 0.0) if len(arr) >= 2 else (np.nan, np.nan)
    )

    summary = {
        "entry_day":                entry_day,
        "exit_day":                 exit_day,
        "N":                        len(out),
        "mean_raw_return":          out["raw_total_return"].mean(),
        "median_raw_return":        out["raw_total_return"].median(),
        "mean_abnormal_return":     out["abnormal_total_return"].mean(),
        "median_abnormal_return":   out["abnormal_total_return"].median(),
        "t_stat_abnormal_return":   t_stat,
        "p_value_abnormal_return":  p_val,
    }
    return out, summary


# ─────────────────────────────────────────
# 10. 버킷(Cross-sectional) 분석
# ─────────────────────────────────────────
def bucket_analysis(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    bucket_col: str = "bonus_ratio",
    bins: list = [0, 0.5, 1.0, 2.0, 10.0],
    labels: list = None,
    car_window: tuple = (0, 5),
):
    if labels is None:
        labels = ["≤0.5", "0.5~1.0", "1.0~2.0", ">2.0"]

    merge_cols = ["stock_code", "event_date"] + (
        [bucket_col] if bucket_col in events.columns else []
    )
    tmp = panel.merge(events[merge_cols], on=["stock_code", "event_date"], how="left")

    grp_col = bucket_col + "_grp"
    tmp[grp_col] = pd.cut(tmp[bucket_col], bins=bins, labels=labels, include_lowest=True)

    a, b = car_window
    car_df = (
        tmp[tmp["rel_day"].between(a, b)]
        .groupby(["stock_code", "event_date", grp_col])["ar"]
        .sum()
        .reset_index(name="CAR")
    )

    summary = car_df.groupby(grp_col)["CAR"].agg(["count", "mean", "median", "std"]).reset_index()
    return car_df, summary


# ─────────────────────────────────────────
# 11. 시각화
# ─────────────────────────────────────────
def plot_caar(aar_test: pd.DataFrame, title: str = "CAAR around Bonus-Issue Announcement"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(aar_test["rel_day"], aar_test["CAAR"], marker="o")
    ax.axvline(0, linestyle="--", color="gray")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Relative Day")
    ax.set_ylabel("CAAR")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_aar(aar_test: pd.DataFrame, title: str = "AAR around Bonus-Issue Announcement"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(aar_test["rel_day"], aar_test["AAR"])
    ax.axvline(0, linestyle="--", color="gray")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Relative Day")
    ax.set_ylabel("AAR")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# 12. 메인
# ─────────────────────────────────────────
if __name__ == "__main__":
    events, prices, market = load_data("events.csv", "prices.csv", "market.csv")

    prices, market = compute_returns(prices, market, use_log_return=USE_LOG_RETURN)
    trading_days   = build_trading_calendar(prices, market)
    prices, market, events = attach_event_time(prices, market, events, trading_days)

    panel, meta = run_event_study(events, prices, market)

    if panel is None:
        print("유효한 이벤트가 없습니다.")
    else:
        aar_test    = summarize_event_study(panel)
        car_summary = compute_car_by_window(
            panel, windows=[(-1, 1), (0, 1), (0, 3), (0, 5), (0, 10), (0, 20)]
        )

        print("\n=== Daily AAR / CAAR ===")
        print(aar_test)
        print("\n=== CAR Window Summary ===")
        print(car_summary)

        for entry, exit_ in [(0, 5), (0, 10)]:
            _, summary = strategy_backtest_from_event(panel, entry_day=entry, exit_day=exit_)
            print(f"\n=== Strategy: Buy day {entry}, Sell day {exit_} ===")
            print(summary)

        plot_aar(aar_test)
        plot_caar(aar_test)

        panel.to_csv("event_panel_ar.csv",      index=False, encoding="utf-8-sig")
        meta.to_csv("event_meta.csv",           index=False, encoding="utf-8-sig")
        aar_test.to_csv("aar_caar_summary.csv", index=False, encoding="utf-8-sig")
        car_summary.to_csv("car_window_summary.csv", index=False, encoding="utf-8-sig")

        print("\n결과 파일 저장 완료")
