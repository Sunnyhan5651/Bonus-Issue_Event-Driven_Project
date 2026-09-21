import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

EVENT_WINDOW = (-10, 20)      # 이벤트 윈도우: 공시일 기준 -10 ~ +20 거래일
ESTIMATION_WINDOW = (-120, -21)  # 추정 윈도우: -120 ~ -21 거래일
MIN_EST_OBS = 60              # alpha, beta 추정에 필요한 최소 관측치
USE_LOG_RETURN = True         # 로그수익률 사용 여부

def load_data(events_path, prices_path, market_path):
    events = pd.read_csv(events_path)
    prices = pd.read_csv(prices_path)
    market = pd.read_csv(market_path)

    events["event_date"] = pd.to_datetime(events["event_date"])
    if "rights_date" in events.columns:
        events["rights_date"] = pd.to_datetime(events["rights_date"], errors="coerce")
    if "list_date" in events.columns:
        events["list_date"] = pd.to_datetime(events["list_date"], errors="coerce")

    prices["date"] = pd.to_datetime(prices["date"])
    market["date"] = pd.to_datetime(market["date"])

    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    events["stock_code"] = events["stock_code"].astype(str).str.zfill(6)

    return events, prices, market

# 수익률 계산 함수
def compute_returns(prices, market, use_log_return=True):
    prices = prices.sort_values(["stock_code", "date"]).copy()
    market = market.sort_values("date").copy()

    if use_log_return:
        prices["ret"] = prices.groupby("stock_code")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        market["mkt_ret"] = np.log(market["market_close"] / market["market_close"].shift(1))
    else:
        prices["ret"] = prices.groupby("stock_code")["close"].pct_change()
        market["mkt_ret"] = market["market_close"].pct_change()

    return prices, market


def build_trading_calendar(prices, market):
    trading_days = pd.DataFrame({"date": sorted(set(prices["date"]).intersection(set(market["date"])))})
    trading_days = trading_days.reset_index(drop=True)
    trading_days["t_index"] = np.arange(len(trading_days))
    return trading_days


def attach_event_time(prices, market, events, trading_days):
    prices = prices.merge(trading_days, on="date", how="inner")
    market = market.merge(trading_days, on="date", how="inner")

    event_map = trading_days.rename(columns={"date": "event_date", "t_index": "event_t_index"})
    events = events.merge(event_map, on="event_date", how="left")

    if events["event_t_index"].isna().any():
        missing = events.loc[events["event_t_index"].isna(), ["stock_code", "event_date"]]
        print("주의: 거래일 캘린더에서 찾지 못한 event_date가 있음")
        print(missing.head())

    return prices, market, events

# 개별 이벤트별 AR 계산
def estimate_market_model(stock_df, market_df, event_t_index,
                          estimation_window=ESTIMATION_WINDOW,
                          event_window=EVENT_WINDOW,
                          min_est_obs=MIN_EST_OBS):
    """
    stock_df: 단일 종목 데이터(date, t_index, ret)
    market_df: 시장 데이터(date, t_index, mkt_ret)
    event_t_index: 이벤트 날짜의 거래일 index
    """

    df = stock_df.merge(
        market_df[["date", "t_index", "mkt_ret"]],
        on=["date", "t_index"],
        how="inner"
    ).copy()

    df["rel_day"] = df["t_index"] - event_t_index

    # 추정 구간
    est = df[(df["rel_day"] >= estimation_window[0]) & (df["rel_day"] <= estimation_window[1])].dropna()
    if len(est) < min_est_obs:
        return None

    X = sm.add_constant(est["mkt_ret"])
    y = est["ret"]
    model = sm.OLS(y, X).fit()

    # 이벤트 구간
    ev = df[(df["rel_day"] >= event_window[0]) & (df["rel_day"] <= event_window[1])].copy()
    ev = ev.dropna(subset=["ret", "mkt_ret"])

    if ev.empty:
        return None

    X_ev = sm.add_constant(ev["mkt_ret"])
    ev["expected_ret"] = model.predict(X_ev)
    ev["ar"] = ev["ret"] - ev["expected_ret"]

    return {
        "alpha": model.params["const"],
        "beta": model.params["mkt_ret"],
        "n_est": len(est),
        "event_df": ev[["date", "t_index", "rel_day", "ret", "mkt_ret", "expected_ret", "ar"]].copy(),
        "model": model
    }

# 전체 이벤트 처리
def run_event_study(events, prices, market,
                    estimation_window=ESTIMATION_WINDOW,
                    event_window=EVENT_WINDOW,
                    min_est_obs=MIN_EST_OBS):
    results = []
    meta_rows = []

    market_df = market.copy()

    for _, ev in events.iterrows():
        stock_code = ev["stock_code"]
        event_date = ev["event_date"]
        event_t_index = ev["event_t_index"]

        if pd.isna(event_t_index):
            continue

        stock_df = prices[prices["stock_code"] == stock_code].copy()
        if stock_df.empty:
            continue

        out = estimate_market_model(
            stock_df=stock_df,
            market_df=market_df,
            event_t_index=event_t_index,
            estimation_window=estimation_window,
            event_window=event_window,
            min_est_obs=min_est_obs
        )

        if out is None:
            continue

        evdf = out["event_df"].copy()
        evdf["stock_code"] = stock_code
        evdf["event_date"] = event_date

        # 부가 정보 붙이기
        for c in events.columns:
            if c not in evdf.columns:
                evdf[c] = ev[c]

        results.append(evdf)

        meta_rows.append({
            "stock_code": stock_code,
            "event_date": event_date,
            "alpha": out["alpha"],
            "beta": out["beta"],
            "n_est": out["n_est"]
        })

    if len(results) == 0:
        return None, None

    panel = pd.concat(results, ignore_index=True)
    meta = pd.DataFrame(meta_rows)

    return panel, meta



# AAR / CAAR 계산

def summarize_event_study(panel):
    """
    panel: event-level abnormal return panel
    """
    aar = panel.groupby("rel_day")["ar"].mean().reset_index(name="AAR")
    aar["CAAR"] = aar["AAR"].cumsum()

    # t-test for AAR by day
    t_rows = []
    for d, g in panel.groupby("rel_day"):
        arr = g["ar"].dropna().values
        if len(arr) >= 2:
            t_stat, p_val = stats.ttest_1samp(arr, 0.0, nan_policy="omit")
        else:
            t_stat, p_val = np.nan, np.nan

        t_rows.append({
            "rel_day": d,
            "N": len(arr),
            "AAR": np.mean(arr) if len(arr) > 0 else np.nan,
            "AAR_t": t_stat,
            "AAR_p": p_val
        })

    aar_test = pd.DataFrame(t_rows).sort_values("rel_day")
    aar_test["CAAR"] = aar_test["AAR"].cumsum()

    return aar_test


# CAR 계산
def compute_car_by_window(panel, windows=[(0, 1), (0, 3), (0, 5), (0, 10), (-1, 1), (-3, 3)]):
    out_rows = []

    for (a, b) in windows:
        tmp = panel[(panel["rel_day"] >= a) & (panel["rel_day"] <= b)].copy()
        car_df = tmp.groupby(["stock_code", "event_date"])["ar"].sum().reset_index(name="CAR")
        car_df["window"] = f"[{a},{b}]"

        arr = car_df["CAR"].dropna().values
        if len(arr) >= 2:
            t_stat, p_val = stats.ttest_1samp(arr, 0.0, nan_policy="omit")
        else:
            t_stat, p_val = np.nan, np.nan

        out_rows.append({
            "window": f"[{a},{b}]",
            "N": len(arr),
            "mean_CAR": np.mean(arr) if len(arr) > 0 else np.nan,
            "median_CAR": np.median(arr) if len(arr) > 0 else np.nan,
            "std_CAR": np.std(arr, ddof=1) if len(arr) > 1 else np.nan,
            "t_stat": t_stat,
            "p_value": p_val
        })

    return pd.DataFrame(out_rows)

# 수익률 검증
def strategy_backtest_from_event(panel, entry_day=0, exit_day=5):
    """
    이벤트일 기준 entry_day에 진입, exit_day에 청산
    raw ret 누적수익률 / abnormal ret 누적수익률 둘 다 계산
    """
    tmp = panel[(panel["rel_day"] >= entry_day) & (panel["rel_day"] <= exit_day)].copy()

    grp = tmp.groupby(["stock_code", "event_date"])
    out = grp.agg(
        raw_cum_ret=("ret", "sum"),
        abnormal_cum_ret=("ar", "sum"),
        n_days=("rel_day", "count")
    ).reset_index()

    # 로그수익률이면 exp(sum)-1
    out["raw_total_return"] = np.exp(out["raw_cum_ret"]) - 1
    out["abnormal_total_return"] = np.exp(out["abnormal_cum_ret"]) - 1

    arr = out["abnormal_total_return"].dropna().values
    if len(arr) >= 2:
        t_stat, p_val = stats.ttest_1samp(arr, 0.0, nan_policy="omit")
    else:
        t_stat, p_val = np.nan, np.nan

    summary = {
        "entry_day": entry_day,
        "exit_day": exit_day,
        "N": len(out),
        "mean_raw_return": out["raw_total_return"].mean(),
        "median_raw_return": out["raw_total_return"].median(),
        "mean_abnormal_return": out["abnormal_total_return"].mean(),
        "median_abnormal_return": out["abnormal_total_return"].median(),
        "t_stat_abnormal_return": t_stat,
        "p_value_abnormal_return": p_val,
    }

    return out, summary


# =========================================================
# 9. bonus_ratio 등으로 cross-sectional bucket 분석
# =========================================================
def bucket_analysis(panel, events, bucket_col="bonus_ratio", bins=[0, 0.5, 1.0, 2.0, 10.0], labels=None, car_window=(0, 5)):
    if labels is None:
        labels = ["<=0.5", "0.5~1.0", "1.0~2.0", ">2.0"]

    tmp = panel.merge(
        events[[c for c in events.columns if c in ["stock_code", "event_date", bucket_col"]],
        on=["stock_code", "event_date"],
        how="left"
    )

    tmp[bucket_col + "_grp"] = pd.cut(tmp[bucket_col], bins=bins, labels=labels, include_lowest=True)

    a, b = car_window
    tmp = tmp[(tmp["rel_day"] >= a) & (tmp["rel_day"] <= b)].copy()
    car_df = tmp.groupby(["stock_code", "event_date", bucket_col + "_grp"])["ar"].sum().reset_index(name="CAR")

    summary = car_df.groupby(bucket_col + "_grp")["CAR"].agg(["count", "mean", "median", "std"]).reset_index()
    return car_df, summary


# =========================================================
# 10. 그림
# =========================================================
def plot_caar(aar_test, title="CAAR around Bonus-Issue Announcement"):
    plt.figure(figsize=(10, 6))
    plt.plot(aar_test["rel_day"], aar_test["CAAR"], marker="o")
    plt.axvline(0, linestyle="--")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Relative Day")
    plt.ylabel("CAAR")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_aar(aar_test, title="AAR around Bonus-Issue Announcement"):
    plt.figure(figsize=(10, 6))
    plt.bar(aar_test["rel_day"], aar_test["AAR"])
    plt.axvline(0, linestyle="--")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Relative Day")
    plt.ylabel("AAR")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


# =========================================================
# 11. 메인 실행 예시
# =========================================================
if __name__ == "__main__":
    events, prices, market = load_data(
        events_path="events.csv",
        prices_path="prices.csv",
        market_path="market.csv"
    )

    prices, market = compute_returns(prices, market, use_log_return=USE_LOG_RETURN)
    trading_days = build_trading_calendar(prices, market)
    prices, market, events = attach_event_time(prices, market, events, trading_days)

    panel, meta = run_event_study(
        events=events,
        prices=prices,
        market=market,
        estimation_window=ESTIMATION_WINDOW,
        event_window=EVENT_WINDOW,
        min_est_obs=MIN_EST_OBS
    )

    if panel is None:
        print("유효한 이벤트가 없습니다.")
    else:
        # AAR / CAAR
        aar_test = summarize_event_study(panel)
        print("\n=== Daily AAR / CAAR ===")
        print(aar_test)

        # CAR windows
        car_summary = compute_car_by_window(
            panel,
            windows=[(-1, 1), (0, 1), (0, 3), (0, 5), (0, 10), (0, 20)]
        )
        print("\n=== CAR Window Summary ===")
        print(car_summary)

        # 전략 수익률 예시
        bt_0_5, bt_sum_0_5 = strategy_backtest_from_event(panel, entry_day=0, exit_day=5)
        bt_0_10, bt_sum_0_10 = strategy_backtest_from_event(panel, entry_day=0, exit_day=10)

        print("\n=== Strategy Summary: Buy at day 0, sell at day 5 ===")
        print(bt_sum_0_5)

        print("\n=== Strategy Summary: Buy at day 0, sell at day 10 ===")
        print(bt_sum_0_10)

        # 플롯
        plot_aar(aar_test)
        plot_caar(aar_test)

        # 저장
        panel.to_csv("event_panel_ar.csv", index=False, encoding="utf-8-sig")
        meta.to_csv("event_meta.csv", index=False, encoding="utf-8-sig")
        aar_test.to_csv("aar_caar_summary.csv", index=False, encoding="utf-8-sig")
        car_summary.to_csv("car_window_summary.csv", index=False, encoding="utf-8-sig")

        print("\n결과 파일 저장 완료")