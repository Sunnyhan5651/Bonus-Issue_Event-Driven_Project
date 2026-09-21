"""
무상증자 공시 알파 전략 백테스트

입력  : dart_bonus_crawler.py 출력 (korea_bonus_issue.csv)
주가  : 한투 API (수정주가 X)
매수  : 공시일이 영업일 → 당일 종가 / 공시일이 비영업일 → 다음 영업일 시가
매도  : 신주배정기준일(=권리락일) T-2 영업일 종가
"""

import re
import time
import datetime
import requests
import urllib3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import OpenDartReader

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# 설정  
# 본인 키 입력 필요
KIS_APP_KEY    = "PSvCIWHm81Fu9pem4Du6BoPoR1bVQ5Pfj3se"
KIS_APP_SECRET = "rpJTDsFCbPS/+B1RvAZd9nTsdrgw18dP0Mn+pmIEDuZthe2uFK8dPsvwmqX2+LHqreo6CwFaeaZmQEimO1r7NX5htbdYVSJ253kVj0G5n2TXn8MrsAkBECI6RgD4dZo5qf9cx4R5kCzZL6Jrm8foJu/auzk3gg7C/S/TmDBEIdJr3zKnlUQ="
KIS_BASE_URL   = "https://openapi.koreainvestment.com:9443"
DART_API_KEY   = "601bd772cea9f0f066d4ae097b8e6223c61e6e6b"

COMMISSION_RATE = 0.00015  # 수수료
TRANSACTION_TAX = 0.0018   # 거래세
INITIAL_CAPITAL = 100_000_000   #초기자본 1억
POSITION_WEIGHT = 0.2  # 포지션 당 가중치, 종목 당 자본의 20%씩 투입한다고 생각
BACKTEST_YEARS  = 10  #백테스트 기간 (여기선 그냥 3년으로 했습니다)
CSV_PATH        = "korea_bonus_issue_stock_split_events.csv"
DART_CACHE      = Path("dart_record_date_cache.csv") # DART 공시자료 캐시 폴더
PRICE_CACHE_DIR = Path("price_cache_kis")   # 종목별 일봉 캐시 폴더(3년치 주가데이터 종목별로 일괄 다운받음)

dart = OpenDartReader(DART_API_KEY)

kis_session = requests.Session()
kis_session.verify = False

_token: dict = {}

def get_token() -> str:    # 한투 토큰 발급 함수 : 이미 유효한 토큰이 발급되어있으면 사용, 없으면 새로 발급
    now = datetime.datetime.now()
    if _token.get("val") and _token.get("exp", now) > now:
        return _token["val"]
    r = kis_session.post(
        f"{KIS_BASE_URL}/oauth2/tokenP",
        json={"grant_type": "client_credentials",
              "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET},
        timeout=10,
    )
    data = r.json()
    _token["val"] = data["access_token"]
    _token["exp"] = now + datetime.timedelta(seconds=int(data.get("expires_in", 86400)) - 60)
    print(f"토큰 발급 완료")
    return _token["val"]

def kis_headers(tr_id: str) -> dict:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {get_token()}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
    }

def fetch_ohlc(code: str, start: str, end: str) -> pd.DataFrame:
    """
    KIS 국내주식 기간별 시세 (일봉)
    FID_ORG_ADJ_PRC = "0" → 원주가 (실제 체결가)
    start / end: "YYYYMMDD"
    """
    url = f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    all_rows = []
    cur_end  = end

    while True:
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD":         code,
            "FID_INPUT_DATE_1":        start,
            "FID_INPUT_DATE_2":        cur_end,
            "FID_PERIOD_DIV_CODE":    "D",
            "FID_ORG_ADJ_PRC":        "0",   # "1"은 수정주가 (수정주가란? : 배당락, 권리락 등으로 인한 가격 변동을 보정한 주가. 무상증자 시에는 보통 수정주가가 원주가보다 낮게 나옴. 백테스트에서는 실제 체결가인 원주가를 사용하기 위해 "0"으로 설정)
        }
        r    = kis_session.get(url, headers=kis_headers("FHKST03010100"),
                               params=params, timeout=15)
        data = r.json()

        if data.get("rt_cd") != "0":
            break

        output2 = data.get("output2", [])
        if not output2:
            break

        for row in output2:
            try:
                all_rows.append({
                    "date":  pd.to_datetime(row["stck_bsop_date"], format="%Y%m%d"),
                    "open":  int(row["stck_oprc"]),
                    "close": int(row["stck_clpr"]),
                })
            except (KeyError, ValueError):
                continue

        last = output2[-1]["stck_bsop_date"]
        if last <= start:
            break
        prev    = datetime.datetime.strptime(last, "%Y%m%d") - datetime.timedelta(days=1)
        cur_end = prev.strftime("%Y%m%d")
        if cur_end < start:
            break
        time.sleep(0.05)

    if not all_rows:
        return pd.DataFrame(columns=["date", "open", "close"])

    df = (pd.DataFrame(all_rows)
          .drop_duplicates("date")
          .sort_values("date")
          .reset_index(drop=True))
    return df[df["date"] >= pd.to_datetime(start)].reset_index(drop=True)


def get_ohlc(code: str, price_start: str, price_end: str) -> pd.DataFrame:
    PRICE_CACHE_DIR.mkdir(exist_ok=True)
    f = PRICE_CACHE_DIR / f"{code}.csv"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"])
    df = fetch_ohlc(code, price_start, price_end)
    if not df.empty:
        df.to_csv(f, index=False)
    return df


def prefetch_all(codes: list, price_start: str, price_end: str):  # 만약 캐시 폴더에 없는 종목이 있다면, 해당 종목 주가 다운로드
    PRICE_CACHE_DIR.mkdir(exist_ok=True)
    missing = [c for c in codes if not (PRICE_CACHE_DIR / f"{c}.csv").exists()]
    if not missing:
        print(f"전 종목 캐시 존재 -> 생략")
        return
    print(f" 일봉 다운로드: {len(missing)}종목")
    for c in tqdm(missing, desc="일봉 다운로드"):
        get_ohlc(c, price_start, price_end)
        time.sleep(0.35)

def price_on_or_after(df: pd.DataFrame, dt: pd.Timestamp, col: str) -> Optional[tuple]:
    """dt 당일 또는 이후 첫 거래일의 (date, price[col])"""
    sub = df[df["date"] >= dt]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return row["date"], int(row[col])


def price_on_or_before(df: pd.DataFrame, dt: pd.Timestamp, col: str) -> Optional[tuple]:
    """dt 당일 또는 이전 마지막 거래일의 (date, price[col])"""
    sub = df[df["date"] <= dt]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return row["date"], int(row[col])


def is_trading_day(df: pd.DataFrame, dt: pd.Timestamp) -> bool:
    return not df[df["date"] == dt].empty


# DART에서 신주배정기준일 파싱 (여기서는 신주배정기준일 == 권리락일로 간주함)
def _parse_record_date(rcept_no: str) -> Optional[str]:
    try:
        xml = dart.document(rcept_no)
    except Exception:
        return None
    if not xml or len(xml) < 10:
        return None

    m = re.search(r"신주배정기준일.*?AUNITVALUE=['\"](\d{8})['\"]", xml, re.DOTALL)
    if m:
        v = m.group(1)
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"

    idx = xml.find("신주배정기준일")
    if idx >= 0:
        snip = xml[idx: idx + 400]
        m2 = re.search(r'AUNITVALUE=[\'"](\d{8})[\'"]', snip)
        if m2:
            v = m2.group(1)
            return f"{v[:4]}-{v[4:6]}-{v[6:]}"
        m3 = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", snip)
        if m3:
            return f"{m3.group(1)}-{m3.group(2).zfill(2)}-{m3.group(3).zfill(2)}"
    return None


def build_record_date_map(events_df: pd.DataFrame) -> dict:
    cache: dict = {}

    if DART_CACHE.exists():
        tmp = pd.read_csv(DART_CACHE, dtype=str)
        for _, r in tmp.iterrows():
            v = r["record_date"]
            cache[r["rcept_no"]] = pd.to_datetime(v) if (v and v != "None" and pd.notna(v)) else None

    # None 비율 90% 초과면 캐시 무효 -> 파싱이 제대로 안됐다고 판단
    if cache and sum(v is None for v in cache.values()) / len(cache) > 0.9:
        print("[DART캐시] 오류 캐시 감지 → 삭제 후 재수집")
        cache = {}
        DART_CACHE.unlink()

    missing = [r for r in events_df["rcept_no"] if r not in cache]
    if missing:
        print(f"[DART] 신주배정기준일 조회: {len(missing)}건 ")
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_parse_record_date, r): r for r in missing}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="DART 기준일"):
                rno    = futures[fut]
                result = fut.result()
                cache[rno] = pd.to_datetime(result) if result else None

        pd.DataFrame([
            {"rcept_no": k, "record_date": str(v.date()) if v else "None"}
            for k, v in cache.items()
        ]).to_csv(DART_CACHE, index=False)

        ok   = sum(v is not None for v in cache.values())
        fail = len(cache) - ok
        print(f"DART 공시자료 크롤링 성공: {ok}건 / 실패(fallback): {fail}건")

    return cache

# 백테스트

def run_backtest(events_df: pd.DataFrame, record_map: dict,
                 price_store: dict) -> pd.DataFrame:
    pos_cap  = INITIAL_CAPITAL * POSITION_WEIGHT
    trades   = []
    fallback = 0
    skip_no_price = 0

    for _, ev in events_df.iterrows():
        code     = str(ev["stock_code"]).zfill(6)
        ann_date = pd.to_datetime(ev["announcement_date"])
        rcept_no = str(ev["rcept_no"])

        ohlc = price_store.get(code)
        if ohlc is None or ohlc.empty:
            skip_no_price += 1
            continue

        # 매수 로직 : 만약 공시자료가 장중에 나왔다면, 당일 종가로 매수. 만약 장마감 후 또는 주말, 공휴일에 나왔다면, 다음 영업일 시가로 매수.

        # 공시일이 영업일 -> 당일 종가
        # 공시일이 비영업일 -> 다음 영업일 시가
        if is_trading_day(ohlc, ann_date):
            buy_result = price_on_or_before(ohlc, ann_date, "close")
        else:
            buy_result = price_on_or_after(ohlc, ann_date, "open")

        if buy_result is None:
            continue
        buy_date, buy_price = buy_result
        if buy_price <= 0:
            continue


        record_date = record_map.get(rcept_no)
        if record_date is None:
            fallback += 1
            record_date = ann_date + pd.offsets.BDay(30)

        # 매도 로직
        # 신주배정기준일 T-1 영업일 종가
        t1 = record_date - pd.offsets.BDay(1)
        sell_result = price_on_or_before(ohlc, t1, "close")
        if sell_result is None:
            continue
        sell_date, sell_price = sell_result
        if sell_price <= 0 or sell_date <= buy_date:
            continue

        shares   = int(pos_cap / buy_price)
        if shares <= 0:
            continue
        cost     = shares * buy_price  * (1 + COMMISSION_RATE)
        proceeds = shares * sell_price * (1 - COMMISSION_RATE - TRANSACTION_TAX)
        net_pnl  = proceeds - cost
        ret_pct  = net_pnl / cost * 100

        trades.append({
            "corp_name":   ev["corp_name"],
            "stock_code":  code,
            "market":      ev.get("market", ""),
            "buy_date":    buy_date,
            "buy_price":   buy_price,
            "record_date": record_date,
            "sell_date":   sell_date,
            "sell_price":  sell_price,
            "shares":      shares,
            "net_pnl":     net_pnl,
            "ret_pct":     ret_pct,
            "hold_days":   (sell_date - buy_date).days,
        })

    if fallback:
        print(f"[주의] 기준일 fallback 적용: {fallback}건")
    if skip_no_price:
        print(f"[주의] 주가 데이터 없어 스킵: {skip_no_price}건")
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).sort_values("buy_date").reset_index(drop=True)

# 성과 출력

def report(df: pd.DataFrame):
    n    = len(df)
    wins = (df["net_pnl"] > 0).sum()
    df = df.copy()
    df["cum_pnl"] = df["net_pnl"].cumsum()
    mdd     = (df["cum_pnl"] - df["cum_pnl"].cummax()).min()
    monthly = df.groupby(df["sell_date"].dt.to_period("M"))["net_pnl"].sum()

    print("\n" + "=" * 55)
    print("  무상증자 공시 알파 전략 Backtest Result")
    print("=" * 55)
    print(f"  총 거래수      : {n} 건")
    print(f"  승률           : {wins/n*100:.1f}%  (승 {wins} / 패 {n-wins})")
    print(f"  초기 자본      : {INITIAL_CAPITAL:>15,.0f} 원")
    print(f"  Net PnL      : {df['net_pnl'].sum():>15,.0f} 원")
    print(f"  평균 수익률    : {df['ret_pct'].mean():.4f}%  (중앙값 {df['ret_pct'].median():.4f}%)")
    print(f"  최대수익 거래 : {df['ret_pct'].max():.4f}%")
    print(f"  최대손실 거래 : {df['ret_pct'].min():.4f}%")
    print(f"  평균 보유일    : {df['hold_days'].mean():.1f} 일")
    print(f"  MDD            : {mdd:>15,.0f} 원")
    print("=" * 55)

    print("\n[월별 PnL]")
    for period, val in monthly.items():
        sign = "+" if val >= 0 else "-"
        bar  = "▇" * min(int(abs(val) / 500_000), 40)
        print(f"  {period}  {sign}{abs(val):>12,.0f}원  {bar}")

    cols = ["corp_name", "stock_code", "buy_date", "buy_price",
            "record_date", "sell_date", "sell_price", "hold_days", "net_pnl", "ret_pct"]
    print("\n[상위 10건]")
    print(df.nlargest(10, "ret_pct")[cols].to_string(index=False))
    print("\n[하위 10건]")
    print(df.nsmallest(10, "ret_pct")[cols].to_string(index=False))



def main():
    end_dt      = datetime.date.today()
    start_dt    = end_dt.replace(year=end_dt.year - BACKTEST_YEARS)
    price_start = start_dt.strftime("%Y%m%d")
    price_end   = end_dt.strftime("%Y%m%d")
    print(f"Window: {price_start} ~ {price_end}")

    # 공시 CSV
    df = pd.read_csv(CSV_PATH, dtype={"stock_code": str, "rcept_no": str})
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    df["rcept_no"] = df["dart_url"].str.extract(r"rcpNo=(\d+)")[0]
    df = df[
        (df["announcement_date"] >= pd.to_datetime(price_start)) &
        (df["announcement_date"] <= pd.to_datetime(price_end))
    ].copy().reset_index(drop=True)
    print(f"총 이벤트 수: {len(df)} 건")

    # DART 기준일 수집
    record_map = build_record_date_map(df)

    # 주가 다운로드
    codes = df["stock_code"].astype(str).str.zfill(6).unique().tolist()
    prefetch_all(codes, price_start, price_end)

    # 메모리 로드
    print("주가 메모리 로드 중...")
    price_store = {c: get_ohlc(c, price_start, price_end) for c in codes}

    # 백테스트
    print("백테스트 실행 중...")
    trades_df = run_backtest(df, record_map, price_store)

    if trades_df.empty:
        print("유효한 거래가 없음")
        return

    report(trades_df)

    out = "backtest_result_bonus_issue.csv"
    trades_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n거래 내역 저장: {out}")


if __name__ == "__main__":
    main()