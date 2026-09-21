import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import re


# =========================================================
# 설정
# =========================================================
API_KEY = None   
START_DATE = "20160101"            
END_DATE   = datetime.today().strftime("%Y%m%d")
SLEEP_SEC = 0.15              

BASE_URL = "https://opendart.fss.or.kr/api/list.json"

# 보고서명 패턴
BONUS_PATTERNS = [r"무상증자\s*결정"] # 유무상증자도 원하면 r"유무상증자\s*결정"도 포함
SPLIT_PATTERNS = [
    r"주식분할\s*결정",
    r"액면분할",         
]

PREFIX_PATTERN = r"^\[[^\]]+\]"


# =========================================================
# 유틸
# =========================================================
def chunk_dates_3m(start_date: str, end_date: str):
    """
    corp_code 없이 DART list.json을 조회할 때 검색기간은 3개월 제한.
    따라서 3개월 단위로 쪼갠다.
    """
    s = datetime.strptime(start_date, "%Y%m%d")
    e = datetime.strptime(end_date, "%Y%m%d")

    windows = []
    cur = s
    while cur <= e:
        nxt = min(cur + relativedelta(months=3) - timedelta(days=1), e)
        windows.append((cur.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
        cur = nxt + timedelta(days=1)
    return windows


def clean_report_name(report_nm: str) -> str:
    """
    [기재정정], [첨부정정] 같은 접두어 제거
    """
    if pd.isna(report_nm):
        return report_nm
    return re.sub(PREFIX_PATTERN, "", report_nm).strip()


def classify_event(report_nm: str):
    """
    보고서명으로 이벤트 분류
    """
    nm = clean_report_name(report_nm)

    for p in BONUS_PATTERNS:
        if re.search(p, nm):
            return "bonus_issue"

    for p in SPLIT_PATTERNS:
        if re.search(p, nm):
            return "stock_split"

    return None


def is_correction(report_nm: str) -> bool:
    """
    정정공시 여부
    """
    if pd.isna(report_nm):
        return False
    return bool(re.match(PREFIX_PATTERN, report_nm.strip()))


def dart_viewer_url(rcept_no: str) -> str:
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"


# DART 공시검색 API 호출
def fetch_list_window(api_key: str, bgn_de: str, end_de: str, corp_cls: str):
    """
    corp_cls: 'Y' (KOSPI), 'K' (KOSDAQ)
    pblntf_ty='B', pblntf_detail_ty='B001' -> 주요사항보고서
    """
    page_no = 1
    rows = []

    while True:
        params = {
            "crtfc_key": api_key,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "pblntf_ty": "B",
            "pblntf_detail_ty": "B001",
            "corp_cls": corp_cls,
            "sort": "date",
            "sort_mth": "asc",
            "page_no": page_no,
            "page_count": 100,
        }

        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        status = data.get("status")
        if status == "013":
            break
        if status != "000":
            raise RuntimeError(
                f"DART API error | corp_cls={corp_cls}, {bgn_de}-{end_de}, "
                f"status={status}, message={data.get('message')}"
            )

        batch = data.get("list", [])
        if not batch:
            break

        rows.extend(batch)

        total_page = int(data.get("total_page", 1))
        if page_no >= total_page:
            break
        page_no += 1
        time.sleep(SLEEP_SEC)

    return rows


def fetch_all_major_reports(api_key: str, start_date: str, end_date: str):
    """
    최근 5년 등 기간 전체를 가져온 뒤,
    KOSPI(Y), KOSDAQ(K) 각각 조회 후 합친다.
    """
    windows = chunk_dates_3m(start_date, end_date)
    all_rows = []

    for corp_cls in ["Y", "K"]:
        for bgn_de, end_de in windows:
            print(f"[{corp_cls}] {bgn_de} ~ {end_de} 조회 중...")
            rows = fetch_list_window(api_key, bgn_de, end_de, corp_cls)
            all_rows.extend(rows)
            time.sleep(SLEEP_SEC)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


# 이벤트 필터링 및 정리
def build_event_table(df_raw: pd.DataFrame):
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    keep_cols = [
        "corp_cls", "corp_name", "corp_code", "stock_code",
        "report_nm", "rcept_no", "rcept_dt", "flr_nm", "rm"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    df = df[keep_cols]

    df["report_nm_clean"] = df["report_nm"].apply(clean_report_name)
    df["is_correction"] = df["report_nm"].apply(is_correction)
    df["event_type"] = df["report_nm"].apply(classify_event)

    # 무상증자 / 주식분할만 남김
    df = df[df["event_type"].notna()].copy()

    # 시장명 매핑
    df["market"] = df["corp_cls"].map({"Y": "KOSPI", "K": "KOSDAQ"})

    # 공시일 datetime
    df["announcement_date"] = pd.to_datetime(df["rcept_dt"], format="%Y%m%d", errors="coerce")

    # DART 링크
    df["dart_url"] = df["rcept_no"].apply(dart_viewer_url)

    # 이벤트명 한글
    df["event_name_kr"] = df["event_type"].map({
        "bonus_issue": "무상증자",
        "stock_split": "주식분할"
    })

    # 중복 제거 전략
    # 같은 회사/같은 접수일/같은 이벤트명 중복 시 1건만
    df = df.sort_values(["stock_code", "announcement_date", "rcept_no"])
    df = df.drop_duplicates(
        subset=["stock_code", "announcement_date", "event_type"],
        keep="first"
    )

    # 보기 좋게 정렬
    df = df.sort_values(["announcement_date", "market", "stock_code"]).reset_index(drop=True)

    # 최종 컬럼
    final_cols = [
        "announcement_date",
        "market",
        "stock_code",
        "corp_name",
        "event_name_kr",
        "event_type",
        "report_nm",
        "report_nm_clean",
        "is_correction",
        "rcept_no",
        "dart_url",
    ]
    return df[final_cols]


# =========================================================
# 메인
# =========================================================
def main():
    raw = fetch_all_major_reports(API_KEY, START_DATE, END_DATE)

    if raw.empty:
        print("조회된 공시가 없습니다.")
        return

    events = build_event_table(raw)

    if events.empty:
        print("무상증자/주식분할 공시가 없습니다.")
        return

    # 파일 저장
    events.to_csv("korea_bonus_issue_stock_split_events.csv", index=False, encoding="utf-8-sig")

    # 분리 저장
    events[events["event_type"] == "bonus_issue"].to_csv(
        "korea_bonus_issue_events.csv", index=False, encoding="utf-8-sig"
    )
    events[events["event_type"] == "stock_split"].to_csv(
        "korea_stock_split_events.csv", index=False, encoding="utf-8-sig"
    )

    summary = (
        events.groupby(["market", "event_name_kr"])
        .size()
        .reset_index(name="count")
        .sort_values(["event_name_kr", "market"])
    )

    print("\n=== 요약 ===")
    print(summary.to_string(index=False))

    print("\n=== 샘플 20건 ===")
    print(events.head(20).to_string(index=False))

    print("\n저장 완료:")
    print("-bonus_issue_only2016to2026.csv")
    print("- korea_bonus_issue_events2016to2026.csv")
    print("- korea_stock_split_events2016to2026.csv")


if __name__ == "__main__":
    main()