# -*- coding: utf-8 -*-
"""
KIS 국내주식 전종목 일별 시가/종가 수집
- 대상: KOSPI + KOSDAQ 상장주식 (ETF 포함임)
- 기간: 2016-01-01 ~ 오늘
- 저장: ./KOSPI_KOSDAQ_Price/price_cache_kis/{종목코드}.xlsx
- 업데이트 가능
- 종목리스트: KIS master file(kospi_code.mst / kosdaq_code.mst) 사용

고려한 사항:
1) 전종목 + 2016년부터 전체 백필은 호출 수가 많아서 오래 걸린다.
2) 한 번 호출당 최대 100건이므로 캘린더 기준 120일 단위로 잘라서 요청한다.
3) FID_ORG_ADJ_PRC: "0" = 수정주가 / "1" = 원주가
"""

from __future__ import annotations
import io
import json
import time
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import requests


APP_KEY    = None
APP_SECRET = None
START_DATE = "20160101"
TODAY = datetime.now().strftime("%Y%m%d")

ADJ_PRC = "1"

ROOT_DIR = Path("./Desktop/KOSPI_KOSDAQ_Price")
CACHE_DIR = ROOT_DIR / "price_cache_kis"
META_DIR = ROOT_DIR / "_전 종목 리스트 및 메타데이터"
TOKEN_FILE = META_DIR / "kis_token.json"

# 100건 제한 우회
CHUNK_DAYS = 120

# 과도한 호출 방지
REQUEST_SLEEP = 0.01

#CSV로 변경가능
SAVE_FORMAT = "xlsx" 

# 기존 파일 있으면 마지막 날짜 다음날부터만 추가
INCREMENTAL = True

EXCLUDE_PREFERRED = False 
EXCLUDE_SPAC = False 


# =========================
# 유틸
# =========================
def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def ymd_to_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d")


def dt_to_ymd(dt_obj: datetime) -> str:
    return dt_obj.strftime("%Y%m%d")


def file_for_code(code: str) -> Path:
    if SAVE_FORMAT.lower() == "csv":
        return CACHE_DIR / f"{code}.csv"
    return CACHE_DIR / f"{code}.xlsx"


class KISClient:
    def __init__(self, app_key: str, app_secret: str, base_url: str) -> None:
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.access_token: Optional[str] = None
        self.token_expire_at: Optional[datetime] = None

    def _load_cached_token(self) -> bool:
        if not TOKEN_FILE.exists():
            return False
        try:
            data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
            token = data.get("access_token")
            expire_at = data.get("expire_at")
            if not token or not expire_at:
                return False

            expire_dt = datetime.fromisoformat(expire_at)
            if datetime.now() >= expire_dt - timedelta(minutes=1):
                return False

            self.access_token = token
            self.token_expire_at = expire_dt
            return True
        except Exception:
            return False

    def _save_cached_token(self, token: str, expires_in: int) -> None:
        expire_dt = datetime.now() + timedelta(seconds=int(expires_in))
        payload = {
            "access_token": token,
            "expire_at": expire_dt.isoformat(),
        }
        TOKEN_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.access_token = token
        self.token_expire_at = expire_dt

    def authenticate(self) -> None:
        if self._load_cached_token():
            return

        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        headers = {"content-type": "application/json; charset=UTF-8"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "access_token" not in data:
            raise RuntimeError(f"토큰 발급 실패: {data}")

        self._save_cached_token(
            token=data["access_token"],
            expires_in=int(data.get("expires_in", 24 * 3600)),
        )

    def _headers(self, tr_id: str) -> dict:
        self.authenticate()
        return {
            "content-type": "application/json; charset=UTF-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }

    def inquire_daily_itemchartprice(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adj_prc: str = "1",
    ) -> pd.DataFrame:
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": adj_prc,
        }

        headers = self._headers(tr_id="FHKST03010100")
        resp = self.session.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rt_cd = data.get("rt_cd")
        msg1 = data.get("msg1", "")
        if rt_cd != "0":
            raise RuntimeError(f"[{code}] API 오류: rt_cd={rt_cd}, msg1={msg1}")

        rows = data.get("output2", []) or []
        if not rows:
            return pd.DataFrame(columns=["date", "open", "close"])

        out = pd.DataFrame(rows)

        # KIS 응답 컬럼명 기준
        # stck_bsop_date: 영업일자
        # stck_oprc: 시가
        # stck_clpr: 종가
        rename_map = {
            "stck_bsop_date": "date",
            "stck_oprc": "open",
            "stck_clpr": "close",
        }
        missing = [k for k in rename_map if k not in out.columns]
        if missing:
            raise RuntimeError(f"[{code}] 예상 컬럼 없음: {missing}, 실제컬럼={list(out.columns)}")

        out = out[list(rename_map.keys())].rename(columns=rename_map)
        out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
        out["open"] = pd.to_numeric(out["open"], errors="coerce")
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out = out.dropna(subset=["date"])
        out = out.sort_values("date").reset_index(drop=True)

        return out


# =========================
# KOSPI / KOSDAQ master
# =========================
def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def parse_kospi_master() -> pd.DataFrame:
    url = "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip"
    content = download_bytes(url)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        mst_name = [n for n in zf.namelist() if n.endswith(".mst")][0]
        raw = zf.read(mst_name).decode("cp949", errors="ignore").splitlines()

    part1_rows = []
    part2_rows = []

    for row in raw:
        rf1 = row[: len(row) - 228]
        rf2 = row[-228:]

        code = rf1[0:9].rstrip()
        std_code = rf1[9:21].rstrip()
        name = rf1[21:].strip()

        part1_rows.append([code, std_code, name])
        part2_rows.append(rf2)

    df1 = pd.DataFrame(part1_rows, columns=["단축코드", "표준코드", "한글명"])

    widths = [
        2, 1, 4, 4, 4,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 9, 5, 5, 1,
        1, 1, 2, 1, 1,
        1, 2, 2, 2, 3,
        1, 3, 12, 12, 8,
        15, 21, 2, 7, 1,
        1, 1, 1, 1, 9,
        9, 9, 5, 9, 8,
        9, 3, 1, 1, 1,
    ]
    cols = [
        "그룹코드", "시가총액규모", "지수업종대분류", "지수업종중분류", "지수업종소분류",
        "제조업", "저유동성", "지배구조지수종목", "KOSPI200섹터업종", "KOSPI100",
        "KOSPI50", "KRX", "ETP", "ELW발행", "KRX100",
        "KRX자동차", "KRX반도체", "KRX바이오", "KRX은행", "SPAC",
        "KRX에너지화학", "KRX철강", "단기과열", "KRX미디어통신", "KRX건설",
        "Non1", "KRX증권", "KRX선박", "KRX섹터_보험", "KRX섹터_운송",
        "SRI", "기준가", "매매수량단위", "시간외수량단위", "거래정지",
        "정리매매", "관리종목", "시장경고", "경고예고", "불성실공시",
        "우회상장", "락구분", "액면변경", "증자구분", "증거금비율",
        "신용가능", "신용기간", "전일거래량", "액면가", "상장일자",
        "상장주수", "자본금", "결산월", "공모가", "우선주",
        "공매도과열", "이상급등", "KRX300", "KOSPI", "매출액",
        "영업이익", "경상이익", "당기순이익", "ROE", "기준년월",
        "시가총액", "그룹사코드", "회사신용한도초과", "담보대출가능", "대주가능",
    ]
    df2 = pd.read_fwf(io.StringIO("\n".join(part2_rows)), widths=widths, names=cols)

    df = pd.concat([df1, df2], axis=1)
    df["market"] = "KOSPI"
    return df


def parse_kosdaq_master() -> pd.DataFrame:
    """
    KIS 공식 샘플 구조 기반.
    """
    url = "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip"
    content = download_bytes(url)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        mst_name = [n for n in zf.namelist() if n.endswith(".mst")][0]
        raw = zf.read(mst_name).decode("cp949", errors="ignore").splitlines()

    part1_rows = []
    part2_rows = []

    for row in raw:
        rf1 = row[: len(row) - 222]
        rf2 = row[-222:]

        code = rf1[0:9].rstrip()
        std_code = rf1[9:21].rstrip()
        name = rf1[21:].strip()

        part1_rows.append([code, std_code, name])
        part2_rows.append(rf2)

    df1 = pd.DataFrame(part1_rows, columns=["단축코드", "표준코드", "한글명"])

    widths = [
        2, 1,
        4, 4, 4, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 9,
        5, 5, 1, 1, 1,
        2, 1, 1, 1, 2,
        2, 2, 3, 1, 3,
        12, 12, 8, 15, 21,
        2, 7, 1, 1, 1,
        1, 9, 9, 9, 5,
        9, 8, 9, 3, 1,
        1, 1,
    ]
    cols = [
        "증권그룹구분코드", "시가총액규모",
        "지수업종대분류", "지수업종중분류", "지수업종소분류", "벤처기업여부", "저유동성종목여부",
        "KRX종목여부", "ETP상품구분코드", "KRX100종목여부",
        "KRX자동차여부", "KRX반도체여부", "KRX바이오여부", "KRX은행여부", "기업인수목적회사여부",
        "KRX에너지화학여부", "KRX철강여부", "단기과열종목구분코드", "KRX미디어통신여부",
        "KRX건설여부", "투자주의환기종목여부", "KRX증권구분", "KRX선박구분",
        "KRX섹터보험여부", "KRX섹터운송여부", "KOSDAQ150지수여부", "주식기준가",
        "정규시장매매수량단위", "시간외시장매매수량단위", "거래정지여부", "정리매매여부",
        "관리종목여부", "시장경고구분코드", "시장경고위험예고여부", "불성실공시여부",
        "우회상장여부", "락구분코드", "액면가변경구분코드", "증자구분코드", "증거금비율",
        "신용주문가능여부", "신용기간", "전일거래량", "주식액면가", "주식상장일자", "상장주수천",
        "자본금", "결산월", "공모가격", "우선주구분코드", "공매도과열종목여부",
        "이상급등종목여부", "KRX300종목여부", "매출액", "영업이익", "경상이익",
        "단기순이익", "ROE", "기준년월", "전일기준시가총액억", "그룹사코드",
        "회사신용한도초과여부", "담보대출가능여부", "대주가능여부",
    ]
    df2 = pd.read_fwf(io.StringIO("\n".join(part2_rows)), widths=widths, names=cols)

    df = pd.concat([df1, df2], axis=1)
    df["market"] = "KOSDAQ"
    return df


def get_all_listed_stocks() -> pd.DataFrame:
    kospi = parse_kospi_master()
    kosdaq = parse_kosdaq_master()
    df = pd.concat([kospi, kosdaq], ignore_index=True)

    # 6자리 종목코드만
    df["code"] = df["단축코드"].astype(str).str.zfill(6).str[:6]
    df["name"] = df.get("한글명", df.get("한글종목명", "")).astype(str)

    # 빈 코드 제거
    df = df[df["code"].str.fullmatch(r"\d{6}", na=False)].copy()

    if EXCLUDE_PREFERRED:
        # KOSPI: 우선주 컬럼, KOSDAQ: 우선주구분코드
        preferred_flag = (
            (df.get("우선주", "").astype(str).str.strip() == "Y")
            | (df.get("우선주구분코드", "").astype(str).str.strip().ne(""))
        )
        df = df[~preferred_flag].copy()

    if EXCLUDE_SPAC:
        spac_flag = (
            (df.get("SPAC", "").astype(str).str.strip() == "Y")
            | (df.get("기업인수목적회사여부", "").astype(str).str.strip() == "Y")
        )
        df = df[~spac_flag].copy()

    df = df[["code", "name", "market"]].drop_duplicates("code").sort_values(["market", "code"])
    return df.reset_index(drop=True)


# =========================
# 저장 / 로드
# =========================
def read_existing_price_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "close", "code", "name", "market"])

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def save_price_file(path: Path, df: pd.DataFrame) -> None:
    df = df.sort_values("date").reset_index(drop=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(path, index=False)


def fetch_range_chunked(
    client: KISClient,
    code: str,
    start_date: str,
    end_date: str,
    adj_prc: str,
) -> pd.DataFrame:
    start_dt = ymd_to_dt(start_date)
    end_dt = ymd_to_dt(end_date)

    all_parts = []
    cur_start = start_dt

    while cur_start <= end_dt:
        cur_end = min(cur_start + timedelta(days=CHUNK_DAYS - 1), end_dt)

        part = client.inquire_daily_itemchartprice(
            code=code,
            start_date=dt_to_ymd(cur_start),
            end_date=dt_to_ymd(cur_end),
            adj_prc=adj_prc,
        )

        if not part.empty:
            all_parts.append(part)

        cur_start = cur_end + timedelta(days=1)
        time.sleep(REQUEST_SLEEP)

    if not all_parts:
        return pd.DataFrame(columns=["date", "open", "close"])

    out = pd.concat(all_parts, ignore_index=True)
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def update_one_symbol(
    client: KISClient,
    code: str,
    name: str,
    market: str,
    global_start_date: str,
    global_end_date: str,
    adj_prc: str,
) -> None:
    path = file_for_code(code)
    old_df = read_existing_price_file(path)

    if INCREMENTAL and not old_df.empty:
        last_date = old_df["date"].max()
        start_dt = last_date + timedelta(days=1)
        if start_dt > ymd_to_dt(global_end_date):
            print(f"[SKIP] {code} {name} - already up to date")
            return
        fetch_start = dt_to_ymd(start_dt)
    else:
        fetch_start = global_start_date

    new_df = fetch_range_chunked(
        client=client,
        code=code,
        start_date=fetch_start,
        end_date=global_end_date,
        adj_prc=adj_prc,
    )

    if new_df.empty and not old_df.empty:
        print(f"[NO NEW] {code} {name}")
        return

    new_df["code"] = code
    new_df["name"] = name
    new_df["market"] = market

    final_df = pd.concat([old_df, new_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["date"], keep="last")
    final_df = final_df.sort_values("date").reset_index(drop=True)

    save_price_file(path, final_df)
    print(f"[OK] {code} {name} -> {path.name}, rows={len(final_df)}")


def main() -> None:
    ensure_dirs()

    print("1) 종목 마스터 다운로드/파싱...")
    stocks = get_all_listed_stocks()
    stocks.to_csv(META_DIR / "listed_stocks_kis.csv", index=False, encoding="utf-8-sig")
    print(f"   total stocks: {len(stocks)}")

    print("2) KIS 인증...")
    client = KISClient(APP_KEY, APP_SECRET, BASE_URL)
    client.authenticate()
    print("   token ready")

    print("3) 시세 수집 시작...")
    for i, row in stocks.iterrows():
        code = row["code"]
        name = row["name"]
        market = row["market"]

        try:
            print(f"[{i+1}/{len(stocks)}] {market} {code} {name}")
            update_one_symbol(
                client=client,
                code=code,
                name=name,
                market=market,
                global_start_date=START_DATE,
                global_end_date=TODAY,
                adj_prc=ADJ_PRC,
            )
        except KeyboardInterrupt:
            print("사용자 중단")
            break
        except Exception as e:
            print(f"[ERR] {code} {name}: {e}")
            time.sleep(0.01)

    print("완료")


if __name__ == "__main__":
    main()