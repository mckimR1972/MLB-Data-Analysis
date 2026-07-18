"""
MLB Statcast 일일 자동 수집 파이프라인
═══════════════════════════════════════
매일 자정 이후 실행 → 전일 경기 데이터를 Oracle DB(TMP_TABLE)에 적재

PK: GAME_PK + AT_BAT_NUMBER + PITCH_NUMBER
    → 하나의 경기에서 타석 번호 + 투구 순번은 절대 중복 불가

사용법:
    1) 일반 실행:   python statcast_pipeline.py
    2) 특정 날짜:   python statcast_pipeline.py --date 2025-07-13
    3) 범위 보충:   python statcast_pipeline.py --backfill 2025-03-27 2025-07-13
"""

import oracledb
import numpy as np
import pandas as pd
from pybaseball import statcast
from datetime import datetime, timedelta
import logging
import os
import argparse
import time

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════

DB_USER = "C##AKSEN"
DB_PASS = "020304"
DB_DSN  = "localhost:1521/xe"
TABLE_NAME = "TMP_TABLE"

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"pipeline_{datetime.now():%Y%m}.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
    handlers=[
        logging.FileHandler(
            os.path.join(LOG_DIR, f"pipeline_{datetime.now():%Y%m}.log"),
            encoding="utf-8",
        ),
        logging.StreamHandler(),  # 콘솔에도 동시 출력
    ],
)
# basicConfig의 handlers와 filename 충돌 방지를 위해 재설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, f"pipeline_{datetime.now():%Y%m}.log"),
        encoding="utf-8",
    )
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


# ══════════════════════════════════════════════════════════
# pybaseball → TMP_TABLE 컬럼 매핑
# ══════════════════════════════════════════════════════════
# pybaseball의 statcast()가 반환하는 컬럼명(소문자)
# → Oracle TMP_TABLE 컬럼명(대문자) 매핑
#
# 대부분은 단순 대문자 변환이지만,
# 이름 자체가 다른 컬럼들이 있어서 명시적 rename이 필요합니다.

COLUMN_RENAME = {
    # pybaseball 컬럼명     →  Oracle TMP_TABLE 컬럼명
    "description":                    "PITCH_DESCRIPTION",
    "events":                         "PITCH_EVENTS",
    "type":                           "SBX_TYPE",
    "estimated_ba_using_speedangle":  "X_BA",
    "estimated_woba_using_speedangle":"XW_OBA",
    "delta_home_win_exp":             "HOME_WIN_EXP",
}

# TMP_TABLE에 존재하는 66개 컬럼 (순서 보존)
TABLE_COLUMNS = [
    "PITCH_TYPE", "RELEASE_SPEED", "RELEASE_POS_X", "RELEASE_POS_Y",
    "RELEASE_POS_Z", "RELEASE_SPIN_RATE", "PFX_X", "PFX_Z",
    "PLATE_X", "PLATE_Z", "VX0", "VY0", "VZ0",
    "AX", "AY", "AZ", "SPIN_AXIS",
    "BATTER", "PITCHER", "GAME_PK", "GAME_DATE",
    "STRIKES", "BALLS", "OUTS_WHEN_UP",
    "ON_1B", "ON_2B", "ON_3B",
    "AT_BAT_NUMBER", "PITCH_NUMBER",
    "PITCH_DESCRIPTION", "PITCH_EVENTS", "DES",
    "STAND", "P_THROWS", "HOME_TEAM", "AWAY_TEAM",
    "SBX_TYPE", "HIT_LOCATION", "BB_TYPE",
    "INNING", "INNING_TOPBOT",
    "HC_X", "HC_Y", "HIT_DISTANCE_SC",
    "LAUNCH_SPEED", "LAUNCH_ANGLE",
    "X_BA", "XW_OBA", "WOBA_VALUE", "WOBA_DENOM", "BABIP_VALUE",
    "LAUNCH_SPEED_ANGLE",
    "HOME_SCORE", "POST_HOME_SCORE", "POST_AWAY_SCORE", "AWAY_SCORE",
    "BAT_SPEED", "SWING_LENGTH",
    "ESTIMATED_SLG_USING_SPEEDANGLE",
    "HOME_WIN_EXP", "ARM_ANGLE",
    "ATTACK_ANGLE", "ATTACK_DIRECTION",
    "INTERCEPT_X", "INTERCEPT_Y", "SWING_PATH_TILT",
]

# PK 컬럼
PK_COLUMNS = ["GAME_PK", "AT_BAT_NUMBER", "PITCH_NUMBER"]



# ══════════════════════════════════════════════════════════
# 데이터 수집
# ══════════════════════════════════════════════════════════

def fetch_statcast_data(target_date: str) -> pd.DataFrame:
    """
    pybaseball로 Statcast 데이터를 가져와서
    TMP_TABLE 스키마에 맞게 변환합니다.

    Parameters
    ----------
    target_date : str — "YYYY-MM-DD" 형식

    Returns
    -------
    pd.DataFrame — TMP_TABLE 컬럼 순서에 맞춘 DataFrame
    """
    logger.info(f"[FETCH] {target_date} 데이터 수집 시작...")

    try:
        df = statcast(start_dt='2026-03-01', end_dt=target_date)
        df = df[df['game_type'] == 'R']
    except Exception as e:
        logger.error(f"[FETCH] API 호출 실패: {e}")
        return pd.DataFrame()

    if df is None or len(df) == 0:
        logger.info(f"[FETCH] {target_date} — 경기 없음 (빈 DataFrame)")
        return pd.DataFrame()

    logger.info(f"[FETCH] 원본: {len(df)}행 × {len(df.columns)}열")

    # ── Step 1: 이름이 다른 컬럼 rename ──
    df = df.rename(columns=COLUMN_RENAME)

    # ── Step 2: 나머지 컬럼은 대문자로 변환 ──
    df.columns = [c.upper() if c not in COLUMN_RENAME.values() else c
                  for c in df.columns]

    # ── Step 3: TMP_TABLE에 있는 컬럼만 추출 (순서 맞춤) ──
    # 테이블에는 있지만 API에 없는 컬럼 → NaN으로 채움
    for col in TABLE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
            logger.warning(f"[FETCH] API에 '{col}' 컬럼 없음 → NaN 채움")

    df = df[TABLE_COLUMNS].copy()

    # ── Step 4: 타입 정리 ──
    # GAME_DATE: datetime → date
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date

    # 정수형 컬럼: float → int (NaN이 있으면 유지)
    int_cols = [
        "RELEASE_SPIN_RATE", "BATTER", "PITCHER", "GAME_PK",
        "STRIKES", "BALLS", "OUTS_WHEN_UP",
        "ON_1B", "ON_2B", "ON_3B",
        "AT_BAT_NUMBER", "PITCH_NUMBER",
        "HIT_LOCATION", "INNING",
        "HIT_DISTANCE_SC", "LAUNCH_ANGLE",
        "WOBA_DENOM", "LAUNCH_SPEED_ANGLE",
        "HOME_SCORE", "POST_HOME_SCORE", "POST_AWAY_SCORE", "AWAY_SCORE",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Step 5: PK 컬럼에 NULL이 있는 행 제거 ──
    before = len(df)
    df = df.dropna(subset=PK_COLUMNS)
    df = df.drop_duplicates(subset=PK_COLUMNS, keep="last")
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"[FETCH] PK 컬럼 NULL → {dropped}행 제거")

    # PK 컬럼을 정수로 확정
    for col in PK_COLUMNS:
        df[col] = df[col].astype(int)

    logger.info(f"[FETCH] 최종: {len(df)}행 × {len(df.columns)}열")
    return df


# ══════════════════════════════════════════════════════════
# DB 적재
# ══════════════════════════════════════════════════════════


def insert_to_oracle(df: pd.DataFrame):
    """
    DataFrame을 TMP_TABLE에 MERGE (upsert) 방식으로 적재합니다.
    PK 중복 시 기존 데이터를 UPDATE, 새 데이터는 INSERT.
    """
    if df.empty:
        logger.info("[DB] 빈 DataFrame — 적재 스킵")
        return 0

    conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
    cursor = conn.cursor()

    try:
        # ── MERGE SQL 생성 ──
        # MERGE는 PK 기준으로 존재 여부를 판단해서
        # 있으면 UPDATE, 없으면 INSERT 합니다.
        non_pk_cols = [c for c in TABLE_COLUMNS if c not in PK_COLUMNS]

        # 바인드 변수 매핑: :1, :2, ... (TABLE_COLUMNS 순서)
        bind_map = {col: f":{i+1}" for i, col in enumerate(TABLE_COLUMNS)}

        merge_sql = f"""
            MERGE INTO {TABLE_NAME} tgt
            USING (
                SELECT {', '.join(f'{bind_map[c]} AS {c}' for c in TABLE_COLUMNS)}
                FROM DUAL
            ) src
            ON ({' AND '.join(f'tgt.{c} = src.{c}' for c in PK_COLUMNS)})
            WHEN MATCHED THEN UPDATE SET
                {', '.join(f'tgt.{c} = src.{c}' for c in non_pk_cols)}
            WHEN NOT MATCHED THEN INSERT
                ({', '.join(TABLE_COLUMNS)})
                VALUES ({', '.join(f'src.{c}' for c in TABLE_COLUMNS)})
        """

        # ── 데이터 변환: NaN → None ──
        data = []
        for _, row in df.iterrows():
            record = []
            for col in TABLE_COLUMNS:
                val = row[col]
                if pd.isna(val):
                    record.append(None)
                elif isinstance(val, (np.integer,)):
                    record.append(int(val))
                elif isinstance(val, (np.floating,)):
                    record.append(float(val))
                else:
                    record.append(val)
            data.append(record)

        # ── 배치 실행 ──
        BATCH_SIZE = 500
        total_inserted = 0

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            cursor.executemany(merge_sql, batch)
            conn.commit()
            total_inserted += len(batch)
            logger.info(
                f"[DB] 배치 {i//BATCH_SIZE + 1}: "
                f"{len(batch)}행 MERGE 완료 "
                f"({total_inserted}/{len(data)})"
            )

        logger.info(f"[DB] 총 {total_inserted}행 적재 완료")
        return total_inserted

    except Exception as e:
        conn.rollback()
        logger.error(f"[DB] 적재 실패: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


# ══════════════════════════════════════════════════════════
# 날짜별 수집 상태 확인
# ══════════════════════════════════════════════════════════

def get_existing_dates() -> set:
    """DB에 이미 적재된 GAME_DATE 목록을 조회합니다."""
    conn = oracledb.connect(user=DB_USER, password=DB_PASS, dsn=DB_DSN)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT DISTINCT GAME_DATE FROM {TABLE_NAME}")
        dates = {row[0] for row in cursor.fetchall()}
        return dates
    except Exception:
        return set()
    finally:
        cursor.close()
        conn.close()


# ══════════════════════════════════════════════════════════
# 메인 실행 함수
# ══════════════════════════════════════════════════════════

def run_single_date(target_date: str):
    """단일 날짜 수집 & 적재"""
    logger.info(f"{'='*60}")
    logger.info(f"[PIPELINE] 대상 날짜: {target_date}")

    df = fetch_statcast_data(target_date)
    if df.empty:
        logger.info(f"[PIPELINE] {target_date} — 데이터 없음, 스킵")
        return 0

    count = insert_to_oracle(df)
    logger.info(f"[PIPELINE] {target_date} 완료 — {count}행 적재")
    return count


def run_daily():
    """
    일일 파이프라인 (스케줄러용).
    전일 데이터를 수집하고, 빠진 날짜가 있으면 자동 보충합니다.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"{'='*60}")
    logger.info(f"[DAILY] 파이프라인 시작 — {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ── 빠진 날짜 자동 보충 (backfill) ──
    try:
        existing_dates = get_existing_dates()
        if existing_dates:
            last_date = max(existing_dates)
            yesterday_date = datetime.strptime(yesterday, "%Y-%m-%d").date()

            # 마지막 적재일 ~ 어제 사이에 빈 날짜 탐색
            current = last_date + timedelta(days=1)
            missing = []
            while current <= yesterday_date:
                if current not in existing_dates:
                    missing.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

            if missing:
                logger.info(f"[BACKFILL] 빠진 날짜 {len(missing)}일 발견: "
                            f"{missing[0]} ~ {missing[-1]}")
                for d in missing:
                    run_single_date(d)
                    time.sleep(5)  # API 부하 방지
            else:
                logger.info("[BACKFILL] 빠진 날짜 없음")
    except Exception as e:
        logger.warning(f"[BACKFILL] 보충 확인 중 오류 (계속 진행): {e}")

    # ── 전일 데이터 수집 ──
    run_single_date(yesterday)

    logger.info(f"[DAILY] 파이프라인 종료 — {datetime.now():%Y-%m-%d %H:%M:%S}")


def run_backfill(start_date: str, end_date: str):
    """
    범위 지정 보충 수집.
    이미 DB에 있는 날짜는 MERGE로 처리되므로 중복 걱정 없음.
    """
    logger.info(f"[BACKFILL] {start_date} ~ {end_date} 보충 수집 시작")

    existing = get_existing_dates()
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    total = 0
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")

        if current in existing:
            logger.info(f"[BACKFILL] {date_str} — 이미 존재, 스킵")
        else:
            count = run_single_date(date_str)
            total += count
            time.sleep(5)  # Baseball Savant 서버 부하 방지

        current += timedelta(days=1)

    logger.info(f"[BACKFILL] 보충 완료 — 총 {total}행 적재")


# ══════════════════════════════════════════════════════════
# CLI 엔트리포인트
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLB Statcast 일일 수집 파이프라인"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="특정 날짜 수집 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--backfill", nargs=2, metavar=("START", "END"),
        help="범위 보충 수집 (YYYY-MM-DD YYYY-MM-DD)"
    )

    args = parser.parse_args()

    if args.backfill:
        run_backfill(args.backfill[0], args.backfill[1])
    elif args.date:
        run_single_date(args.date)
    else:
        run_daily()