# -*- coding: utf-8 -*-
"""
경영 위기 조기 경보 시스템 (Pure ML 버전)
- 4단계 경보 시스템: 안전(Green) -> 주의(Yellow) -> 경고(Orange) -> 위험(Red)
- 순수 LightGBM 기반 머신러닝 예측 (룰 기반 제거)
- 모든 데이터셋 특성 활용 (set1, set2, set3, rental, flow)
- 세대별 매출 변화 분석 포함
- 분석 기간: 2023-01 ~ 2024-12
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import shap

# OpenAI API (선택사항)
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import os
    load_dotenv()  # .env 파일 로드
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI 라이브러리가 설치되지 않았습니다. LLM 분석 기능이 비활성화됩니다.")
    print("   설치: pip install openai python-dotenv")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class MLEarlyWarningSystem:
    """Pure ML 기반 경영 위기 조기 경보 시스템"""

    def __init__(self, data_path='./data/', start_date='2023-01', end_date='2024-12'):
        """초기화"""
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date, format='%Y-%m')
        self.end_date = pd.to_datetime(end_date, format='%Y-%m')

        self.merchant_data = None
        self.sales_data = None
        self.customer_data = None
        self.rental_data = None
        self.flow_data = None
        self.merged_data = None
        self.model = None
        self.feature_cols = []
        self.label_encoders = {}

        # 경보 레벨 정의
        self.WARNING_LEVELS = {
            0: {'name': '안전', 'color': 'green', 'emoji': '🟢'},
            1: {'name': '주의', 'color': 'yellow', 'emoji': '🟡'},
            2: {'name': '경고', 'color': 'orange', 'emoji': '🟠'},
            3: {'name': '위험', 'color': 'red', 'emoji': '🔴'}
        }

        # LLM 설정
        self.use_llm = OPENAI_AVAILABLE and os.getenv('USE_LLM_ANALYSIS', 'false').lower() == 'true'
        self.openai_client = None
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

        if self.use_llm:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your-api-key-here':
                self.openai_client = OpenAI(api_key=api_key)
                print(f"✅ LLM 분석 활성화 (모델: {self.openai_model})")
            else:
                self.use_llm = False
                print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다. LLM 분석이 비활성화됩니다.")
        else:
            print("ℹ️  LLM 분석 비활성화 (기본 분석 모드)")

    def load_data(self):
        """데이터 로드"""
        print("=" * 80)
        print(f"📊 데이터 로드 중 ({self.start_date.strftime('%Y-%m')} ~ {self.end_date.strftime('%Y-%m')})")
        print("=" * 80)

        # 가맹점 정보
        self.merchant_data = pd.read_csv(f'{self.data_path}big_data_set1_f_v2.csv', encoding='utf-8-sig')
        print(f"✓ 가맹점 정보: {len(self.merchant_data):,}개")

        # 매출 데이터
        self.sales_data = pd.read_csv(f'{self.data_path}big_data_set2_f_sorted.csv', encoding='utf-8-sig')
        self.sales_data['TA_YM'] = pd.to_datetime(self.sales_data['TA_YM'], format='%Y%m')

        # 기간 필터링
        self.sales_data = self.sales_data[
            (self.sales_data['TA_YM'] >= self.start_date) &
            (self.sales_data['TA_YM'] <= self.end_date)
        ]
        print(f"✓ 매출 데이터: {len(self.sales_data):,}건 ({self.start_date.strftime('%Y-%m')} ~ {self.end_date.strftime('%Y-%m')})")

        # 고객 데이터
        self.customer_data = pd.read_csv(f'{self.data_path}big_data_set3_f_sorted.csv', encoding='utf-8-sig')
        self.customer_data['TA_YM'] = pd.to_datetime(self.customer_data['TA_YM'], format='%Y%m')

        # 기간 필터링
        self.customer_data = self.customer_data[
            (self.customer_data['TA_YM'] >= self.start_date) &
            (self.customer_data['TA_YM'] <= self.end_date)
        ]
        print(f"✓ 고객 데이터: {len(self.customer_data):,}건")

        # 임대료 데이터
        self.rental_data = pd.read_csv(f'{self.data_path}rental_p.csv', encoding='utf-8-sig')
        print(f"✓ 임대료 데이터: {len(self.rental_data):,}건")

        # 유동인구 데이터
        self.flow_data = pd.read_csv(f'{self.data_path}flow_f.csv', encoding='utf-8-sig')
        print(f"✓ 유동인구 데이터: {len(self.flow_data):,}건")
        print()

    def merge_all_data(self):
        """모든 데이터 통합"""
        print("🔗 데이터 통합 중...")

        # 매출 + 고객 데이터
        self.merged_data = pd.merge(
            self.sales_data,
            self.customer_data,
            on=['ENCODED_MCT', 'TA_YM'],
            how='inner'
        )

        # 가맹점 정보 추가
        self.merged_data = pd.merge(
            self.merged_data,
            self.merchant_data[['ENCODED_MCT', 'MCT_NM', 'HPSN_MCT_ZCD_NM', 'HPSN_MCT_BZN_CD_NM',
                                'MCT_SIGUNGU_NM', 'LEGAL_DONG', 'ARE_D']],
            on='ENCODED_MCT',
            how='left'
        )

        # 임대료 데이터 (분기 -> 월로 확장)
        self.rental_data['기간(분기)'] = pd.to_datetime(self.rental_data['기간(분기)'], format='%Y%m')
        rental_expanded = []
        for _, row in self.rental_data.iterrows():
            for i in range(3):
                new_row = row.copy()
                new_row['TA_YM'] = row['기간(분기)'] + pd.DateOffset(months=i)
                rental_expanded.append(new_row)

        rental_df = pd.DataFrame(rental_expanded)

        # 기간 필터링
        rental_df = rental_df[
            (rental_df['TA_YM'] >= self.start_date) &
            (rental_df['TA_YM'] <= self.end_date)
        ]

        self.merged_data = pd.merge(
            self.merged_data,
            rental_df[['TA_YM', '행정구역', '전체(단위:원/평)', '1층(단위:원/평)', '1층 외(단위:원/평)']],
            left_on=['TA_YM', 'LEGAL_DONG'],
            right_on=['TA_YM', '행정구역'],
            how='left'
        )
        
        # 유동인구 데이터 (분기 -> 월로 확장)
        self.flow_data['기간(분기)'] = pd.to_datetime(self.flow_data['기간(분기)'], format='%Y%m')
        flow_expanded = []
        for _, row in self.flow_data.iterrows():
            for i in range(3):
                new_row = row.copy()
                new_row['TA_YM'] = row['기간(분기)'] + pd.DateOffset(months=i)
                flow_expanded.append(new_row)

        flow_df = pd.DataFrame(flow_expanded)

        # 기간 필터링
        flow_df = flow_df[
            (flow_df['TA_YM'] >= self.start_date) &
            (flow_df['TA_YM'] <= self.end_date)
        ]

        self.merged_data = pd.merge(
            self.merged_data,
            flow_df[['TA_YM', '행적구역', '유동인구(단위:명/1ha)', '주거인구(단위:명/1ha)', '직장인구(단위:명/1ha)']],
            left_on=['TA_YM', 'LEGAL_DONG'],
            right_on=['TA_YM', '행적구역'],
            how='left'
        )
        
        print(f"임대료 \n {self.merged_data.head()}")

        print(f"✓ 통합 데이터: {len(self.merged_data):,}건")
        print(f"  기간: {self.merged_data['TA_YM'].min().strftime('%Y-%m')} ~ {self.merged_data['TA_YM'].max().strftime('%Y-%m')}")
        print(f"  가맹점 수: {self.merged_data['ENCODED_MCT'].nunique():,}개\n")

    def extract_numeric_value(self, value_str):
        """구간 문자열에서 중간값 추출"""
        if pd.isna(value_str) or value_str == '':
            return np.nan

        value_str = str(value_str)

        if '90%초과' in value_str or '하위 10%' in value_str:
            return 95.0

        if '_' in value_str and '%' in value_str:
            parts = value_str.split('_')
            if len(parts) > 1:
                range_part = parts[1].replace('%', '')
                if '-' in range_part:
                    low, high = map(float, range_part.split('-'))
                    return (low + high) / 2
                elif '이하' in range_part:
                    return float(range_part.replace('이하', '')) / 2

        return np.nan

    def create_comprehensive_features(self):
        """모든 데이터셋을 활용한 종합 특성 생성"""
        print("🔧 종합 특성 엔지니어링 중...")

        # 정렬
        self.merged_data = self.merged_data.sort_values(['ENCODED_MCT', 'TA_YM'])

        # === 1. 기본 시계열 정보 ===
        self.merged_data['운영개월수'] = self.merged_data.groupby('ENCODED_MCT').cumcount() + 1
        self.merged_data['월'] = self.merged_data['TA_YM'].dt.month
        self.merged_data['분기'] = self.merged_data['TA_YM'].dt.quarter
        self.merged_data['연도'] = self.merged_data['TA_YM'].dt.year

        # === 2. 매출 데이터 (set2) 숫자형 변환 ===
        sales_cols = [
            'MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT',
            'M12_SME_RY_SAA_PCE_RT', 'M12_SME_BZN_SAA_PCE_RT', 'DLV_SAA_RAT'
        ]

        for col in sales_cols:
            if col in self.merged_data.columns:
                self.merged_data[f'{col}_num'] = self.merged_data[col].apply(self.extract_numeric_value)

        # 승인율
        if 'APV_CE_RAT' in self.merged_data.columns:
            self.merged_data['APV_CE_RAT'] = self.merged_data['APV_CE_RAT'].replace('', np.nan)
            self.merged_data['APV_CE_RAT_num'] = self.merged_data['APV_CE_RAT'].apply(
                lambda x: self.extract_numeric_value(x) if pd.notna(x) else np.nan
            )

        # === 3. 고객 데이터 (set3) 처리 ===

        # -999999.9 값을 null로 처리
        self.merged_data = self.merged_data.replace(-999999.9, np.nan)

        # 3-1. 성별/연령별 비율 (숫자형 변환)
        demographic_cols = [
            'M12_MAL_1020_RAT', 'M12_MAL_30_RAT', 'M12_MAL_40_RAT', 'M12_MAL_50_RAT', 'M12_MAL_60_RAT',
            'M12_FME_1020_RAT', 'M12_FME_30_RAT', 'M12_FME_40_RAT', 'M12_FME_50_RAT', 'M12_FME_60_RAT'
        ]

        for col in demographic_cols:
            if col in self.merged_data.columns:
                self.merged_data[col] = pd.to_numeric(self.merged_data[col], errors='coerce')

        # 3-2. 재구매/신규 고객 비율
        if 'MCT_UE_CLN_REU_RAT' in self.merged_data.columns:
            self.merged_data['MCT_UE_CLN_REU_RAT'] = pd.to_numeric(
                self.merged_data['MCT_UE_CLN_REU_RAT'], errors='coerce'
            )

        if 'MCT_UE_CLN_NEW_RAT' in self.merged_data.columns:
            self.merged_data['MCT_UE_CLN_NEW_RAT'] = pd.to_numeric(
                self.merged_data['MCT_UE_CLN_NEW_RAT'], errors='coerce'
            )

        # 3-3. 거주/직장/유입 고객 비율 컬럼 제거 (사용하지 않음)
        # RC_M1_SHC_RSD_UE_CLN_RAT, RC_M1_SHC_WP_UE_CLN_RAT, RC_M1_SHC_FLP_UE_CLN_RAT
        unused_location_cols = ['RC_M1_SHC_RSD_UE_CLN_RAT', 'RC_M1_SHC_WP_UE_CLN_RAT', 'RC_M1_SHC_FLP_UE_CLN_RAT']
        for col in unused_location_cols:
            if col in self.merged_data.columns:
                self.merged_data = self.merged_data.drop(columns=[col])

        print(f"  ✓ 사용하지 않는 컬럼 제거: {unused_location_cols}")

        # === 4. 세대별(연령대별) 매출 특성 생성 ===
        print("  📊 세대별 매출 분석 중...")

        # 4-1. 세대별 매출 집중도
        male_cols = [c for c in demographic_cols if 'MAL' in c]
        female_cols = [c for c in demographic_cols if 'FME' in c]

        # 남성 고객 비중
        if male_cols:
            self.merged_data['남성고객_비중'] = self.merged_data[male_cols].sum(axis=1)

        # 여성 고객 비중
        if female_cols:
            self.merged_data['여성고객_비중'] = self.merged_data[female_cols].sum(axis=1)

        # 연령대별 집중도
        if 'M12_MAL_1020_RAT' in self.merged_data.columns and 'M12_FME_1020_RAT' in self.merged_data.columns:
            self.merged_data['2030세대_비중'] = (
                self.merged_data['M12_MAL_1020_RAT'] + self.merged_data['M12_MAL_30_RAT'] +
                self.merged_data['M12_FME_1020_RAT'] + self.merged_data['M12_FME_30_RAT']
            )

            self.merged_data['4050세대_비중'] = (
                self.merged_data['M12_MAL_40_RAT'] + self.merged_data['M12_MAL_50_RAT'] +
                self.merged_data['M12_FME_40_RAT'] + self.merged_data['M12_FME_50_RAT']
            )

            self.merged_data['60대이상_비중'] = (
                self.merged_data['M12_MAL_60_RAT'] + self.merged_data['M12_FME_60_RAT']
            )

        # 4-2. 세대별 변화율 (1개월, 3개월)
        generation_cols = ['2030세대_비중', '4050세대_비중', '60대이상_비중']

        for col in generation_cols:
            if col in self.merged_data.columns:
                # 1개월 변화
                self.merged_data[f'{col}_변화_1M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].diff(1)

                # 3개월 변화
                self.merged_data[f'{col}_변화_3M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].diff(3)

        # 4-3. 주력 세대 식별
        if all(col in self.merged_data.columns for col in generation_cols):
            def get_main_generation(row):
                # 모든 세대 비중이 NaN이면 None 반환 (나중에 처리)
                if pd.isna(row['2030세대_비중']) and pd.isna(row['4050세대_비중']) and pd.isna(row['60대이상_비중']):
                    return None

                # NaN을 0으로 처리하여 비교
                gen_2030 = row['2030세대_비중'] if pd.notna(row['2030세대_비중']) else 0
                gen_4050 = row['4050세대_비중'] if pd.notna(row['4050세대_비중']) else 0
                gen_60 = row['60대이상_비중'] if pd.notna(row['60대이상_비중']) else 0

                # 모두 0이면 None
                if gen_2030 == 0 and gen_4050 == 0 and gen_60 == 0:
                    return None

                # 최대값을 가진 세대 반환
                max_val = max(gen_2030, gen_4050, gen_60)
                if gen_2030 == max_val:
                    return '2030'
                elif gen_4050 == max_val:
                    return '4050'
                else:
                    return '60+'

            self.merged_data['주력세대'] = self.merged_data.apply(get_main_generation, axis=1)

            # None 값 통계 출력
            null_count = self.merged_data['주력세대'].isna().sum()
            if null_count > 0:
                print(f"  ⚠️ 주력세대 정보 없음: {null_count:,}건 (전체의 {null_count/len(self.merged_data)*100:.1f}%)")
                print(f"     → 세대별 고객 데이터가 없는 가맹점입니다.")

        # 4-4. 고객 다양성 지수
        if demographic_cols:
            demographic_data = self.merged_data[demographic_cols].fillna(0)

            def calculate_entropy(row):
                probs = row / (row.sum() + 1e-10)
                probs = probs[probs > 0]
                return -np.sum(probs * np.log2(probs + 1e-10))

            self.merged_data['고객다양성지수'] = demographic_data.apply(calculate_entropy, axis=1)

        # === 5. 시계열 특성 (매출, 이용건수, 고객 수) ===
        print("  📈 시계열 특성 생성 중...")

        key_metrics = ['RC_M1_SAA_num', 'RC_M1_TO_UE_CT_num', 'RC_M1_UE_CUS_CN_num', 'RC_M1_AV_NP_AT_num']

        for col in key_metrics:
            if col in self.merged_data.columns:
                # 변화율 (1M, 3M, 6M)
                self.merged_data[f'{col}_변화율_1M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].pct_change(1) * 100

                self.merged_data[f'{col}_변화율_3M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].pct_change(3) * 100

                self.merged_data[f'{col}_변화율_6M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].pct_change(6) * 100

                # 이동평균 (3M, 6M)
                self.merged_data[f'{col}_MA3'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].transform(
                        lambda x: x.rolling(window=3, min_periods=1).mean()
                    )

                self.merged_data[f'{col}_MA6'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].transform(
                        lambda x: x.rolling(window=6, min_periods=1).mean()
                    )

                # 추세 (변화율의 이동평균)
                change_col = f'{col}_변화율_1M'
                if change_col in self.merged_data.columns:
                    self.merged_data[f'{col}_추세3M'] = \
                        self.merged_data.groupby('ENCODED_MCT')[change_col].transform(
                            lambda x: x.rolling(window=3, min_periods=1).mean()
                        )

                # 변동성 (표준편차)
                self.merged_data[f'{col}_STD3M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].transform(
                        lambda x: x.rolling(window=3, min_periods=1).std()
                    )

                # 연속 하락 개월 수
                self.merged_data[f'{col}_하락여부'] = (
                    self.merged_data[f'{col}_변화율_1M'] < -5
                ).astype(int)

                def count_consecutive(group):
                    result = []
                    count = 0
                    for val in group:
                        if val == 1:
                            count += 1
                        else:
                            count = 0
                        result.append(count)
                    return pd.Series(result, index=group.index)

                self.merged_data[f'{col}_연속하락'] = \
                    self.merged_data.groupby('ENCODED_MCT')[f'{col}_하락여부'].apply(
                        count_consecutive
                    ).values

        # === 6. 임대료 특성 (rental) ===
        print("  🏢 임대료 특성 생성 중...")

        if '전체(단위:원/평)' in self.merged_data.columns:
            # 임대료 변화율
            self.merged_data['임대료_변화_3M'] = \
                self.merged_data.groupby('ENCODED_MCT')['전체(단위:원/평)'].pct_change(3) * 100

            # 임대료 대비 매출 효율
            if 'RC_M1_SAA_num' in self.merged_data.columns:
                self.merged_data['임대료대비매출효율'] = \
                    self.merged_data['RC_M1_SAA_num'] / (
                        self.merged_data['전체(단위:원/평)'] / 10000 + 1
                    )

        # === 7. 유동인구 특성 (flow) ===
        print("  👥 유동인구 특성 생성 중...")

        if '유동인구(단위:명/1ha)' in self.merged_data.columns:
            # 유동인구 변화율
            self.merged_data['유동인구_변화_3M'] = \
                self.merged_data.groupby('ENCODED_MCT')['유동인구(단위:명/1ha)'].pct_change(3) * 100

            # 유동인구 대비 매출
            if 'RC_M1_SAA_num' in self.merged_data.columns:
                self.merged_data['유동인구대비매출'] = \
                    self.merged_data['RC_M1_SAA_num'] / (
                        self.merged_data['유동인구(단위:명/1ha)'] / 1000 + 1
                    )

        # 주거/직장 인구 비율
        if '주거인구(단위:명/1ha)' in self.merged_data.columns and '직장인구(단위:명/1ha)' in self.merged_data.columns:
            total_pop = (
                self.merged_data['주거인구(단위:명/1ha)'] +
                self.merged_data['직장인구(단위:명/1ha)'] + 1
            )
            self.merged_data['주거인구_비율'] = self.merged_data['주거인구(단위:명/1ha)'] / total_pop * 100
            self.merged_data['직장인구_비율'] = self.merged_data['직장인구(단위:명/1ha)'] / total_pop * 100

        # === 8. 가맹점 정보 (set1) 특성 ===
        print("  🏪 가맹점 정보 특성 생성 중...")

        # 개업 경과 일수
        if 'ARE_D' in self.merged_data.columns:
            self.merged_data['ARE_D'] = pd.to_datetime(
                self.merged_data['ARE_D'], format='%Y%m%d', errors='coerce'
            )
            self.merged_data['개업경과일'] = (
                self.merged_data['TA_YM'] - self.merged_data['ARE_D']
            ).dt.days

        print(f"✓ 특성 생성 완료: {len(self.merged_data.columns)}개 컬럼\n")

    def create_target_variable(self):
        """타겟 변수 생성 (미래 위험 예측)"""
        print("🎯 타겟 변수 생성 중...")

        # 미래 3개월 후 매출 하락 여부로 레이블 생성
        self.merged_data = self.merged_data.sort_values(['ENCODED_MCT', 'TA_YM'])

        # 3개월 후 매출 변화율
        self.merged_data['미래_3M_매출변화'] = \
            self.merged_data.groupby('ENCODED_MCT')['RC_M1_SAA_num'].shift(-3)

        self.merged_data['미래_3M_매출변화율'] = (
            (self.merged_data['미래_3M_매출변화'] - self.merged_data['RC_M1_SAA_num']) /
            (self.merged_data['RC_M1_SAA_num'] + 1)
        ) * 100

        # 타겟 레이블 정의 (4개 클래스 균형있게)
        conditions = [
            self.merged_data['미래_3M_매출변화율'] > 10,                          # 0: 안전 (성장)
            self.merged_data['미래_3M_매출변화율'].between(-10, 10),            # 1: 주의 (안정)
            self.merged_data['미래_3M_매출변화율'].between(-30, -10, inclusive='left'),  # 2: 경고 (하락)
            self.merged_data['미래_3M_매출변화율'] <= -30                        # 3: 위험 (급락)
        ]

        self.merged_data['경보레벨'] = np.select(
            conditions,
            [0, 1, 2, 3],
            default=1  # 기본값은 주의
        )

        # 추가 위험 요인 반영
        # 연속 하락이 있으면 레벨 상향 조정
        if 'RC_M1_SAA_num_연속하락' in self.merged_data.columns:
            # 연속 3개월 이상 하락 + 현재 레벨이 주의 이하면 -> 경고로
            self.merged_data.loc[
                (self.merged_data['RC_M1_SAA_num_연속하락'] >= 3) &
                (self.merged_data['경보레벨'] <= 1),
                '경보레벨'
            ] = 2

            # 연속 5개월 이상 하락 + 현재 레벨이 경고 이하면 -> 위험으로
            self.merged_data.loc[
                (self.merged_data['RC_M1_SAA_num_연속하락'] >= 5) &
                (self.merged_data['경보레벨'] <= 2),
                '경보레벨'
            ] = 3

        # 고객 수 급감도 위험 요인
        if 'RC_M1_UE_CUS_CN_num_변화율_3M' in self.merged_data.columns:
            self.merged_data.loc[
                (self.merged_data['RC_M1_UE_CUS_CN_num_변화율_3M'] < -40) &
                (self.merged_data['경보레벨'] <= 2),
                '경보레벨'
            ] = 3

        # 미래 데이터가 없는 행 제거 (마지막 3개월)
        self.merged_data = self.merged_data[self.merged_data['미래_3M_매출변화율'].notna()]

        print(f"✓ 타겟 변수 생성 완료")
        print(f"  레이블 분포:")
        for level in range(4):
            count = (self.merged_data['경보레벨'] == level).sum()
            pct = count / len(self.merged_data) * 100
            print(f"    {self.WARNING_LEVELS[level]['emoji']} {self.WARNING_LEVELS[level]['name']}: {count:,}건 ({pct:.1f}%)")
        print()

    def prepare_ml_features(self):
        """머신러닝용 특성 준비"""
        print("🔨 ML 특성 준비 중...")

        # 제외할 컬럼
        exclude_cols = [
            'ENCODED_MCT', 'TA_YM', 'MCT_BSE_AR', 'MCT_NM', 'MCT_BRD_NUM', 'ARE_D',
            '__filled_flag__', '__fill_method__', '__impute_source__',
            '경보레벨', 'LEGAL_DONG', '행정구역', '행적구역', 'MCT_SIGUNGU_NM',
            '기간', '미래_3M_매출변화', '미래_3M_매출변화율',
            # 원본 범주형 컬럼 (인코딩 전)
            'MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN',
            'RC_M1_AV_NP_AT', 'APV_CE_RAT',
            # 사용하지 않는 거주/직장/유입 컬럼 (이미 삭제되었지만 명시)
            'RC_M1_SHC_RSD_UE_CLN_RAT', 'RC_M1_SHC_WP_UE_CLN_RAT', 'RC_M1_SHC_FLP_UE_CLN_RAT'
        ]

        # 범주형 변수 인코딩
        categorical_cols = ['HPSN_MCT_ZCD_NM', 'HPSN_MCT_BZN_CD_NM', '주력세대', '월', '분기', '연도']

        for col in categorical_cols:
            if col in self.merged_data.columns:
                le = LabelEncoder()
                self.merged_data[f'{col}_encoded'] = le.fit_transform(
                    self.merged_data[col].fillna('Unknown').astype(str)
                )
                self.label_encoders[col] = le
                exclude_cols.append(col)  # 원본 제외

        # 특성 선택
        all_cols = set(self.merged_data.columns)
        exclude_set = set(exclude_cols)
        potential_features = all_cols - exclude_set

        self.feature_cols = []
        for col in potential_features:
            if self.merged_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # 결측치 비율 확인
                missing_rate = self.merged_data[col].isna().sum() / len(self.merged_data)
                if missing_rate < 0.9:  # 90% 이상 결측 제외
                    self.feature_cols.append(col)

        print(f"✓ 선택된 특성: {len(self.feature_cols)}개")

        # 특성 그룹별 개수
        feature_groups = {
            '매출/운영': [f for f in self.feature_cols if any(x in f for x in ['SAA', 'TO_UE', 'CUS', 'OPE', 'AV_NP'])],
            '세대별': [f for f in self.feature_cols if any(x in f for x in ['MAL', 'FME', '세대', '고객다양성'])],
            '시계열': [f for f in self.feature_cols if any(x in f for x in ['변화율', 'MA', '추세', 'STD', '연속'])],
            '임대료': [f for f in self.feature_cols if '임대료' in f],
            '유동인구': [f for f in self.feature_cols if any(x in f for x in ['유동인구', '주거인구', '직장인구'])],
            '기타': []
        }

        assigned = set()
        for group in feature_groups.values():
            assigned.update(group)

        feature_groups['기타'] = [f for f in self.feature_cols if f not in assigned]

        print(f"\n  특성 그룹별 분포:")
        for group_name, features in feature_groups.items():
            if features:
                print(f"    {group_name}: {len(features)}개")
        print()

    def train_model(self):
        """LightGBM 모델 학습"""
        print("=" * 80)
        print("🚀 LightGBM 모델 학습")
        print("=" * 80)

        # 학습 데이터 준비 (운영 3개월 이상)
        train_data = self.merged_data[self.merged_data['운영개월수'] >= 3].copy()

        X = train_data[self.feature_cols].fillna(0)
        y = train_data['경보레벨']

        # 실제 존재하는 클래스 확인
        unique_classes = sorted(y.unique())
        num_classes = len(unique_classes)

        print(f"✓ 실제 클래스 수: {num_classes}개 ({unique_classes})")

        # 시계열 분할 (최근 20%를 테스트)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"✓ 학습 데이터: {len(X_train):,}건")
        print(f"✓ 테스트 데이터: {len(X_test):,}건")

        # 학습/테스트 데이터의 클래스 분포
        print(f"\n  학습 데이터 클래스 분포:")
        for cls in unique_classes:
            count = (y_train == cls).sum()
            print(f"    클래스 {cls}: {count:,}건 ({count/len(y_train)*100:.1f}%)")

        print(f"\n  테스트 데이터 클래스 분포:")
        for cls in unique_classes:
            count = (y_test == cls).sum()
            print(f"    클래스 {cls}: {count:,}건 ({count/len(y_test)*100:.1f}%)")
        print()

        # LightGBM 데이터셋
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # 파라미터
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,  # 실제 클래스 수로 설정
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 40,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 8,
            'min_child_samples': 30,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }

        # 학습
        print("🔄 모델 학습 시작...")
        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_train, lgb_eval],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )

        print("\n✓ 학습 완료!\n")

        # 평가
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred_class = np.argmax(y_pred, axis=1)

        print("=" * 80)
        print("📊 모델 성능 평가")
        print("=" * 80)
        print("\n[분류 리포트]")

        # 실제 존재하는 클래스에 대한 레이블만 사용
        target_names = [f"{self.WARNING_LEVELS[i]['emoji']} {self.WARNING_LEVELS[i]['name']}"
                        for i in unique_classes]

        print(classification_report(
            y_test, y_pred_class,
            labels=unique_classes,
            target_names=target_names,
            zero_division=0
        ))

        # 전체 데이터 예측
        print("🔮 전체 데이터 예측 중...")
        X_all = self.merged_data[self.feature_cols].fillna(0)
        y_pred_all = self.model.predict(X_all, num_iteration=self.model.best_iteration)

        self.merged_data['예측_경보레벨'] = np.argmax(y_pred_all, axis=1)

        # 각 클래스별 확률 저장
        for i in range(num_classes):
            self.merged_data[f'예측_확률_{i}'] = y_pred_all[:, i]

        # 4개 클래스가 아닌 경우 나머지 확률 컬럼도 0으로 채우기
        for i in range(num_classes, 4):
            self.merged_data[f'예측_확률_{i}'] = 0.0

        # 예측 위험점수 (확률 가중 평균)
        # 각 클래스별 확률에 가중치를 곱해서 합산
        risk_score = self.merged_data['예측_확률_0'] * 0

        if '예측_확률_1' in self.merged_data.columns:
            risk_score += self.merged_data['예측_확률_1'] * 33
        if '예측_확률_2' in self.merged_data.columns:
            risk_score += self.merged_data['예측_확률_2'] * 66
        if '예측_확률_3' in self.merged_data.columns:
            risk_score += self.merged_data['예측_확률_3'] * 100

        self.merged_data['예측_위험점수'] = risk_score

        print("✓ 예측 완료!\n")

        return X_test, y_test, y_pred

    def visualize_feature_importance(self, save_path='ml_feature_importance.png', top_n=40):
        """특성 중요도 시각화"""
        print("📊 특성 중요도 시각화 중...")

        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        fig.suptitle('LightGBM 특성 중요도 분석 (Gain 기준)', fontsize=18, fontweight='bold')

        importance = self.model.feature_importance(importance_type='gain')
        indices = np.argsort(importance)[::-1][:top_n]

        # 특성 그룹별 색상
        colors = []
        for idx in indices:
            feat_name = self.feature_cols[idx]
            if any(x in feat_name for x in ['세대', 'MAL', 'FME', '고객다양성']):
                colors.append('#FF6B6B')  # 빨강 - 세대별
            elif any(x in feat_name for x in ['변화율', 'MA', '추세', 'STD', '연속']):
                colors.append('#4ECDC4')  # 청록 - 시계열
            elif '임대료' in feat_name:
                colors.append('#FFE66D')  # 노랑 - 임대료
            elif any(x in feat_name for x in ['유동인구', '주거인구', '직장인구']):
                colors.append('#95E1D3')  # 민트 - 인구
            else:
                colors.append('#A8DADC')  # 회청 - 기타

        bars = ax.barh(range(top_n), importance[indices], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([self.feature_cols[i] for i in indices], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('중요도 (Gain)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} 특성 중요도', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # 범례
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', label='세대별 특성'),
            Patch(facecolor='#4ECDC4', label='시계열 특성'),
            Patch(facecolor='#FFE66D', label='임대료 특성'),
            Patch(facecolor='#95E1D3', label='유동인구 특성'),
            Patch(facecolor='#A8DADC', label='기타')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def visualize_generation_analysis(self, save_path='generation_analysis.png'):
        """세대별 매출 분석 시각화"""
        print("📊 세대별 분석 시각화 중...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('세대별 매출 변화 분석 (2023-01 ~ 2024-12)', fontsize=18, fontweight='bold', y=0.995)

        # 1. 월별 세대 비중 추이
        ax1 = axes[0, 0]
        monthly_gen = self.merged_data.groupby('TA_YM')[
            ['2030세대_비중', '4050세대_비중', '60대이상_비중']
        ].mean()

        ax1.plot(monthly_gen.index, monthly_gen['2030세대_비중'],
                marker='o', linewidth=2.5, label='2030세대', color='#FF6B6B')
        ax1.plot(monthly_gen.index, monthly_gen['4050세대_비중'],
                marker='s', linewidth=2.5, label='4050세대', color='#4ECDC4')
        ax1.plot(monthly_gen.index, monthly_gen['60대이상_비중'],
                marker='^', linewidth=2.5, label='60대 이상', color='#95E1D3')

        ax1.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax1.set_ylabel('평균 비중 (%)', fontsize=11, fontweight='bold')
        ax1.set_title('월별 세대 비중 추이', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)

        # 2. 주력세대별 가맹점 수
        ax2 = axes[0, 1]
        latest_data = self.merged_data[self.merged_data['TA_YM'] == self.merged_data['TA_YM'].max()]

        if '주력세대' in latest_data.columns:
            # NaN 값을 '정보없음'으로 표시
            gen_counts = latest_data['주력세대'].fillna('정보없음').value_counts()

            # 원하는 순서로 정렬: 2030, 4050, 정보없음, 60+
            order = ['2030', '4050', '60+', '정보없음']
            gen_counts = gen_counts.reindex([g for g in order if g in gen_counts.index])

            colors_gen = {
                '2030': '#FF6B6B',
                '4050': '#4ECDC4',
                '60+': '#95E1D3',
                '정보없음': '#DDDDDD'
            }
            colors_list = [colors_gen.get(g, '#CCCCCC') for g in gen_counts.index]

            bars = ax2.bar(range(len(gen_counts)), gen_counts.values,
                          color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
            ax2.set_xticks(range(len(gen_counts)))
            ax2.set_xticklabels(gen_counts.index, fontsize=11)
            ax2.set_ylabel('가맹점 수', fontsize=11, fontweight='bold')
            ax2.set_title('주력 세대별 가맹점 분포', fontsize=13, fontweight='bold', pad=10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')

            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 3. 세대별 평균 위험점수
        ax3 = axes[1, 0]
        if '주력세대' in latest_data.columns and '예측_위험점수' in latest_data.columns:
            # NaN 값을 '정보없음'으로 변환 후 그룹화
            latest_data_copy = latest_data.copy()
            latest_data_copy['주력세대'] = latest_data_copy['주력세대'].fillna('정보없음')
            gen_risk = latest_data_copy.groupby('주력세대')['예측_위험점수'].mean()

            # 원하는 순서로 정렬: 2030, 4050, 정보없음, 60+ (위에서 아래로)
            order = ['정보없음', '60+', '4050', '2030']
            gen_risk = gen_risk.reindex([g for g in order if g in gen_risk.index])

            bars = ax3.barh(range(len(gen_risk)), gen_risk.values,
                           color='coral', alpha=0.7, edgecolor='black')
            ax3.set_yticks(range(len(gen_risk)))
            ax3.set_yticklabels(gen_risk.index, fontsize=11)
            ax3.set_xlabel('평균 예측 위험점수', fontsize=11, fontweight='bold')
            ax3.set_title('주력 세대별 평균 위험점수', fontsize=13, fontweight='bold', pad=10)
            ax3.grid(axis='x', alpha=0.3, linestyle='--')

            # 위험 구간 표시
            ax3.axvline(x=33, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
            ax3.axvline(x=66, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax3.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{width:.1f}',
                        ha='left', va='center', fontsize=10, fontweight='bold')

        # 4. 세대 변화와 매출 변화의 관계
        ax4 = axes[1, 1]
        if '2030세대_비중_변화_3M' in self.merged_data.columns and 'RC_M1_SAA_num_변화율_3M' in self.merged_data.columns:
            sample_data = self.merged_data[
                self.merged_data['2030세대_비중_변화_3M'].notna() &
                self.merged_data['RC_M1_SAA_num_변화율_3M'].notna()
            ].sample(min(1000, len(self.merged_data)))

            scatter = ax4.scatter(
                sample_data['2030세대_비중_변화_3M'],
                sample_data['RC_M1_SAA_num_변화율_3M'],
                c=sample_data['예측_경보레벨'],
                cmap='RdYlGn_r',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.5
            )

            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            ax4.set_xlabel('2030세대 비중 변화 (3개월, %p)', fontsize=11, fontweight='bold')
            ax4.set_ylabel('매출 변화율 (3개월, %)', fontsize=11, fontweight='bold')
            ax4.set_title('2030세대 비중 변화 vs 매출 변화', fontsize=13, fontweight='bold', pad=10)
            ax4.grid(True, alpha=0.3, linestyle='--')

            cbar = plt.colorbar(scatter, ax=ax4)
            cbar.set_label('예측 경보 레벨', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def visualize_merchant_detail(self, merchant_id, save_path='merchant_detail.png'):
        """가맹점별 상세 분석 시각화 (매출/세대/위험성 추이)"""
        print(f"🔍 가맹점 상세 분석: {merchant_id}")

        merchant_data = self.merged_data[self.merged_data['ENCODED_MCT'] == merchant_id].copy()
        merchant_data = merchant_data.sort_values('TA_YM')

        if len(merchant_data) == 0:
            print(f"❌ 해당 가맹점 데이터 없음: {merchant_id}")
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 가맹점 정보
        latest = merchant_data.iloc[-1]
        mct_name = latest.get('MCT_NM', 'N/A')
        mct_type = latest.get('HPSN_MCT_BZN_CD_NM', 'N/A')

        fig.suptitle(f'가맹점 상세 분석\n{mct_name} ({mct_type})',
                     fontsize=18, fontweight='bold', y=0.98)

        # 1. 매출 변화 추이
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()

        # 매출 수준 (왼쪽 축)
        l1 = ax1.plot(merchant_data['TA_YM'], merchant_data['RC_M1_SAA_num'],
                     marker='o', linewidth=2.5, markersize=6, color='#2E86AB', label='매출 수준')
        ax1.fill_between(merchant_data['TA_YM'], merchant_data['RC_M1_SAA_num'],
                         alpha=0.2, color='#2E86AB')

        # 매출 변화율 (오른쪽 축)
        l2 = ax1_twin.plot(merchant_data['TA_YM'], merchant_data['RC_M1_SAA_num_변화율_3M'],
                          marker='s', linewidth=2, markersize=5, color='#E63946',
                          label='3개월 변화율', linestyle='--')
        ax1_twin.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

        ax1.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax1.set_ylabel('매출 수준 (백분위)', fontsize=11, fontweight='bold', color='#2E86AB')
        ax1_twin.set_ylabel('변화율 (%)', fontsize=11, fontweight='bold', color='#E63946')
        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1_twin.tick_params(axis='y', labelcolor='#E63946')
        ax1.tick_params(axis='x', rotation=45)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)
        ax1.set_title('매출 변화 추이', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. 남성 세대별 비중 변화 추이
        ax2 = axes[0, 1]
        male_age_cols = ['M12_MAL_1020_RAT', 'M12_MAL_30_RAT', 'M12_MAL_40_RAT',
                        'M12_MAL_50_RAT', 'M12_MAL_60_RAT']
        male_age_labels = ['👨 10-20대', '👨 30대', '👨 40대', '👨 50대', '👨 60대+']
        male_colors = ['#FF6B6B', '#FF8E53', '#FFA94D', '#FFD93D', '#6BCB77']

        for col, label, color in zip(male_age_cols, male_age_labels, male_colors):
            if col in merchant_data.columns:
                data = merchant_data[col].fillna(0)
                ax2.plot(merchant_data['TA_YM'], data,
                        marker='o', linewidth=2, label=label, color=color, alpha=0.8)

        ax2.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax2.set_ylabel('남성 고객 비중 (%)', fontsize=11, fontweight='bold')
        ax2.set_title('👨 남성 고객 - 연령대별 추이', fontsize=13, fontweight='bold', pad=10)
        ax2.legend(fontsize=9, loc='best', ncol=2)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, max(100, merchant_data[male_age_cols].max().max() * 1.1) if any(col in merchant_data.columns for col in male_age_cols) else 100)

        # 3. 여성 세대별 비중 변화 추이
        ax3 = axes[0, 2]
        female_age_cols = ['M12_FME_1020_RAT', 'M12_FME_30_RAT', 'M12_FME_40_RAT',
                          'M12_FME_50_RAT', 'M12_FME_60_RAT']
        female_age_labels = ['👩 10-20대', '👩 30대', '👩 40대', '👩 50대', '👩 60대+']
        female_colors = ['#E84393', '#D63031', '#FD79A8', '#FDCB6E', '#00B894']

        for col, label, color in zip(female_age_cols, female_age_labels, female_colors):
            if col in merchant_data.columns:
                data = merchant_data[col].fillna(0)
                ax3.plot(merchant_data['TA_YM'], data,
                        marker='s', linewidth=2, label=label, color=color, alpha=0.8)

        ax3.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax3.set_ylabel('여성 고객 비중 (%)', fontsize=11, fontweight='bold')
        ax3.set_title('👩 여성 고객 - 연령대별 추이', fontsize=13, fontweight='bold', pad=10)
        ax3.legend(fontsize=9, loc='best', ncol=2)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, max(100, merchant_data[female_age_cols].max().max() * 1.1) if any(col in merchant_data.columns for col in female_age_cols) else 100)

        # 4. 위험성 추이 (위험점수 + 경보레벨)
        ax4 = axes[1, 0]

        # 위험점수 라인
        ax4.plot(merchant_data['TA_YM'], merchant_data['예측_위험점수'],
                marker='o', linewidth=3, markersize=7, color='#E63946', label='예측 위험점수')
        ax4.fill_between(merchant_data['TA_YM'], merchant_data['예측_위험점수'],
                         alpha=0.2, color='#E63946')

        # 위험 구간 표시
        ax4.axhline(y=33, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='주의')
        ax4.axhline(y=66, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='경고')

        ax4.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax4.set_ylabel('위험점수', fontsize=11, fontweight='bold')
        ax4.set_title('위험성 추이', fontsize=13, fontweight='bold', pad=10)
        ax4.legend(fontsize=10, loc='best')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 100)

        # 5. 성별 고객 비중 비교
        ax5 = axes[1, 1]

        # 최근 3개월 평균 계산
        recent_3m = merchant_data.tail(3)
        male_total = recent_3m[male_age_cols].sum(axis=1).mean() if any(col in recent_3m.columns for col in male_age_cols) else 0
        female_total = recent_3m[female_age_cols].sum(axis=1).mean() if any(col in recent_3m.columns for col in female_age_cols) else 0

        gender_data = [male_total, female_total]
        gender_labels = ['👨 남성', '👩 여성']
        gender_colors = ['#3498db', '#e74c3c']

        bars = ax5.bar(gender_labels, gender_data, color=gender_colors, alpha=0.7, edgecolor='black', linewidth=2)

        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax5.set_ylabel('고객 비중 (%)', fontsize=11, fontweight='bold')
        ax5.set_title('성별 고객 비중 비교 (최근 3개월 평균)', fontsize=13, fontweight='bold', pad=10)
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax5.set_ylim(0, max(gender_data) * 1.2 if max(gender_data) > 0 else 100)

        # 6. 현재 상태 요약
        ax6 = axes[1, 2]
        ax6.axis('off')

        pred_level = int(latest['예측_경보레벨'])
        pred_info = self.WARNING_LEVELS[pred_level]

        # 주력 세대 정보
        main_gen = latest.get('주력세대', 'N/A')
        if pd.isna(main_gen):
            main_gen = '정보없음'

        summary = f"""
        【 현재 경보 상태 】

        {pred_info['emoji']} 예측 레벨: {pred_info['name']}
        📊 위험점수: {latest['예측_위험점수']:.1f}점

        【 최근 지표 (3개월 평균) 】

        📈 매출 변화율: {latest.get('RC_M1_SAA_num_추세3M', 0):.1f}%
        👥 고객 변화율: {latest.get('RC_M1_UE_CUS_CN_num_추세3M', 0):.1f}%
        🔄 연속 하락: {latest.get('RC_M1_SAA_num_연속하락', 0):.0f}개월

        【 성별 고객 정보 】

        👨 남성 고객: {male_total:.1f}%
        👩 여성 고객: {female_total:.1f}%
        🎯 주력 세대: {main_gen}

        【 운영 정보 】

        📅 운영 기간: {latest.get('운영개월수', 0):.0f}개월
        🏢 업종: {mct_type}
        📍 지역: {latest.get('MCT_SIGUNGU_NM', 'N/A')}

        【 AI 분석 】
        """

        # LLM 분석 사용
        if self.use_llm:
            llm_analysis = self.analyze_risk_with_llm(latest)
            summary += f"\n{llm_analysis}"
        else:
            # 기본 분석
            if pred_level == 3:
                summary += "\n🔴 즉각 대응 필요\n    - 경영 전략 전면 재검토\n    - 전문 컨설팅 필수"
            elif pred_level == 2:
                summary += "\n🟠 개선 방안 수립\n    - 타겟 세대 마케팅 강화\n    - 비용 구조 최적화"
            elif pred_level == 1:
                summary += "\n🟡 예방적 관리\n    - 현 상태 모니터링\n    - 성장 기회 탐색"
            else:
                summary += "\n🟢 안정적 운영\n    - 현재 전략 유지\n    - 추가 성장 도모"

        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=pred_info['color'], alpha=0.15),
                family='malgun gothic')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def visualize_gender_generation_analysis(self, save_path='gender_generation_analysis.png'):
        """성별 구분 세대별 분석 시각화"""
        print("📊 성별 구분 세대별 분석 시각화 중...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('성별 구분 세대별 매출 변화 분석 (2023-01 ~ 2024-12)',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. 남성 세대별 월별 추이
        ax1 = axes[0, 0]
        male_cols = ['M12_MAL_1020_RAT', 'M12_MAL_30_RAT', 'M12_MAL_40_RAT', 'M12_MAL_50_RAT', 'M12_MAL_60_RAT']

        if all(col in self.merged_data.columns for col in male_cols):
            monthly_male = self.merged_data.groupby('TA_YM')[male_cols].mean()

            ax1.plot(monthly_male.index, monthly_male['M12_MAL_1020_RAT'],
                    marker='o', linewidth=2, label='10-20대', color='#4ECDC4')
            ax1.plot(monthly_male.index, monthly_male['M12_MAL_30_RAT'],
                    marker='s', linewidth=2, label='30대', color='#6DD5DB')
            ax1.plot(monthly_male.index, monthly_male['M12_MAL_40_RAT'],
                    marker='^', linewidth=2, label='40대', color='#8CDDE3')
            ax1.plot(monthly_male.index, monthly_male['M12_MAL_50_RAT'],
                    marker='D', linewidth=2, label='50대', color='#ABE5EB')
            ax1.plot(monthly_male.index, monthly_male['M12_MAL_60_RAT'],
                    marker='v', linewidth=2, label='60대+', color='#CAEDF3')

            ax1.set_xlabel('기간', fontsize=11, fontweight='bold')
            ax1.set_ylabel('평균 비중 (%)', fontsize=11, fontweight='bold')
            ax1.set_title('남성 고객 - 연령대별 추이', fontsize=13, fontweight='bold', pad=10)
            ax1.legend(fontsize=9, loc='best', ncol=2)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(axis='x', rotation=45)

        # 2. 여성 세대별 월별 추이
        ax2 = axes[0, 1]
        female_cols = ['M12_FME_1020_RAT', 'M12_FME_30_RAT', 'M12_FME_40_RAT', 'M12_FME_50_RAT', 'M12_FME_60_RAT']

        if all(col in self.merged_data.columns for col in female_cols):
            monthly_female = self.merged_data.groupby('TA_YM')[female_cols].mean()

            ax2.plot(monthly_female.index, monthly_female['M12_FME_1020_RAT'],
                    marker='o', linewidth=2, label='10-20대', color='#FF6B6B')
            ax2.plot(monthly_female.index, monthly_female['M12_FME_30_RAT'],
                    marker='s', linewidth=2, label='30대', color='#FF8787')
            ax2.plot(monthly_female.index, monthly_female['M12_FME_40_RAT'],
                    marker='^', linewidth=2, label='40대', color='#FFA5A5')
            ax2.plot(monthly_female.index, monthly_female['M12_FME_50_RAT'],
                    marker='D', linewidth=2, label='50대', color='#FFC3C3')
            ax2.plot(monthly_female.index, monthly_female['M12_FME_60_RAT'],
                    marker='v', linewidth=2, label='60대+', color='#FFE1E1')

            ax2.set_xlabel('기간', fontsize=11, fontweight='bold')
            ax2.set_ylabel('평균 비중 (%)', fontsize=11, fontweight='bold')
            ax2.set_title('여성 고객 - 연령대별 추이', fontsize=13, fontweight='bold', pad=10)
            ax2.legend(fontsize=9, loc='best', ncol=2)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.tick_params(axis='x', rotation=45)

        # 3. 성별 비교 (최신 월)
        ax3 = axes[0, 2]
        latest_data = self.merged_data[self.merged_data['TA_YM'] == self.merged_data['TA_YM'].max()]

        if '남성고객_비중' in latest_data.columns and '여성고객_비중' in latest_data.columns:
            male_avg = latest_data['남성고객_비중'].mean()
            female_avg = latest_data['여성고객_비중'].mean()

            bars = ax3.bar(['남성', '여성'], [male_avg, female_avg],
                          color=['#4ECDC4', '#FF6B6B'], alpha=0.7, edgecolor='black', linewidth=2)
            ax3.set_ylabel('평균 비중 (%)', fontsize=11, fontweight='bold')
            ax3.set_title('성별 고객 비중 비교', fontsize=13, fontweight='bold', pad=10)
            ax3.grid(axis='y', alpha=0.3, linestyle='--')

            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 4. 남성 연령대별 평균 위험점수
        ax4 = axes[1, 0]
        if all(col in latest_data.columns for col in male_cols):
            # 주력 남성 연령대 계산
            latest_data_copy = latest_data.copy()

            def get_main_male_age(row):
                ages = {
                    '10-20대': row.get('M12_MAL_1020_RAT', 0),
                    '30대': row.get('M12_MAL_30_RAT', 0),
                    '40대': row.get('M12_MAL_40_RAT', 0),
                    '50대': row.get('M12_MAL_50_RAT', 0),
                    '60대+': row.get('M12_MAL_60_RAT', 0)
                }
                if all(pd.isna(v) or v == 0 for v in ages.values()):
                    return None
                return max(ages, key=ages.get)

            latest_data_copy['주력_남성연령'] = latest_data_copy.apply(get_main_male_age, axis=1)
            latest_data_copy['주력_남성연령'] = latest_data_copy['주력_남성연령'].fillna('정보없음')

            male_risk = latest_data_copy.groupby('주력_남성연령')['예측_위험점수'].mean() # .sort_values(ascending=False)

            bars = ax4.barh(range(len(male_risk)), male_risk.values,
                           color='#4ECDC4', alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(male_risk)))
            ax4.set_yticklabels(male_risk.index, fontsize=10)
            ax4.set_xlabel('평균 위험점수', fontsize=11, fontweight='bold')
            ax4.set_title('남성 연령대별 평균 위험점수', fontsize=13, fontweight='bold', pad=10)
            ax4.grid(axis='x', alpha=0.3, linestyle='--')
            ax4.axvline(x=33, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
            ax4.axvline(x=66, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{width:.1f}',
                        ha='left', va='center', fontsize=9, fontweight='bold')

        # 5. 여성 연령대별 평균 위험점수
        ax5 = axes[1, 1]
        if all(col in latest_data.columns for col in female_cols):
            latest_data_copy = latest_data.copy()

            def get_main_female_age(row):
                ages = {
                    '10-20대': row.get('M12_FME_1020_RAT', 0),
                    '30대': row.get('M12_FME_30_RAT', 0),
                    '40대': row.get('M12_FME_40_RAT', 0),
                    '50대': row.get('M12_FME_50_RAT', 0),
                    '60대+': row.get('M12_FME_60_RAT', 0)
                }
                if all(pd.isna(v) or v == 0 for v in ages.values()):
                    return None
                return max(ages, key=ages.get)

            latest_data_copy['주력_여성연령'] = latest_data_copy.apply(get_main_female_age, axis=1)
            latest_data_copy['주력_여성연령'] = latest_data_copy['주력_여성연령'].fillna('정보없음')

            female_risk = latest_data_copy.groupby('주력_여성연령')['예측_위험점수'].mean() # .sort_values(ascending=False)

            bars = ax5.barh(range(len(female_risk)), female_risk.values,
                           color='#FF6B6B', alpha=0.7, edgecolor='black')
            ax5.set_yticks(range(len(female_risk)))
            ax5.set_yticklabels(female_risk.index, fontsize=10)
            ax5.set_xlabel('평균 위험점수', fontsize=11, fontweight='bold')
            ax5.set_title('여성 연령대별 평균 위험점수', fontsize=13, fontweight='bold', pad=10)
            ax5.grid(axis='x', alpha=0.3, linestyle='--')
            ax5.axvline(x=33, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
            ax5.axvline(x=66, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax5.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{width:.1f}',
                        ha='left', va='center', fontsize=9, fontweight='bold')

        # 6. 성별×세대 조합 히트맵
        ax6 = axes[1, 2]
        if all(col in self.merged_data.columns for col in male_cols + female_cols):
            # 평균 계산
            heatmap_data = []
            ages = ['10-20대', '30대', '40대', '50대', '60대+']

            male_avgs = [
                latest_data['M12_MAL_1020_RAT'].mean(),
                latest_data['M12_MAL_30_RAT'].mean(),
                latest_data['M12_MAL_40_RAT'].mean(),
                latest_data['M12_MAL_50_RAT'].mean(),
                latest_data['M12_MAL_60_RAT'].mean()
            ]

            female_avgs = [
                latest_data['M12_FME_1020_RAT'].mean(),
                latest_data['M12_FME_30_RAT'].mean(),
                latest_data['M12_FME_40_RAT'].mean(),
                latest_data['M12_FME_50_RAT'].mean(),
                latest_data['M12_FME_60_RAT'].mean()
            ]

            heatmap_df = pd.DataFrame({
                '남성': male_avgs,
                '여성': female_avgs
            }, index=ages)

            im = ax6.imshow(heatmap_df.T, cmap='YlOrRd', aspect='auto')

            ax6.set_xticks(range(len(ages)))
            ax6.set_xticklabels(ages, fontsize=10)
            ax6.set_yticks([0, 1])
            ax6.set_yticklabels(['남성', '여성'], fontsize=11)
            ax6.set_title('성별×연령대 고객 비중 히트맵', fontsize=13, fontweight='bold', pad=10)

            # 값 표시
            for i in range(2):
                for j in range(len(ages)):
                    text = ax6.text(j, i, f'{heatmap_df.iloc[j, i]:.1f}',
                                   ha="center", va="center", color="black", fontsize=10, fontweight='bold')

            plt.colorbar(im, ax=ax6, label='평균 비중 (%)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def visualize_time_period_analysis(self, save_path='time_period_analysis.png'):
        """시간 기간별 분석 시각화 (2023-01 ~ 2024-12)"""
        print("📊 기간별 분석 시각화 중...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('기간별 경영 지표 분석 (2023-01 ~ 2024-12)',
                     fontsize=18, fontweight='bold', y=0.995)

        # 월별 집계
        monthly_agg = self.merged_data.groupby('TA_YM').agg({
            '예측_위험점수': 'mean',
            'RC_M1_SAA_num': 'mean',
            'RC_M1_TO_UE_CT_num': 'mean',
            'ENCODED_MCT': 'nunique'
        }).reset_index()

        # 1. 월별 평균 위험점수 추이
        ax1 = axes[0, 0]
        ax1.plot(monthly_agg['TA_YM'], monthly_agg['예측_위험점수'],
                marker='o', linewidth=3, markersize=6, color='#E63946', label='평균 위험점수')
        ax1.fill_between(monthly_agg['TA_YM'], monthly_agg['예측_위험점수'],
                         alpha=0.3, color='#E63946')

        # 위험 구간 표시
        ax1.axhline(y=33, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='주의')
        ax1.axhline(y=66, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='경고')

        ax1.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax1.set_ylabel('평균 위험점수', fontsize=11, fontweight='bold')
        ax1.set_title('월별 평균 위험점수 추이', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)

        # 2. 월별 경보 레벨 분포
        ax2 = axes[0, 1]
        monthly_warnings = self.merged_data.groupby(['TA_YM', '예측_경보레벨']).size().unstack(fill_value=0)

        colors = ['green', 'yellow', 'orange', 'red']
        for level in range(4):
            if level in monthly_warnings.columns:
                ax2.plot(monthly_warnings.index, monthly_warnings[level],
                        marker='o', linewidth=2.5, markersize=5,
                        label=f"{self.WARNING_LEVELS[level]['emoji']} {self.WARNING_LEVELS[level]['name']}",
                        color=colors[level])

        ax2.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax2.set_ylabel('가맹점 수', fontsize=11, fontweight='bold')
        ax2.set_title('월별 경보 레벨별 가맹점 수', fontsize=13, fontweight='bold', pad=10)
        ax2.legend(fontsize=9, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)

        # 3. 연도별/분기별 비교
        ax3 = axes[1, 0]
        self.merged_data['연도_분기'] = (
            self.merged_data['연도'].astype(str) + '-Q' +
            self.merged_data['분기'].astype(str)
        )

        quarter_risk = self.merged_data.groupby('연도_분기')['예측_위험점수'].mean().sort_index()

        bars = ax3.bar(range(len(quarter_risk)), quarter_risk.values,
                      color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(quarter_risk)))
        ax3.set_xticklabels(quarter_risk.index, fontsize=10, rotation=45)
        ax3.set_ylabel('평균 위험점수', fontsize=11, fontweight='bold')
        ax3.set_title('분기별 평균 위험점수', fontsize=13, fontweight='bold', pad=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # 색상 구분
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height >= 66:
                bar.set_color('red')
            elif height >= 33:
                bar.set_color('orange')
            else:
                bar.set_color('green')
            bar.set_alpha(0.7)

            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 4. 활동 가맹점 수 추이
        ax4 = axes[1, 1]
        ax4.plot(monthly_agg['TA_YM'], monthly_agg['ENCODED_MCT'],
                marker='o', linewidth=3, markersize=6, color='#457B9D', label='활동 가맹점 수')
        ax4.fill_between(monthly_agg['TA_YM'], monthly_agg['ENCODED_MCT'],
                         alpha=0.3, color='#457B9D')

        ax4.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax4.set_ylabel('가맹점 수', fontsize=11, fontweight='bold')
        ax4.set_title('월별 활동 가맹점 수', fontsize=13, fontweight='bold', pad=10)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def visualize_confusion_matrix(self, y_test, y_pred, save_path='ml_confusion_matrix.png'):
        """혼동 행렬"""
        print("📊 혼동 행렬 시각화 중...")

        y_pred_class = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_test, y_pred_class)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=[f"{self.WARNING_LEVELS[i]['emoji']} {self.WARNING_LEVELS[i]['name']}"
                                for i in range(4)],
                    yticklabels=[f"{self.WARNING_LEVELS[i]['emoji']} {self.WARNING_LEVELS[i]['name']}"
                                for i in range(4)],
                    cbar_kws={'label': '건수'})

        ax.set_xlabel('예측 경보 레벨', fontsize=13, fontweight='bold')
        ax.set_ylabel('실제 경보 레벨', fontsize=13, fontweight='bold')
        ax.set_title('혼동 행렬 (Confusion Matrix)', fontsize=16, fontweight='bold', pad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def generate_report(self, output_file='ml_warning_report.csv'):
        """경보 리포트 생성"""
        print("📝 경보 리포트 생성 중...")

        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month].copy()

        report_cols = [
            'ENCODED_MCT', 'MCT_NM', 'HPSN_MCT_BZN_CD_NM', 'MCT_SIGUNGU_NM',
            '예측_경보레벨', '예측_위험점수',
            '예측_확률_0', '예측_확률_1', '예측_확률_2', '예측_확률_3',
            'RC_M1_SAA_num', 'RC_M1_SAA_num_변화율_3M', 'RC_M1_SAA_num_연속하락',
            '주력세대', '2030세대_비중', '4050세대_비중', '60대이상_비중',
            '운영개월수', '개업경과일'
        ]

        report_cols = [col for col in report_cols if col in latest_data.columns]
        report = latest_data[report_cols].copy()

        # 경보명 추가
        report['경보명'] = report['예측_경보레벨'].map(
            lambda x: self.WARNING_LEVELS[int(x)]['name']
        )

        report = report.sort_values('예측_위험점수', ascending=False)

        # LLM 분석 추가 (고위험 가맹점 상위 20개만)
        if self.use_llm:
            print("🤖 LLM 분석 진행 중 (고위험 상위 20개)...")
            llm_analyses = []

            for idx, row in report.head(20).iterrows():
                analysis = self.analyze_risk_with_llm(row)
                llm_analyses.append(analysis)
                print(f"  ✓ {row.get('MCT_NM', 'N/A')} 분석 완료")

            # 상위 20개에만 LLM 분석 결과 추가
            report.loc[report.head(20).index, 'AI_분석'] = llm_analyses

        report.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 리포트 저장: {output_file}")
        print(f"  총 {len(report):,}개 가맹점\n")

        return report

    def analyze_risk_with_llm(self, merchant_data_row, shap_values=None):
        """LLM을 사용한 위험 예측 분석"""
        if not self.use_llm or self.openai_client is None:
            return self._default_risk_analysis(merchant_data_row)

        try:
            # 가맹점 데이터 요약
            pred_level = int(merchant_data_row['예측_경보레벨'])
            risk_score = merchant_data_row['예측_위험점수']

            # LLM 프롬프트 구성
            prompt = f"""당신은 가맹점 경영 위기 분석 전문가입니다. 다음 가맹점의 위험 예측 결과를 분석하고 설명해주세요.

【 가맹점 정보 】
- 가맹점명: {merchant_data_row.get('MCT_NM', 'N/A')}
- 업종: {merchant_data_row.get('HPSN_MCT_BZN_CD_NM', 'N/A')}
- 지역: {merchant_data_row.get('MCT_SIGUNGU_NM', 'N/A')}
- 운영 기간: {merchant_data_row.get('운영개월수', 0):.0f}개월

【 예측 결과 】
- 예측 경보 레벨: {self.WARNING_LEVELS[pred_level]['name']} ({self.WARNING_LEVELS[pred_level]['emoji']})
- 위험 점수: {risk_score:.1f}/100점
- 안전 확률: {merchant_data_row.get('예측_확률_0', 0)*100:.1f}%
- 주의 확률: {merchant_data_row.get('예측_확률_1', 0)*100:.1f}%
- 경고 확률: {merchant_data_row.get('예측_확률_2', 0)*100:.1f}%
- 위험 확률: {merchant_data_row.get('예측_확률_3', 0)*100:.1f}%

【 주요 지표 】
- 매출 수준 (백분위): {merchant_data_row.get('RC_M1_SAA_num', 0):.1f}
- 3개월 매출 변화율: {merchant_data_row.get('RC_M1_SAA_num_변화율_3M', 0):.1f}%
- 연속 하락 개월: {merchant_data_row.get('RC_M1_SAA_num_연속하락', 0):.0f}개월
- 고객 수 변화율: {merchant_data_row.get('RC_M1_UE_CUS_CN_num_변화율_3M', 0):.1f}%

【 세대별 정보 】
- 주력 세대: {merchant_data_row.get('주력세대', '정보없음')}
- 2030세대 비중: {merchant_data_row.get('2030세대_비중', 0):.1f}%
- 4050세대 비중: {merchant_data_row.get('4050세대_비중', 0):.1f}%
- 60대 이상 비중: {merchant_data_row.get('60대이상_비중', 0):.1f}%

【 요청사항 】
다음 형식으로 간결하게 분석해주세요 (최대 150자):

1. 위험 요인 (1-2가지 핵심 요인)
2. 구체적 권장 조치 (실행 가능한 1-2가지)

**간결하고 실용적으로 작성해주세요.**"""

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "당신은 가맹점 경영 위기 분석 전문가입니다. 간결하고 실용적인 조언을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            content = response.choices[0].message.content
            return content.strip() if content else self._default_risk_analysis(merchant_data_row)

        except Exception as e:
            print(f"⚠️  LLM 분석 오류: {e}")
            return self._default_risk_analysis(merchant_data_row)

    def analyze_situation_with_llm(self, merchant_data_row):
        """LLM을 사용한 상황 분석"""
        if not self.use_llm or self.openai_client is None:
            return self._default_situation_analysis(merchant_data_row)

        try:
            pred_level = int(merchant_data_row['예측_경보레벨'])

            prompt = f"""당신은 가맹점 경영 컨설턴트입니다. 다음 가맹점의 현재 상황을 종합적으로 분석해주세요.

【 기본 정보 】
- 업종: {merchant_data_row.get('HPSN_MCT_BZN_CD_NM', 'N/A')}
- 지역: {merchant_data_row.get('MCT_SIGUNGU_NM', 'N/A')}
- 운영 기간: {merchant_data_row.get('운영개월수', 0):.0f}개월

【 경영 지표 】
- 현재 경보: {self.WARNING_LEVELS[pred_level]['name']}
- 매출 추세: {merchant_data_row.get('RC_M1_SAA_num_추세3M', 0):.1f}%
- 매출 변동성: {merchant_data_row.get('RC_M1_SAA_num_STD3M', 0):.1f}
- 고객 수 추세: {merchant_data_row.get('RC_M1_UE_CUS_CN_num_추세3M', 0):.1f}%

【 고객 구조 】
- 주력 세대: {merchant_data_row.get('주력세대', '정보없음')}
- 2030세대 변화: {merchant_data_row.get('2030세대_비중_변화_3M', 0):.1f}%p
- 4050세대 변화: {merchant_data_row.get('4050세대_비중_변화_3M', 0):.1f}%p
- 고객 다양성: {merchant_data_row.get('고객다양성지수', 0):.2f}

【 요청사항 】
다음을 포함하여 200자 이내로 분석해주세요:
1. 현재 상황 진단 (핵심 이슈 1-2개)
2. 근본 원인 추정
3. 즉시 실행 가능한 개선 방안 1가지"""

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "당신은 데이터 기반 경영 컨설턴트입니다. 명확하고 실행 가능한 조언을 제공합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )

            content = response.choices[0].message.content
            return content.strip() if content else self._default_situation_analysis(merchant_data_row)

        except Exception as e:
            print(f"⚠️  LLM 상황 분석 오류: {e}")
            return self._default_situation_analysis(merchant_data_row)

    def _default_risk_analysis(self, merchant_data_row):
        """기본 위험 분석 (LLM 없이)"""
        pred_level = int(merchant_data_row['예측_경보레벨'])

        if pred_level == 3:
            return "위험 요인: 매출 급락 및 연속 하락\n권장 조치: 즉각 경영 전략 재검토, 전문 컨설팅 필수"
        elif pred_level == 2:
            return "위험 요인: 매출 하락 및 고객 감소\n권장 조치: 타겟 마케팅 강화, 비용 구조 최적화"
        elif pred_level == 1:
            return "현재 상태: 안정적이나 모니터링 필요\n권장 조치: 예방적 관리, 성장 기회 탐색"
        else:
            return "현재 상태: 안정적 운영\n권장 조치: 현재 전략 유지, 추가 성장 도모"

    def _default_situation_analysis(self, merchant_data_row):
        """기본 상황 분석 (LLM 없이)"""
        pred_level = int(merchant_data_row['예측_경보레벨'])
        sales_change = merchant_data_row.get('RC_M1_SAA_num_변화율_3M', 0)
        main_gen = merchant_data_row.get('주력세대', '정보없음')

        if pred_level >= 2:
            return f"상황: {sales_change:.1f}% 매출 하락 중. 주력 세대({main_gen}) 이탈 가능성.\n개선안: 세대별 맞춤 마케팅 및 고객 재유치 캠페인 실행."
        else:
            return f"상황: 안정적 운영 중. 주력 세대: {main_gen}\n제안: 현 고객층 유지 및 신규 세대 확보 병행."

    def print_summary(self):
        """요약 통계"""
        print("=" * 80)
        print("📊 ML 경영 위기 조기 경보 시스템 - 요약")
        print("=" * 80)

        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month]

        print(f"\n📅 분석 기간: {self.merged_data['TA_YM'].min().strftime('%Y년 %m월')} ~ {latest_month.strftime('%Y년 %m월')}")
        print(f"🏢 분석 가맹점 수: {len(latest_data):,}개\n")

        print("【 예측 경보 레벨별 현황 】")
        for level in range(4):
            name = self.WARNING_LEVELS[level]['name']
            emoji = self.WARNING_LEVELS[level]['emoji']
            count = (latest_data['예측_경보레벨'] == level).sum()
            pct = count / len(latest_data) * 100
            print(f"  {emoji} {name:4s}: {count:6,}개 ({pct:5.1f}%)")

        print(f"\n【 예측 위험점수 통계 】")
        print(f"  평균: {latest_data['예측_위험점수'].mean():.1f}점")
        print(f"  중앙값: {latest_data['예측_위험점수'].median():.1f}점")
        print(f"  최대: {latest_data['예측_위험점수'].max():.1f}점")

        print("\n【 고위험 가맹점 (Top 10) 】")
        high_risk = latest_data.nlargest(10, '예측_위험점수')[
            ['MCT_NM', 'HPSN_MCT_BZN_CD_NM', '예측_위험점수', '예측_경보레벨']
        ]
        for idx, row in enumerate(high_risk.itertuples(), 1):
            level_name = self.WARNING_LEVELS[int(row.예측_경보레벨)]['name']
            mct_name = str(row.MCT_NM)[:15] if pd.notna(row.MCT_NM) else 'N/A'
            bzn_name = str(row.HPSN_MCT_BZN_CD_NM)[:15] if pd.notna(row.HPSN_MCT_BZN_CD_NM) else 'N/A'
            print(f"  {idx:2d}. {mct_name:15s} | {bzn_name:15s} | {row.예측_위험점수:5.1f}점 | {level_name}")

        # 세대별 통계
        if '주력세대' in latest_data.columns:
            print("\n【 주력 세대별 현황 】")
            # NaN 값을 '정보없음'으로 표시
            latest_data_copy = latest_data.copy()
            latest_data_copy['주력세대'] = latest_data_copy['주력세대'].fillna('정보없음')

            gen_stats = latest_data_copy.groupby('주력세대').agg({
                'ENCODED_MCT': 'count',
                '예측_위험점수': 'mean'
            }).round(1)
            gen_stats.columns = ['가맹점 수', '평균 위험점수']
            for gen, row in gen_stats.iterrows():
                gen_display = str(gen) if gen != '정보없음' else '정보없음'
                print(f"  {gen_display:8s}: {int(row['가맹점 수']):5,}개 (평균 위험점수: {row['평균 위험점수']:5.1f}점)")

        print("\n" + "=" * 80 + "\n")


def main():
    """메인 실행"""
    print("=" * 80)
    print("🚀 ML 기반 경영 위기 조기 경보 시스템")
    print("   (Pure LightGBM / 전체 데이터셋 활용 / 세대별 분석 포함)")
    print("=" * 80)
    print()

    # 시스템 초기화 (2023-01 ~ 2024-12)
    ews = MLEarlyWarningSystem(
        data_path='./data/',
        start_date='2023-01',
        end_date='2024-12'
    )

    # 1. 데이터 로드
    ews.load_data()

    # 2. 데이터 통합
    ews.merge_all_data()

    # 3. 종합 특성 생성
    ews.create_comprehensive_features()

    # 4. 타겟 변수 생성
    ews.create_target_variable()

    # 5. ML 특성 준비
    ews.prepare_ml_features()

    # 6. 모델 학습
    X_test, y_test, y_pred = ews.train_model()

    # 7. 요약
    ews.print_summary()

    # 8. 시각화
    print("📊 시각화 생성 중...\n")
    ews.visualize_feature_importance('ml_feature_importance.png', top_n=40)
    ews.visualize_confusion_matrix(y_test, y_pred, 'ml_confusion_matrix.png')
    ews.visualize_generation_analysis('generation_analysis.png')
    ews.visualize_gender_generation_analysis('gender_generation_analysis.png')
    ews.visualize_time_period_analysis('time_period_analysis.png')

    # 9. 리포트
    report = ews.generate_report('ml_warning_report.csv')

    # 10. 고위험 가맹점 상세 분석
    print("🔍 고위험 가맹점 상세 분석...\n")
    top_risk = report.nlargest(5, '예측_위험점수')['ENCODED_MCT'].values

    for idx, mct_id in enumerate(top_risk, 1):
        ews.visualize_merchant_detail(mct_id, f'merchant_detail_top{idx}.png')

    print("=" * 80)
    print("✅ ML 경영 위기 조기 경보 시스템 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  📊 ml_feature_importance.png - 특성 중요도 (색상별 그룹)")
    print("  📊 ml_confusion_matrix.png - 혼동 행렬")
    print("  📊 generation_analysis.png - 세대별 분석 (통합)")
    print("  📊 gender_generation_analysis.png - 성별 구분 세대별 분석")
    print("  📊 time_period_analysis.png - 기간별 분석 (2023-2024)")
    print("  🔍 merchant_detail_top1~5.png - 고위험 가맹점 상세 분석")
    print("  📝 ml_warning_report.csv - 경보 리포트")
    print()


if __name__ == "__main__":
    main()
