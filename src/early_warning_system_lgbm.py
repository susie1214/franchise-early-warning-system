# -*- coding: utf-8 -*-
"""
경영 위기 조기 경보 시스템 (LightGBM 버전)
- 4단계 경보 시스템: 안전(Green) -> 주의(Yellow) -> 경고(Orange) -> 위험(Red)
- LightGBM을 활용한 머신러닝 기반 위험도 예측
- 시계열 특성 엔지니어링 강화
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap

# 한글 폰트 설정
plt.rcParams['font.family'] = ['gulim', 'Symbol']
plt.rcParams['axes.unicode_minus'] = False

class LGBMEarlyWarningSystem:
    """LightGBM 기반 경영 위기 조기 경보 시스템"""

    def __init__(self, data_path='./data/'):
        """초기화"""
        self.data_path = data_path
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

    def load_data(self):
        """데이터 로드"""
        print("=" * 80)
        print("📊 데이터 로드 중...")
        print("=" * 80)

        self.merchant_data = pd.read_csv(f'{self.data_path}big_data_set1_f_v2.csv', encoding='utf-8-sig')
        print(f"✓ 가맹점 정보: {len(self.merchant_data):,}개")

        self.sales_data = pd.read_csv(f'{self.data_path}big_data_set2_f_sorted.csv', encoding='utf-8-sig')
        self.sales_data['TA_YM'] = pd.to_datetime(self.sales_data['TA_YM'], format='%Y%m')
        print(f"✓ 매출 데이터: {len(self.sales_data):,}건")

        self.customer_data = pd.read_csv(f'{self.data_path}big_data_set3_f_sorted.csv', encoding='utf-8-sig')
        self.customer_data['TA_YM'] = pd.to_datetime(self.customer_data['TA_YM'], format='%Y%m')
        print(f"✓ 고객 데이터: {len(self.customer_data):,}건")

        self.rental_data = pd.read_csv(f'{self.data_path}rental_p.csv', encoding='utf-8-sig')
        print(f"✓ 임대료 데이터: {len(self.rental_data):,}건")

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

        # 가맹점 정보
        self.merged_data = pd.merge(
            self.merged_data,
            self.merchant_data,
            on='ENCODED_MCT',
            how='left'
        )

        # 임대료 데이터 (분기 -> 월로 확장)
        self.rental_data['기간'] = pd.to_datetime(self.rental_data['기간(분기)'], format='%Y%m')
        rental_expanded = []
        for _, row in self.rental_data.iterrows():
            for i in range(3):  # 분기당 3개월
                new_row = row.copy()
                new_row['TA_YM'] = row['기간'] + pd.DateOffset(months=i)
                rental_expanded.append(new_row)

        rental_df = pd.DataFrame(rental_expanded)
        self.merged_data = pd.merge(
            self.merged_data,
            rental_df[['TA_YM', '행정구역', '전체(단위:원/평)', '1층(단위:원/평)', '1층 외(단위:원/평)']],
            left_on=['TA_YM', 'LEGAL_DONG'],
            right_on=['TA_YM', '행정구역'],
            how='left'
        )

        # 유동인구 데이터 (분기 -> 월로 확장)
        self.flow_data['기간'] = pd.to_datetime(self.flow_data['기간(분기)'], format='%Y%m')
        flow_expanded = []
        for _, row in self.flow_data.iterrows():
            for i in range(3):
                new_row = row.copy()
                new_row['TA_YM'] = row['기간'] + pd.DateOffset(months=i)
                flow_expanded.append(new_row)

        flow_df = pd.DataFrame(flow_expanded)
        self.merged_data = pd.merge(
            self.merged_data,
            flow_df[['TA_YM', '행적구역', '유동인구(단위:명/1ha)', '주거인구(단위:명/1ha)', '직장인구(단위:명/1ha)']],
            left_on=['TA_YM', 'LEGAL_DONG'],
            right_on=['TA_YM', '행적구역'],
            how='left'
        )

        print(f"✓ 통합 데이터: {len(self.merged_data):,}건\n")

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

    def create_advanced_features(self):
        """고급 시계열 특성 생성"""
        print("🔧 고급 특성 엔지니어링 중...")

        # 정렬
        self.merged_data = self.merged_data.sort_values(['ENCODED_MCT', 'TA_YM'])

        # 운영 기간
        self.merged_data['운영개월수'] = self.merged_data.groupby('ENCODED_MCT').cumcount() + 1

        # 숫자형 변환
        numeric_cols = [
            'MCT_OPE_MS_CN', 'RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT',
            'M12_SME_RY_SAA_PCE_RT', 'M12_SME_BZN_SAA_PCE_RT'
        ]

        for col in numeric_cols:
            if col in self.merged_data.columns:
                self.merged_data[f'{col}_num'] = self.merged_data[col].apply(self.extract_numeric_value)

        # 고객 특성 숫자형
        customer_cols = [
            'M12_MAL_1020_RAT', 'M12_MAL_30_RAT', 'M12_MAL_40_RAT', 'M12_MAL_50_RAT', 'M12_MAL_60_RAT',
            'M12_FME_1020_RAT', 'M12_FME_30_RAT', 'M12_FME_40_RAT', 'M12_FME_50_RAT', 'M12_FME_60_RAT',
            'MCT_UE_CLN_REU_RAT', 'MCT_UE_CLN_NEW_RAT'
        ]

        for col in customer_cols:
            if col in self.merged_data.columns:
                self.merged_data[col] = pd.to_numeric(self.merged_data[col], errors='coerce')

        # === 시계열 특성 ===

        # 1. 변화율 (1개월, 3개월, 6개월)
        key_metrics = ['RC_M1_SAA_num', 'RC_M1_TO_UE_CT_num', 'RC_M1_UE_CUS_CN_num']

        for col in key_metrics:
            if col in self.merged_data.columns:
                # 1개월 변화율
                self.merged_data[f'{col}_변화율_1M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].pct_change(1) * 100

                # 3개월 변화율
                self.merged_data[f'{col}_변화율_3M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].pct_change(3) * 100

                # 6개월 변화율
                self.merged_data[f'{col}_변화율_6M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].pct_change(6) * 100

        # 2. 이동평균 (3개월, 6개월)
        for col in key_metrics:
            if col in self.merged_data.columns:
                self.merged_data[f'{col}_MA3'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].transform(
                        lambda x: x.rolling(window=3, min_periods=1).mean()
                    )

                self.merged_data[f'{col}_MA6'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].transform(
                        lambda x: x.rolling(window=6, min_periods=1).mean()
                    )

        # 3. 추세 (3개월, 6개월 평균 변화율)
        for col in key_metrics:
            col_change = f'{col}_변화율_1M'
            if col_change in self.merged_data.columns:
                self.merged_data[f'{col}_추세3M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col_change].transform(
                        lambda x: x.rolling(window=3, min_periods=1).mean()
                    )

                self.merged_data[f'{col}_추세6M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col_change].transform(
                        lambda x: x.rolling(window=6, min_periods=1).mean()
                    )

        # 4. 변동성 (표준편차)
        for col in key_metrics:
            if col in self.merged_data.columns:
                self.merged_data[f'{col}_STD3M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].transform(
                        lambda x: x.rolling(window=3, min_periods=1).std()
                    )

        # 5. 최대/최소 대비 비율
        for col in key_metrics:
            if col in self.merged_data.columns:
                self.merged_data[f'{col}_MAX6M'] = \
                    self.merged_data.groupby('ENCODED_MCT')[col].transform(
                        lambda x: x.rolling(window=6, min_periods=1).max()
                    )
                self.merged_data[f'{col}_vs_MAX'] = \
                    (self.merged_data[col] / self.merged_data[f'{col}_MAX6M'] * 100)

        # 6. 계절성 (월별 더미)
        self.merged_data['월'] = self.merged_data['TA_YM'].dt.month
        self.merged_data['분기'] = self.merged_data['TA_YM'].dt.quarter

        # 7. 연속 하락 개월 수
        for col in key_metrics:
            col_change = f'{col}_변화율_1M'
            if col_change in self.merged_data.columns:
                # 하락 여부
                self.merged_data[f'{col}_하락여부'] = (self.merged_data[col_change] < 0).astype(int)

                # 연속 하락 개월
                def count_consecutive_decline(group):
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
                    self.merged_data.groupby('ENCODED_MCT')[f'{col}_하락여부'].apply(count_consecutive_decline).values

        # 8. 임대료 대비 매출 효율
        if '전체(단위:원/평)' in self.merged_data.columns and 'RC_M1_SAA_num' in self.merged_data.columns:
            self.merged_data['임대료대비매출효율'] = \
                self.merged_data['RC_M1_SAA_num'] / (self.merged_data['전체(단위:원/평)'] / 1000 + 1)

        # 9. 유동인구 대비 매출 효율
        if '유동인구(단위:명/1ha)' in self.merged_data.columns and 'RC_M1_SAA_num' in self.merged_data.columns:
            self.merged_data['유동인구대비매출'] = \
                self.merged_data['RC_M1_SAA_num'] / (self.merged_data['유동인구(단위:명/1ha)'] / 1000 + 1)

        # 10. 고객 다양성 지수 (성별/연령별 분포의 엔트로피)
        gender_age_cols = [col for col in customer_cols if 'MAL' in col or 'FME' in col]
        if gender_age_cols:
            # 엔트로피 계산
            gender_age_data = self.merged_data[gender_age_cols].fillna(0)

            def calculate_entropy(row):
                probs = row / (row.sum() + 1e-10)
                probs = probs[probs > 0]
                return -np.sum(probs * np.log2(probs + 1e-10))

            self.merged_data['고객다양성지수'] = gender_age_data.apply(calculate_entropy, axis=1)

        print(f"✓ 특성 생성 완료: {len(self.merged_data.columns)}개 컬럼\n")

    def create_target_labels(self):
        """타겟 레이블 생성 (룰 기반으로 학습용 레이블 생성)"""
        print("🎯 타겟 레이블 생성 중...")

        # 위험점수 계산 (룰 기반)
        self.merged_data['위험점수_rule'] = 0

        # 1. 매출 하락
        sales_trend = self.merged_data['RC_M1_SAA_num_추세3M'].fillna(0)
        self.merged_data.loc[sales_trend < -30, '위험점수_rule'] += 40
        self.merged_data.loc[(sales_trend >= -30) & (sales_trend < -15), '위험점수_rule'] += 30
        self.merged_data.loc[(sales_trend >= -15) & (sales_trend < -5), '위험점수_rule'] += 15

        # 2. 이용건수 감소
        usage_trend = self.merged_data['RC_M1_TO_UE_CT_num_추세3M'].fillna(0)
        self.merged_data.loc[usage_trend < -30, '위험점수_rule'] += 30
        self.merged_data.loc[(usage_trend >= -30) & (usage_trend < -15), '위험점수_rule'] += 20
        self.merged_data.loc[(usage_trend >= -15) & (usage_trend < -5), '위험점수_rule'] += 10

        # 3. 고객 수 감소
        customer_trend = self.merged_data['RC_M1_UE_CUS_CN_num_추세3M'].fillna(0)
        self.merged_data.loc[customer_trend < -30, '위험점수_rule'] += 20
        self.merged_data.loc[(customer_trend >= -30) & (customer_trend < -15), '위험점수_rule'] += 13
        self.merged_data.loc[(customer_trend >= -15) & (customer_trend < -5), '위험점수_rule'] += 7

        # 4. 절대 매출 수준
        sales_level = self.merged_data['RC_M1_SAA_num'].fillna(50)
        self.merged_data.loc[sales_level > 90, '위험점수_rule'] += 10
        self.merged_data.loc[(sales_level > 75) & (sales_level <= 90), '위험점수_rule'] += 7

        # 5. 연속 하락
        if 'RC_M1_SAA_num_연속하락' in self.merged_data.columns:
            self.merged_data.loc[self.merged_data['RC_M1_SAA_num_연속하락'] >= 3, '위험점수_rule'] += 10

        # 6. 재구매율
        if 'MCT_UE_CLN_REU_RAT' in self.merged_data.columns:
            reuse = self.merged_data['MCT_UE_CLN_REU_RAT'].fillna(50)
            self.merged_data.loc[reuse < 10, '위험점수_rule'] += 5

        # 경보 레벨 (0-3)
        conditions = [
            self.merged_data['위험점수_rule'] < 25,
            (self.merged_data['위험점수_rule'] >= 25) & (self.merged_data['위험점수_rule'] < 50),
            (self.merged_data['위험점수_rule'] >= 50) & (self.merged_data['위험점수_rule'] < 75),
            self.merged_data['위험점수_rule'] >= 75
        ]
        self.merged_data['경보레벨'] = np.select(conditions, [0, 1, 2, 3], default=0)

        print(f"✓ 레이블 분포:")
        for level in range(4):
            count = (self.merged_data['경보레벨'] == level).sum()
            pct = count / len(self.merged_data) * 100
            print(f"  {self.WARNING_LEVELS[level]['emoji']} {self.WARNING_LEVELS[level]['name']}: {count:,}건 ({pct:.1f}%)")
        print()

    def prepare_features(self):
        """LightGBM 학습용 특성 준비"""
        print("🔨 학습용 특성 준비 중...")

        # 결측치가 너무 많거나 사용하지 않을 컬럼 제외
        exclude_cols = [
            'ENCODED_MCT', 'TA_YM', 'MCT_BSE_AR', 'MCT_NM', 'MCT_BRD_NUM', 'ARE_D', 'MCT_ME_D',
            '__filled_flag__', '__fill_method__', '__impute_source__',
            '경보레벨', '위험점수_rule', 'LEGAL_DONG', '행정구역', '행적구역',
            'MCT_SIGUNGU_NM', '기간'
        ]

        # 범주형 변수 인코딩
        categorical_cols = ['HPSN_MCT_ZCD_NM', 'HPSN_MCT_BZN_CD_NM', '월', '분기']

        for col in categorical_cols:
            if col in self.merged_data.columns:
                le = LabelEncoder()
                self.merged_data[f'{col}_encoded'] = le.fit_transform(
                    self.merged_data[col].fillna('Unknown').astype(str)
                )
                self.label_encoders[col] = le

        # 특성 컬럼 선택
        all_cols = set(self.merged_data.columns)
        exclude_set = set(exclude_cols)

        # 숫자형 + 인코딩된 범주형만
        potential_features = all_cols - exclude_set

        self.feature_cols = []
        for col in potential_features:
            if self.merged_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # 결측치가 80% 이상인 컬럼 제외
                missing_rate = self.merged_data[col].isna().sum() / len(self.merged_data)
                if missing_rate < 0.8:
                    self.feature_cols.append(col)

        print(f"✓ 선택된 특성: {len(self.feature_cols)}개")
        print(f"  주요 특성: {self.feature_cols[:10]}")
        print()

    def train_lgbm_model(self):
        """LightGBM 모델 학습"""
        print("=" * 80)
        print("🚀 LightGBM 모델 학습 중...")
        print("=" * 80)

        # 학습 데이터 준비 (운영개월수 3개월 이상만 사용)
        train_data = self.merged_data[self.merged_data['운영개월수'] >= 3].copy()

        X = train_data[self.feature_cols].fillna(0)
        y = train_data['경보레벨']

        # Train/Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✓ 학습 데이터: {len(X_train):,}건")
        print(f"✓ 테스트 데이터: {len(X_test):,}건\n")

        # LightGBM 데이터셋
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # 파라미터
        params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_child_samples': 20
        }

        # 학습
        print("🔄 모델 학습 시작...")
        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_eval],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        print("\n✓ 학습 완료!\n")

        # 예측 및 평가
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        y_pred_class = np.argmax(y_pred, axis=1)

        print("=" * 80)
        print("📊 모델 성능 평가")
        print("=" * 80)
        print("\n[분류 리포트]")
        print(classification_report(
            y_test, y_pred_class,
            target_names=[f"{self.WARNING_LEVELS[i]['emoji']} {self.WARNING_LEVELS[i]['name']}"
                          for i in range(4)]
        ))

        # 전체 데이터에 예측 수행
        print("🔮 전체 데이터 예측 중...")
        X_all = self.merged_data[self.feature_cols].fillna(0)
        y_pred_all = self.model.predict(X_all, num_iteration=self.model.best_iteration)

        self.merged_data['예측_경보레벨'] = np.argmax(y_pred_all, axis=1)
        self.merged_data['예측_위험확률_0'] = y_pred_all[:, 0]
        self.merged_data['예측_위험확률_1'] = y_pred_all[:, 1]
        self.merged_data['예측_위험확률_2'] = y_pred_all[:, 2]
        self.merged_data['예측_위험확률_3'] = y_pred_all[:, 3]

        # 위험점수 (0-100)
        self.merged_data['예측_위험점수'] = (
            self.merged_data['예측_위험확률_1'] * 25 +
            self.merged_data['예측_위험확률_2'] * 50 +
            self.merged_data['예측_위험확률_3'] * 75
        )

        print("✓ 예측 완료!\n")

        return X_test, y_test, y_pred

    def visualize_feature_importance(self, save_path='feature_importance.png'):
        """특성 중요도 시각화"""
        print("📊 특성 중요도 시각화 중...")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('LightGBM 특성 중요도 분석', fontsize=18, fontweight='bold')

        # 1. Gain 기준
        importance_gain = self.model.feature_importance(importance_type='gain')
        indices = np.argsort(importance_gain)[::-1][:30]

        ax1 = axes[0]
        ax1.barh(range(30), importance_gain[indices], color='skyblue', edgecolor='black')
        ax1.set_yticks(range(30))
        ax1.set_yticklabels([self.feature_cols[i] for i in indices], fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel('중요도 (Gain)', fontsize=12, fontweight='bold')
        ax1.set_title('특성 중요도 - Gain 기준 (Top 30)', fontsize=14, fontweight='bold', pad=10)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')

        # 2. Split 기준
        importance_split = self.model.feature_importance(importance_type='split')
        indices_split = np.argsort(importance_split)[::-1][:30]

        ax2 = axes[1]
        ax2.barh(range(30), importance_split[indices_split], color='lightcoral', edgecolor='black')
        ax2.set_yticks(range(30))
        ax2.set_yticklabels([self.feature_cols[i] for i in indices_split], fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('중요도 (Split)', fontsize=12, fontweight='bold')
        ax2.set_title('특성 중요도 - Split 기준 (Top 30)', fontsize=14, fontweight='bold', pad=10)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def visualize_confusion_matrix(self, y_test, y_pred, save_path='confusion_matrix.png'):
        """혼동 행렬 시각화"""
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

    def visualize_predictions(self, save_path='prediction_analysis.png'):
        """예측 결과 시각화"""
        print("📊 예측 결과 시각화 중...")

        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'LightGBM 기반 경영 위기 예측 분석 ({latest_month.strftime("%Y년 %m월")})',
                     fontsize=18, fontweight='bold', y=0.995)

        # 1. 예측 경보 레벨 분포
        ax1 = axes[0, 0]
        pred_counts = latest_data['예측_경보레벨'].value_counts().sort_index()
        colors = [self.WARNING_LEVELS[i]['color'] for i in range(4)]

        bars = ax1.bar(range(len(pred_counts)), pred_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_xticks(range(4))
        ax1.set_xticklabels([f"{self.WARNING_LEVELS[i]['emoji']} {self.WARNING_LEVELS[i]['name']}"
                              for i in range(4)], fontsize=11)
        ax1.set_ylabel('가맹점 수', fontsize=12, fontweight='bold')
        ax1.set_title('예측 경보 레벨별 분포', fontsize=14, fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}개\n({height/len(latest_data)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. 예측 위험점수 분포
        ax2 = axes[0, 1]
        ax2.hist(latest_data['예측_위험점수'], bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=25, color='yellow', linestyle='--', linewidth=2, label='주의')
        ax2.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='경고')
        ax2.axvline(x=75, color='red', linestyle='--', linewidth=2, label='위험')
        ax2.set_xlabel('예측 위험점수', fontsize=12, fontweight='bold')
        ax2.set_ylabel('가맹점 수', fontsize=12, fontweight='bold')
        ax2.set_title('예측 위험점수 분포', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # 3. 룰 기반 vs ML 예측 비교
        ax3 = axes[1, 0]
        comparison = pd.crosstab(latest_data['경보레벨'], latest_data['예측_경보레벨'])
        sns.heatmap(comparison, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, cbar_kws={'label': '건수'})
        ax3.set_xlabel('ML 예측 레벨', fontsize=12, fontweight='bold')
        ax3.set_ylabel('룰 기반 레벨', fontsize=12, fontweight='bold')
        ax3.set_title('룰 기반 vs ML 예측 비교', fontsize=14, fontweight='bold', pad=10)
        ax3.set_xticklabels([self.WARNING_LEVELS[i]['name'] for i in range(4)], rotation=0)
        ax3.set_yticklabels([self.WARNING_LEVELS[i]['name'] for i in range(4)], rotation=0)

        # 4. 업종별 평균 예측 위험점수
        ax4 = axes[1, 1]
        if 'HPSN_MCT_ZCD_NM' in latest_data.columns:
            industry_risk = latest_data.groupby('HPSN_MCT_ZCD_NM')['예측_위험점수'].mean().sort_values(ascending=False).head(15)

            bars = ax4.barh(range(len(industry_risk)), industry_risk.values, color='coral', alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(industry_risk)))
            ax4.set_yticklabels(industry_risk.index, fontsize=9)
            ax4.set_xlabel('평균 예측 위험점수', fontsize=12, fontweight='bold')
            ax4.set_title('업종별 평균 예측 위험점수 (Top 15)', fontsize=14, fontweight='bold', pad=10)
            ax4.grid(axis='x', alpha=0.3, linestyle='--')

            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width >= 75:
                    bar.set_color('red')
                elif width >= 50:
                    bar.set_color('orange')
                elif width >= 25:
                    bar.set_color('yellow')
                else:
                    bar.set_color('green')
                bar.set_alpha(0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def visualize_merchant_lgbm(self, encoded_mct, save_path='merchant_lgbm_detail.png'):
        """LightGBM 예측 기반 가맹점 상세 분석"""
        print(f"🔍 가맹점 상세 분석 (LightGBM): {encoded_mct}")

        merchant_ts = self.merged_data[self.merged_data['ENCODED_MCT'] == encoded_mct].copy()
        merchant_ts = merchant_ts.sort_values('TA_YM')

        if len(merchant_ts) == 0:
            print(f"❌ 해당 가맹점 데이터 없음")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        mct_info = merchant_ts.iloc[-1]
        fig.suptitle(f'가맹점 상세 분석 (LightGBM 예측)\n{mct_info.get("MCT_NM", "N/A")} ({mct_info.get("HPSN_MCT_BZN_CD_NM", "N/A")})',
                     fontsize=18, fontweight='bold', y=0.98)

        # 1. 경보 레벨 비교 (룰 vs ML)
        ax1 = axes[0, 0]
        x = range(len(merchant_ts))
        ax1.plot(merchant_ts['TA_YM'], merchant_ts['경보레벨'],
                marker='o', linewidth=2.5, label='룰 기반', color='#2E86AB')
        ax1.plot(merchant_ts['TA_YM'], merchant_ts['예측_경보레벨'],
                marker='s', linewidth=2.5, label='ML 예측', color='#F18F01')
        ax1.fill_between(merchant_ts['TA_YM'], merchant_ts['경보레벨'], alpha=0.2, color='#2E86AB')
        ax1.fill_between(merchant_ts['TA_YM'], merchant_ts['예측_경보레벨'], alpha=0.2, color='#F18F01')

        ax1.set_ylabel('경보 레벨', fontsize=11, fontweight='bold')
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels([self.WARNING_LEVELS[i]['name'] for i in range(4)])
        ax1.set_title('경보 레벨 추이 (룰 기반 vs ML 예측)', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)

        # 2. 위험 확률 분포
        ax2 = axes[0, 1]
        ax2.stackplot(merchant_ts['TA_YM'],
                     merchant_ts['예측_위험확률_0'] * 100,
                     merchant_ts['예측_위험확률_1'] * 100,
                     merchant_ts['예측_위험확률_2'] * 100,
                     merchant_ts['예측_위험확률_3'] * 100,
                     labels=['안전', '주의', '경고', '위험'],
                     colors=['green', 'yellow', 'orange', 'red'],
                     alpha=0.7)
        ax2.set_ylabel('위험 확률 (%)', fontsize=11, fontweight='bold')
        ax2.set_title('레벨별 위험 확률 추이', fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)

        # 3. 위험점수 비교
        ax3 = axes[1, 0]
        ax3.plot(merchant_ts['TA_YM'], merchant_ts['위험점수_rule'],
                marker='o', linewidth=2, label='룰 기반 점수', color='#2E86AB')
        ax3.plot(merchant_ts['TA_YM'], merchant_ts['예측_위험점수'],
                marker='s', linewidth=2, label='ML 예측 점수', color='#F18F01')

        ax3.axhline(y=25, color='yellow', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.axhline(y=75, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

        ax3.set_ylabel('위험점수', fontsize=11, fontweight='bold')
        ax3.set_title('위험점수 비교', fontsize=13, fontweight='bold', pad=10)
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(axis='x', rotation=45)

        # 4. 현재 상태 요약
        ax4 = axes[1, 1]
        ax4.axis('off')

        latest = merchant_ts.iloc[-1]
        pred_level = int(latest['예측_경보레벨'])
        pred_info = self.WARNING_LEVELS[pred_level]

        summary = f"""
        【 ML 예측 경보 상태 】

        {pred_info['emoji']} 예측 레벨: {pred_info['name']}
        📊 예측 위험점수: {latest['예측_위험점수']:.1f}점

        【 레벨별 확률 】

        🟢 안전: {latest['예측_위험확률_0']*100:.1f}%
        🟡 주의: {latest['예측_위험확률_1']*100:.1f}%
        🟠 경고: {latest['예측_위험확률_2']*100:.1f}%
        🔴 위험: {latest['예측_위험확률_3']*100:.1f}%

        【 최근 추세 】

        📈 매출 변화: {latest.get('RC_M1_SAA_num_추세3M', 0):.1f}%
        👥 고객 변화: {latest.get('RC_M1_UE_CUS_CN_num_추세3M', 0):.1f}%
        🔄 연속 하락: {latest.get('RC_M1_SAA_num_연속하락', 0):.0f}개월

        【 AI 권장 조치 】
        """

        if pred_level == 3:
            summary += "\n🔴 즉각 대응 필요\n    ML 모델이 높은 위험 감지\n    전문 컨설팅 권장"
        elif pred_level == 2:
            summary += "\n🟠 면밀한 모니터링\n    경영 개선 방안 수립\n    정기 점검 강화"
        elif pred_level == 1:
            summary += "\n🟡 예방적 조치\n    추세 관찰 필요\n    개선 기회 탐색"
        else:
            summary += "\n🟢 안정적 운영\n    현 전략 유지\n    성장 기회 모색"

        ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=pred_info['color'], alpha=0.2),
                family=['gulim', 'Symbol'])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 저장: {save_path}\n")
        plt.show()

    def generate_report(self, output_file='lgbm_warning_report.csv'):
        """경보 리포트 생성"""
        print("📝 경보 리포트 생성 중...")

        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month].copy()

        report_cols = [
            'ENCODED_MCT', 'MCT_NM', 'HPSN_MCT_BZN_CD_NM', 'MCT_SIGUNGU_NM',
            '경보레벨', '예측_경보레벨', '위험점수_rule', '예측_위험점수',
            '예측_위험확률_0', '예측_위험확률_1', '예측_위험확률_2', '예측_위험확률_3',
            'RC_M1_SAA_num', 'RC_M1_SAA_num_추세3M', 'RC_M1_SAA_num_연속하락',
            '운영개월수'
        ]

        report_cols = [col for col in report_cols if col in latest_data.columns]
        report = latest_data[report_cols].copy()
        report = report.sort_values('예측_위험점수', ascending=False)

        report.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 리포트 저장: {output_file}")
        print(f"  총 {len(report):,}개 가맹점\n")

        return report

    def print_summary(self):
        """요약 통계"""
        print("=" * 80)
        print("📊 LightGBM 경영 위기 조기 경보 시스템 - 요약")
        print("=" * 80)

        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month]

        print(f"\n📅 분석 기준: {latest_month.strftime('%Y년 %m월')}")
        print(f"🏢 분석 가맹점 수: {len(latest_data):,}개\n")

        print("【 ML 예측 경보 레벨별 현황 】")
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
            print(f"  {idx:2d}. {row.MCT_NM:15s} | {row.HPSN_MCT_BZN_CD_NM:15s} | {row.예측_위험점수:5.1f}점 | {level_name}")

        print("\n" + "=" * 80 + "\n")


def main():
    """메인 실행"""
    print("=" * 80)
    print("🚀 LightGBM 기반 경영 위기 조기 경보 시스템")
    print("=" * 80)
    print()

    ews = LGBMEarlyWarningSystem(data_path='./data/')

    # 1. 데이터 로드
    ews.load_data()

    # 2. 데이터 통합
    ews.merge_all_data()

    # 3. 고급 특성 생성
    ews.create_advanced_features()

    # 4. 타겟 레이블 생성
    ews.create_target_labels()

    # 5. 특성 준비
    ews.prepare_features()

    # 6. LightGBM 모델 학습
    X_test, y_test, y_pred = ews.train_lgbm_model()

    # 7. 요약 통계
    ews.print_summary()

    # 8. 시각화
    print("📊 시각화 생성 중...\n")

    ews.visualize_feature_importance('lgbm_feature_importance.png')
    ews.visualize_confusion_matrix(y_test, y_pred, 'lgbm_confusion_matrix.png')
    ews.visualize_predictions('lgbm_prediction_analysis.png')

    # 9. 리포트 생성
    report = ews.generate_report('lgbm_warning_report.csv')

    # 10. 고위험 가맹점 상세 분석
    print("🔍 고위험 가맹점 상세 분석...\n")
    top_risk = report.nlargest(3, '예측_위험점수')['ENCODED_MCT'].values

    for idx, mct_id in enumerate(top_risk, 1):
        ews.visualize_merchant_lgbm(mct_id, f'lgbm_merchant_top{idx}.png')

    print("=" * 80)
    print("✅ LightGBM 경영 위기 조기 경보 시스템 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  📊 lgbm_feature_importance.png - 특성 중요도")
    print("  📊 lgbm_confusion_matrix.png - 혼동 행렬")
    print("  📊 lgbm_prediction_analysis.png - 예측 분석")
    print("  🔍 lgbm_merchant_top1~3.png - 고위험 가맹점 상세")
    print("  📝 lgbm_warning_report.csv - 전체 경보 리포트")
    print()


if __name__ == "__main__":
    main()
