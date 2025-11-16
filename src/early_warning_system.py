"""
경영 위기 조기 경보 시스템
4단계 경보 시스템: 안전(Green) -> 주의(Yellow) -> 경고(Orange) -> 위험(Red)
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

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class EarlyWarningSystem:
    """경영 위기 조기 경보 시스템"""

    def __init__(self, data_path='./data/'):
        """
        초기화
        Args:
            data_path: 데이터 파일 경로
        """
        self.data_path = data_path
        self.merchant_data = None
        self.sales_data = None
        self.customer_data = None
        self.rental_data = None
        self.flow_data = None
        self.merged_data = None

        # 경보 레벨 정의
        self.WARNING_LEVELS = {
            0: {'name': '안전', 'color': 'green', 'emoji': '🟢'},
            1: {'name': '주의', 'color': 'yellow', 'emoji': '🟡'},
            2: {'name': '경고', 'color': 'orange', 'emoji': '🟠'},
            3: {'name': '위험', 'color': 'red', 'emoji': '🔴'}
        }

    def load_data(self):
        """데이터 로드"""
        print("📊 데이터 로드 중...")

        # 가맹점 정보
        self.merchant_data = pd.read_csv(f'{self.data_path}big_data_set1_f_v2.csv', encoding='utf-8-sig')
        print(f"  ✓ 가맹점 정보: {len(self.merchant_data):,}개")

        # 매출 데이터 (시계열)
        self.sales_data = pd.read_csv(f'{self.data_path}big_data_set2_f_sorted.csv', encoding='utf-8-sig')
        self.sales_data['TA_YM'] = pd.to_datetime(self.sales_data['TA_YM'], format='%Y%m')
        print(f"  ✓ 매출 데이터: {len(self.sales_data):,}건")

        # 고객 데이터 (시계열)
        self.customer_data = pd.read_csv(f'{self.data_path}big_data_set3_f_sorted.csv', encoding='utf-8-sig')
        self.customer_data['TA_YM'] = pd.to_datetime(self.customer_data['TA_YM'], format='%Y%m')
        print(f"  ✓ 고객 데이터: {len(self.customer_data):,}건")

        # 임대료 데이터
        self.rental_data = pd.read_csv(f'{self.data_path}rental_p.csv', encoding='utf-8-sig')
        print(f"  ✓ 임대료 데이터: {len(self.rental_data):,}건")

        # 유동인구 데이터
        self.flow_data = pd.read_csv(f'{self.data_path}flow_f.csv', encoding='utf-8-sig')
        print(f"  ✓ 유동인구 데이터: {len(self.flow_data):,}건")

        print("✅ 데이터 로드 완료!\n")

    def merge_all_data(self):
        """모든 데이터 통합"""
        print("🔗 데이터 통합 중...")

        # 매출 + 고객 데이터 병합
        self.merged_data = pd.merge(
            self.sales_data,
            self.customer_data,
            on=['ENCODED_MCT', 'TA_YM'],
            how='inner'
        )

        # 가맹점 정보 추가
        self.merged_data = pd.merge(
            self.merged_data,
            self.merchant_data,
            on='ENCODED_MCT',
            how='left'
        )

        print(f"✅ 통합 데이터: {len(self.merged_data):,}건\n")

    def extract_numeric_value(self, value_str):
        """구간 문자열에서 중간값 추출 (예: '4_50-75%' -> 62.5)"""
        if pd.isna(value_str) or value_str == '':
            return np.nan

        value_str = str(value_str)

        # '6_90%초과' 케이스
        if '90%초과' in value_str or '하위 10%' in value_str:
            return 95.0

        # 숫자_범위% 형태
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

    def create_time_series_features(self):
        """시계열 특성 생성"""
        print("📈 시계열 특성 생성 중...")

        # 가맹점별 정렬
        self.merged_data = self.merged_data.sort_values(['ENCODED_MCT', 'TA_YM'])

        # 운영 기간 (월 수)
        self.merged_data['운영개월수'] = self.merged_data.groupby('ENCODED_MCT').cumcount() + 1

        # 매출 추세 계산을 위한 숫자형 변환
        numeric_cols = ['RC_M1_SAA', 'RC_M1_TO_UE_CT', 'RC_M1_UE_CUS_CN', 'RC_M1_AV_NP_AT', 'MCT_OPE_MS_CN']

        for col in numeric_cols:
            if col in self.merged_data.columns:
                self.merged_data[f'{col}_numeric'] = self.merged_data[col].apply(self.extract_numeric_value)

        # 시계열 변화율 계산 (전월 대비)
        for col in ['RC_M1_SAA_numeric', 'RC_M1_TO_UE_CT_numeric', 'RC_M1_UE_CUS_CN_numeric']:
            if col in self.merged_data.columns:
                self.merged_data[f'{col}_변화율'] = self.merged_data.groupby('ENCODED_MCT')[col].pct_change() * 100

                # 3개월 이동평균
                self.merged_data[f'{col}_MA3'] = self.merged_data.groupby('ENCODED_MCT')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )

                # 3개월 추세 (최근 3개월 평균 변화)
                self.merged_data[f'{col}_추세3M'] = self.merged_data.groupby('ENCODED_MCT')[f'{col}_변화율'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )

        print(f"✅ 시계열 특성 생성 완료!\n")

    def calculate_warning_score(self):
        """경보 점수 계산"""
        print("⚠️ 경보 점수 계산 중...")

        # 점수 초기화
        self.merged_data['위험점수'] = 0

        # 1. 매출 하락 점수 (가중치: 40%)
        sales_change = self.merged_data['RC_M1_SAA_numeric_추세3M'].fillna(0)
        self.merged_data.loc[sales_change < -30, '위험점수'] += 40
        self.merged_data.loc[(sales_change >= -30) & (sales_change < -15), '위험점수'] += 30
        self.merged_data.loc[(sales_change >= -15) & (sales_change < -5), '위험점수'] += 15

        # 2. 이용건수 감소 점수 (가중치: 30%)
        usage_change = self.merged_data['RC_M1_TO_UE_CT_numeric_추세3M'].fillna(0)
        self.merged_data.loc[usage_change < -30, '위험점수'] += 30
        self.merged_data.loc[(usage_change >= -30) & (usage_change < -15), '위험점수'] += 20
        self.merged_data.loc[(usage_change >= -15) & (usage_change < -5), '위험점수'] += 10

        # 3. 고객 수 감소 점수 (가중치: 20%)
        customer_change = self.merged_data['RC_M1_UE_CUS_CN_numeric_추세3M'].fillna(0)
        self.merged_data.loc[customer_change < -30, '위험점수'] += 20
        self.merged_data.loc[(customer_change >= -30) & (customer_change < -15), '위험점수'] += 13
        self.merged_data.loc[(customer_change >= -15) & (customer_change < -5), '위험점수'] += 7

        # 4. 절대 매출 수준 (가중치: 10%)
        sales_level = self.merged_data['RC_M1_SAA_numeric'].fillna(50)
        self.merged_data.loc[sales_level > 90, '위험점수'] += 10  # 매출이 하위 10%
        self.merged_data.loc[(sales_level > 75) & (sales_level <= 90), '위험점수'] += 7
        self.merged_data.loc[(sales_level > 50) & (sales_level <= 75), '위험점수'] += 3

        # 5. 운영 안정성 점수 추가
        # 운영개월수가 짧은 경우 가중치
        self.merged_data.loc[self.merged_data['운영개월수'] < 6, '위험점수'] += 5

        # 6. 재구매율 감소 (MCT_UE_CLN_REU_RAT)
        if 'MCT_UE_CLN_REU_RAT' in self.merged_data.columns:
            reuse_rate = self.merged_data['MCT_UE_CLN_REU_RAT'].fillna(50)
            self.merged_data.loc[reuse_rate < 10, '위험점수'] += 5
            self.merged_data.loc[(reuse_rate >= 10) & (reuse_rate < 30), '위험점수'] += 3

        # 경보 레벨 결정 (0-100 점수 -> 0-3 레벨)
        conditions = [
            self.merged_data['위험점수'] < 25,
            (self.merged_data['위험점수'] >= 25) & (self.merged_data['위험점수'] < 50),
            (self.merged_data['위험점수'] >= 50) & (self.merged_data['위험점수'] < 75),
            self.merged_data['위험점수'] >= 75
        ]
        choices = [0, 1, 2, 3]

        self.merged_data['경보레벨'] = np.select(conditions, choices, default=0)

        # 경보명 추가
        self.merged_data['경보명'] = self.merged_data['경보레벨'].map(
            lambda x: self.WARNING_LEVELS[x]['name']
        )

        print(f"✅ 경보 점수 계산 완료!\n")

    def get_latest_warnings(self, top_n=20):
        """최신 경보 현황 조회"""
        # 최신 월 데이터만 추출
        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month].copy()

        # 위험 순으로 정렬
        latest_data = latest_data.sort_values('위험점수', ascending=False).head(top_n)

        return latest_data

    def visualize_warning_distribution(self, save_path='warning_distribution.png'):
        """경보 레벨 분포 시각화"""
        print("📊 경보 레벨 분포 시각화 중...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('경영 위기 조기 경보 시스템 - 전체 현황', fontsize=20, fontweight='bold', y=0.995)

        # 최신 데이터
        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month]

        # 1. 경보 레벨별 가맹점 수
        ax1 = axes[0, 0]
        warning_counts = latest_data['경보명'].value_counts()
        colors = [self.WARNING_LEVELS[i]['color'] for i in range(4)]

        bars = ax1.bar(range(len(warning_counts)), warning_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_xticks(range(len(warning_counts)))
        ax1.set_xticklabels([f"{self.WARNING_LEVELS[i]['emoji']} {self.WARNING_LEVELS[i]['name']}"
                              for i in range(4)], fontsize=12)
        ax1.set_ylabel('가맹점 수', fontsize=12, fontweight='bold')
        ax1.set_title(f'경보 레벨별 분포 ({latest_month.strftime("%Y년 %m월")})',
                      fontsize=14, fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}개\n({height/len(latest_data)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 2. 시계열 경보 레벨 추이
        ax2 = axes[0, 1]
        monthly_warnings = self.merged_data.groupby(['TA_YM', '경보명']).size().unstack(fill_value=0)

        for level in range(4):
            name = self.WARNING_LEVELS[level]['name']
            if name in monthly_warnings.columns:
                ax2.plot(monthly_warnings.index, monthly_warnings[name],
                        marker='o', linewidth=2.5, markersize=6,
                        label=f"{self.WARNING_LEVELS[level]['emoji']} {name}",
                        color=self.WARNING_LEVELS[level]['color'])

        ax2.set_xlabel('기간', fontsize=12, fontweight='bold')
        ax2.set_ylabel('가맹점 수', fontsize=12, fontweight='bold')
        ax2.set_title('월별 경보 레벨 추이', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)

        # 3. 위험점수 분포
        ax3 = axes[1, 0]
        ax3.hist(latest_data['위험점수'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=25, color='yellow', linestyle='--', linewidth=2, label='주의 (25점)')
        ax3.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='경고 (50점)')
        ax3.axvline(x=75, color='red', linestyle='--', linewidth=2, label='위험 (75점)')
        ax3.set_xlabel('위험점수', fontsize=12, fontweight='bold')
        ax3.set_ylabel('가맹점 수', fontsize=12, fontweight='bold')
        ax3.set_title('위험점수 분포', fontsize=14, fontweight='bold', pad=10)
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        # 4. 업종별 평균 위험점수
        ax4 = axes[1, 1]
        if 'HPSN_MCT_BZN_CD_NM' in latest_data.columns:
            industry_risk = latest_data.groupby('HPSN_MCT_ZCD_NM')['위험점수'].mean().sort_values(ascending=False).head(15)

            bars = ax4.barh(range(len(industry_risk)), industry_risk.values, color='coral', alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(industry_risk)))
            ax4.set_yticklabels(industry_risk.index, fontsize=9)
            ax4.set_xlabel('평균 위험점수', fontsize=12, fontweight='bold')
            ax4.set_title('업종별 평균 위험점수 (Top 15)', fontsize=14, fontweight='bold', pad=10)
            ax4.grid(axis='x', alpha=0.3, linestyle='--')

            # 위험 구간 색상
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

                ax4.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{width:.1f}',
                        ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 저장: {save_path}\n")
        plt.show()

    def visualize_time_series_analysis(self, save_path='timeseries_analysis.png'):
        """시계열 분석 시각화"""
        print("📈 시계열 분석 시각화 중...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('시계열 기반 경영 지표 분석', fontsize=20, fontweight='bold', y=0.995)

        # 월별 평균 지표 계산
        monthly_avg = self.merged_data.groupby('TA_YM').agg({
            'RC_M1_SAA_numeric': 'mean',
            'RC_M1_TO_UE_CT_numeric': 'mean',
            'RC_M1_UE_CUS_CN_numeric': 'mean',
            '위험점수': 'mean'
        }).reset_index()

        # 1. 매출 수준 추이
        ax1 = axes[0, 0]
        ax1.plot(monthly_avg['TA_YM'], monthly_avg['RC_M1_SAA_numeric'],
                marker='o', linewidth=2.5, markersize=7, color='#2E86AB', label='평균 매출 수준')
        ax1.fill_between(monthly_avg['TA_YM'], monthly_avg['RC_M1_SAA_numeric'],
                         alpha=0.3, color='#2E86AB')
        ax1.set_ylabel('매출 수준 (백분위)', fontsize=12, fontweight='bold')
        ax1.set_title('월별 평균 매출 수준 추이', fontsize=14, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(fontsize=10)

        # 2. 이용건수 추이
        ax2 = axes[0, 1]
        ax2.plot(monthly_avg['TA_YM'], monthly_avg['RC_M1_TO_UE_CT_numeric'],
                marker='s', linewidth=2.5, markersize=7, color='#A23B72', label='평균 이용건수')
        ax2.fill_between(monthly_avg['TA_YM'], monthly_avg['RC_M1_TO_UE_CT_numeric'],
                         alpha=0.3, color='#A23B72')
        ax2.set_ylabel('이용건수 (백분위)', fontsize=12, fontweight='bold')
        ax2.set_title('월별 평균 이용건수 추이', fontsize=14, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(fontsize=10)

        # 3. 고객 수 추이
        ax3 = axes[1, 0]
        ax3.plot(monthly_avg['TA_YM'], monthly_avg['RC_M1_UE_CUS_CN_numeric'],
                marker='^', linewidth=2.5, markersize=7, color='#F18F01', label='평균 고객 수')
        ax3.fill_between(monthly_avg['TA_YM'], monthly_avg['RC_M1_UE_CUS_CN_numeric'],
                         alpha=0.3, color='#F18F01')
        ax3.set_ylabel('고객 수 (백분위)', fontsize=12, fontweight='bold')
        ax3.set_title('월별 평균 고객 수 추이', fontsize=14, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(fontsize=10)

        # 4. 평균 위험점수 추이
        ax4 = axes[1, 1]
        ax4.plot(monthly_avg['TA_YM'], monthly_avg['위험점수'],
                marker='D', linewidth=2.5, markersize=7, color='#C73E1D', label='평균 위험점수')
        ax4.fill_between(monthly_avg['TA_YM'], monthly_avg['위험점수'],
                         alpha=0.3, color='#C73E1D')

        # 위험 구간 표시
        ax4.axhline(y=25, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='주의')
        ax4.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='경고')
        ax4.axhline(y=75, color='red', linestyle='--', linewidth=2, alpha=0.7, label='위험')

        ax4.set_ylabel('위험점수', fontsize=12, fontweight='bold')
        ax4.set_title('월별 평균 위험점수 추이', fontsize=14, fontweight='bold', pad=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(fontsize=10, loc='best')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 저장: {save_path}\n")
        plt.show()

    def visualize_merchant_detail(self, encoded_mct, save_path='merchant_detail.png'):
        """특정 가맹점 상세 분석 시각화"""
        print(f"🔍 가맹점 상세 분석: {encoded_mct}")

        merchant_ts = self.merged_data[self.merged_data['ENCODED_MCT'] == encoded_mct].copy()
        merchant_ts = merchant_ts.sort_values('TA_YM')

        if len(merchant_ts) == 0:
            print(f"❌ 해당 가맹점 데이터 없음: {encoded_mct}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 가맹점 정보
        mct_info = merchant_ts.iloc[-1]
        fig.suptitle(f'가맹점 상세 분석\n{mct_info.get("MCT_NM", "N/A")} ({mct_info.get("HPSN_MCT_BZN_CD_NM", "N/A")})',
                     fontsize=18, fontweight='bold', y=0.98)

        # 1. 매출/이용건수/고객 수 추이
        ax1 = axes[0, 0]
        ax1_twin1 = ax1.twinx()
        ax1_twin2 = ax1.twinx()
        ax1_twin2.spines['right'].set_position(('outward', 60))

        l1 = ax1.plot(merchant_ts['TA_YM'], merchant_ts['RC_M1_SAA_numeric'],
                     marker='o', linewidth=2, color='#2E86AB', label='매출 수준')
        l2 = ax1_twin1.plot(merchant_ts['TA_YM'], merchant_ts['RC_M1_TO_UE_CT_numeric'],
                           marker='s', linewidth=2, color='#A23B72', label='이용건수')
        l3 = ax1_twin2.plot(merchant_ts['TA_YM'], merchant_ts['RC_M1_UE_CUS_CN_numeric'],
                           marker='^', linewidth=2, color='#F18F01', label='고객 수')

        ax1.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax1.set_ylabel('매출 수준', fontsize=11, fontweight='bold', color='#2E86AB')
        ax1_twin1.set_ylabel('이용건수', fontsize=11, fontweight='bold', color='#A23B72')
        ax1_twin2.set_ylabel('고객 수', fontsize=11, fontweight='bold', color='#F18F01')

        ax1.tick_params(axis='y', labelcolor='#2E86AB')
        ax1_twin1.tick_params(axis='y', labelcolor='#A23B72')
        ax1_twin2.tick_params(axis='y', labelcolor='#F18F01')
        ax1.tick_params(axis='x', rotation=45)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=9)
        ax1.set_title('핵심 경영 지표 추이', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. 위험점수 및 경보 레벨 추이
        ax2 = axes[0, 1]
        colors_map = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'}
        colors = [colors_map[level] for level in merchant_ts['경보레벨']]

        ax2.bar(merchant_ts['TA_YM'], merchant_ts['위험점수'], color=colors, alpha=0.6, edgecolor='black')
        ax2.axhline(y=25, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=75, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        ax2.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax2.set_ylabel('위험점수', fontsize=11, fontweight='bold')
        ax2.set_title('위험점수 및 경보 레벨 추이', fontsize=13, fontweight='bold', pad=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # 3. 변화율 추이 (3개월 추세)
        ax3 = axes[1, 0]
        ax3.plot(merchant_ts['TA_YM'], merchant_ts['RC_M1_SAA_numeric_추세3M'].fillna(0),
                marker='o', linewidth=2, label='매출 변화율', color='#2E86AB')
        ax3.plot(merchant_ts['TA_YM'], merchant_ts['RC_M1_TO_UE_CT_numeric_추세3M'].fillna(0),
                marker='s', linewidth=2, label='이용건수 변화율', color='#A23B72')
        ax3.plot(merchant_ts['TA_YM'], merchant_ts['RC_M1_UE_CUS_CN_numeric_추세3M'].fillna(0),
                marker='^', linewidth=2, label='고객 수 변화율', color='#F18F01')

        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax3.axhline(y=-5, color='yellow', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axhline(y=-15, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axhline(y=-30, color='red', linestyle='--', linewidth=1, alpha=0.5)

        ax3.set_xlabel('기간', fontsize=11, fontweight='bold')
        ax3.set_ylabel('변화율 (%)', fontsize=11, fontweight='bold')
        ax3.set_title('3개월 평균 변화율 추세', fontsize=13, fontweight='bold', pad=10)
        ax3.legend(fontsize=9, loc='best')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # 4. 현재 상태 요약
        ax4 = axes[1, 1]
        ax4.axis('off')

        latest = merchant_ts.iloc[-1]
        warning_level = int(latest['경보레벨'])
        warning_info = self.WARNING_LEVELS[warning_level]

        summary_text = f"""
        【 현재 경보 상태 】

        {warning_info['emoji']} 경보 레벨: {warning_info['name']}
        📊 위험점수: {latest['위험점수']:.1f}점

        【 최근 추세 (3개월) 】

        📈 매출 변화율: {latest['RC_M1_SAA_numeric_추세3M']:.1f}%
        📊 이용건수 변화율: {latest['RC_M1_TO_UE_CT_numeric_추세3M']:.1f}%
        👥 고객 수 변화율: {latest['RC_M1_UE_CUS_CN_numeric_추세3M']:.1f}%

        【 운영 정보 】

        🏢 업종: {latest.get('HPSN_MCT_BZN_CD_NM', 'N/A')}
        📍 지역: {latest.get('MCT_SIGUNGU_NM', 'N/A')}
        📅 운영 개월: {latest['운영개월수']:.0f}개월

        【 권장 조치 】
        """

        if warning_level == 3:
            summary_text += "\n🔴 즉시 경영 개선 필요\n    - 매출/고객 확보 전략 수립\n    - 비용 구조 재검토\n    - 전문가 컨설팅 고려"
        elif warning_level == 2:
            summary_text += "\n🟠 경영 상황 주시 필요\n    - 마케팅 활동 강화\n    - 고객 만족도 개선\n    - 경쟁력 분석"
        elif warning_level == 1:
            summary_text += "\n🟡 주의 관찰 필요\n    - 추세 모니터링\n    - 예방적 개선 활동\n    - 정기적 점검"
        else:
            summary_text += "\n🟢 양호한 상태 유지\n    - 현재 전략 지속\n    - 성장 기회 탐색\n    - 정기적인 모니터링"

        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=warning_info['color'], alpha=0.2),
                family='monospace')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 저장: {save_path}\n")
        plt.show()

    def generate_warning_report(self, output_file='warning_report.csv'):
        """경보 리포트 생성"""
        print("📝 경보 리포트 생성 중...")

        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month].copy()

        # 필요한 컬럼만 선택
        report_cols = [
            'ENCODED_MCT', 'MCT_NM', 'HPSN_MCT_BZN_CD_NM', 'MCT_SIGUNGU_NM',
            '경보레벨', '경보명', '위험점수',
            'RC_M1_SAA_numeric', 'RC_M1_TO_UE_CT_numeric', 'RC_M1_UE_CUS_CN_numeric',
            'RC_M1_SAA_numeric_추세3M', 'RC_M1_TO_UE_CT_numeric_추세3M', 'RC_M1_UE_CUS_CN_numeric_추세3M',
            '운영개월수'
        ]

        # 존재하는 컬럼만 선택
        report_cols = [col for col in report_cols if col in latest_data.columns]
        report = latest_data[report_cols].copy()

        # 정렬
        report = report.sort_values('위험점수', ascending=False)

        # 저장
        report.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 리포트 저장: {output_file}")
        print(f"   총 {len(report):,}개 가맹점 분석 완료\n")

        return report

    def print_summary(self):
        """요약 통계 출력"""
        print("=" * 80)
        print("📊 경영 위기 조기 경보 시스템 - 요약 리포트")
        print("=" * 80)

        latest_month = self.merged_data['TA_YM'].max()
        latest_data = self.merged_data[self.merged_data['TA_YM'] == latest_month]

        print(f"\n📅 분석 기준: {latest_month.strftime('%Y년 %m월')}")
        print(f"🏢 분석 가맹점 수: {len(latest_data):,}개\n")

        print("【 경보 레벨별 현황 】")
        for level in range(4):
            name = self.WARNING_LEVELS[level]['name']
            emoji = self.WARNING_LEVELS[level]['emoji']
            count = len(latest_data[latest_data['경보레벨'] == level])
            pct = count / len(latest_data) * 100
            print(f"  {emoji} {name:4s}: {count:6,}개 ({pct:5.1f}%)")

        print(f"\n【 위험점수 통계 】")
        print(f"  평균: {latest_data['위험점수'].mean():.1f}점")
        print(f"  중앙값: {latest_data['위험점수'].median():.1f}점")
        print(f"  최대: {latest_data['위험점수'].max():.1f}점")
        print(f"  최소: {latest_data['위험점수'].min():.1f}점")

        print("\n【 고위험 가맹점 (Top 10) 】")
        high_risk = latest_data.nlargest(10, '위험점수')[['MCT_NM', 'HPSN_MCT_BZN_CD_NM', '위험점수', '경보명']]
        for idx, row in enumerate(high_risk.itertuples(), 1):
            print(f"  {idx:2d}. {row.MCT_NM:15s} | {row.HPSN_MCT_BZN_CD_NM:15s} | {row.위험점수:5.1f}점 | {row.경보명}")

        print("\n" + "=" * 80 + "\n")


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("🚨 경영 위기 조기 경보 시스템 (Early Warning System)")
    print("=" * 80)
    print()

    # 시스템 초기화
    ews = EarlyWarningSystem(data_path='./data/')

    # 1. 데이터 로드
    ews.load_data()

    # 2. 데이터 통합
    ews.merge_all_data()

    # 3. 시계열 특성 생성
    ews.create_time_series_features()

    # 4. 경보 점수 계산
    ews.calculate_warning_score()

    # 5. 요약 통계 출력
    ews.print_summary()

    # 6. 시각화
    print("📊 시각화 생성 중...\n")

    # 전체 현황
    ews.visualize_warning_distribution('warning_distribution.png')

    # 시계열 분석
    ews.visualize_time_series_analysis('timeseries_analysis.png')

    # 7. 리포트 생성
    report = ews.generate_warning_report('warning_report.csv')

    # 8. 고위험 가맹점 상세 분석 (상위 3개)
    print("🔍 고위험 가맹점 상세 분석...\n")
    top_risk_merchants = report.nlargest(3, '위험점수')['ENCODED_MCT'].values

    for idx, mct_id in enumerate(top_risk_merchants, 1):
        ews.visualize_merchant_detail(mct_id, f'merchant_detail_top{idx}.png')

    print("=" * 80)
    print("✅ 경영 위기 조기 경보 시스템 분석 완료!")
    print("=" * 80)
    print("\n생성된 파일:")
    print("  📊 warning_distribution.png - 경보 레벨 전체 현황")
    print("  📈 timeseries_analysis.png - 시계열 분석")
    print("  🔍 merchant_detail_top1~3.png - 고위험 가맹점 상세 분석")
    print("  📝 warning_report.csv - 전체 경보 리포트")
    print()


if __name__ == "__main__":
    main()
