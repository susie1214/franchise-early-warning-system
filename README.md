<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=260&color=0:00B4DB,100:0083B0&text=Franchise%20Early%20Warning%20System%20(EWS)&fontAlign=50&fontAlignY=35&fontSize=45&desc=LightGBM%20%2B%20Time-Series%20%2B%20SHAP%20%2B%20LLM&descAlign=50&descAlignY=60&animation=fadeIn" />
</p>

---
**🚨 ML 기반 경영 위기 조기 경보 시스템**

Franchise Early Warning System (EWS) using LightGBM + SHAP + LLM
팀명: tuna2000 · 인원: 2명

---
**📌 프로젝트 요약**

본 프로젝트는 5개 데이터셋을 통합한 24개월 시계열 기반 패널 데이터를 활용해
가맹점의 매출 하락 위험을 조기에 감지(Early Warning System)하는
AI 기반 경영위기 예측 모델을 구축한 연구 프로젝트입니다.

LightGBM의 다중분류 모델을 기반으로 4단계 경보 레벨(안전~위험)을 산출하며,
결과는 SHAP 해석성과 LLM(Claude)을 통해
실행 가능한 비즈니스 인사이트로 자동 변환됩니다.

📂 Table of Contents

🧩 프로젝트 개요

🛠 기술 스택

📊 데이터셋 설명

🧪 특성 엔지니어링 (60+ Features)

🎯 타겟(4단계 경보 레벨)

⚡ LightGBM 모델 학습

📈 모델 성능

🔍 SHAP 해석성

🤖 LLM 통합 분석

📊 시각화 & 리포트

🧠 고찰

📌 한계 & 개선 방향

📁 프로젝트 구조

🔚 결론

---
**🧩 프로젝트 개요**
경영 위기는 사후 대응으로는 이미 늦은 경우가 많음
→ 매출·고객·상권 정보 등 다양한 지표를 종합한 조기 감지(EWS)가 필요함.

본 모델은

5개 데이터셋 통합

60+ 비즈니스/시계열 특성

LightGBM 기반 4단계 위험 분류

Time-Series Cross Validation

월별 백테스트(18개월)

SHAP / LLM 분석 자동화

까지 완비한 현업 수준의 의사결정 지원 시스템입니다.

---
**🛠 기술 스택**
Category	Tools
ML Model	LightGBM (Multiclass)
Feature	Pandas, NumPy
Explainability	SHAP TreeExplainer
LLM	OpenAI GPT API
Visualization	Matplotlib, Seaborn
Validation	Time Series CV, Monthly Backtest
Runtime	Python 3.10+

---
**📊 데이터셋 설명**

총 5개 데이터셋을 통합하여 2023.01~2024.12 (24개월) 시계열 패널 생성.

데이터셋	내용
가맹점 정보	업종, 지역, 운영기간
매출/운영	매출액, 고객수, 이용건수, 승인율, 배달비중
고객 특성	10개 세그먼트(성별×연령), 재구매율, 신규비율
임대료	1층/비1층, 임대료 변화율
유동인구	유동·주거·직장 인구 변화율
🧪 특성 엔지니어링 (60+ Features)
🔹 1) 시계열 기반 특성

1M / 3M / 6M 변화율
3M / 6M 이동평균(MA)
3M 추세 지표
3M 변동성(Std)

최근 6M 최대 대비 비율

🔹 2) 세대별 고객 특성

2030 / 4050 / 60+ 그룹화

생성 특성:
세대별 비중
1M/3M 변화율
주력 세대
고객 다양성 지수(엔트로피)

🔹 3) 고급 특성

연속 하락 개월 수
임대료·유동인구 대비 매출 효율
동종업계 대비 상대 지표

---
**🎯 타겟(4단계 경보 레벨)**

미래 3개월 매출 변화율 기반:

레벨	기준	의미
0	> +10%	안전
1	-10%~+10%	주의
2	-30%~-10%	경고
3	≤ -30%	위험

추가 기준
고객 급감
연속 하락
→ 위험도 레벨 보정

**⚡ LightGBM 모델 학습**
🔹 모델 파라미터
params = {
    'objective': 'multiclass',
    'num_class': 4,
    'boosting_type': 'gbdt',
    'num_leaves': 40,
    'learning_rate': 0.03,
    'max_depth': 8,
    'min_child_samples': 30,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'is_unbalance': True
}

🔹 학습 전략

Time Series Split (5-Fold Rolling Window)
Early Stopping 100 rounds
Class Imbalance 보정 (is_unbalance, scale_pos_weight)

**📈 모델 성능**
🔹 교차검증 (TS-CV)

F1 = 0.86
Recall = 0.85
PR-AUC = 0.88

🔹 월별 백테스트 (2023.06 ~ 2024.12 / 18개월)

평균 F1 = 0.86 (±0.01)
→ 시간에 따라 매우 안정적

🔹 클래스별 성능
클래스	Precision	Recall	F1
안전	0.92	0.92	0.91
주의	0.85	0.84	0.84
경고	0.81	0.82	0.82
위험	0.88	0.89	0.88
평균	0.87	0.86	0.86
**🔍 SHAP 해석성**

상위 40개 변수 중요도 시각화
개별 가맹점 SHAP Force Plot 생성
멀티클래스 오류 해결:
TreeExplainer(model, model_output='raw')


주요 영향 변수:

매출 3M 변화율
고객수 3M 변화율
2030세대 비중 변화
임대료 대비 매출 비율
연속 하락 개월수

**🤖 LLM 통합 분석**
🔹 목적

위험 가맹점 Top 20에 대해
리스크 요인 자동 도출 → 개선 전략 제안

🔹 LLM 입력 정보

업종 / 지역 / 운영기간
모델 예측 위험레벨
매출·고객 변화율
세대별 고객 변화
SHAP 주요 요인

🔹 출력 예시
위험 요인:
1. 최근 3개월 매출 -28% 감소
2. 2030세대 비중 20%p 감소

권장 조치:
1. 젊은층 타겟 SNS 캠페인 진행
2. 배달 플랫폼 신규 유입 프로모션 실시

**📊 시각화 & 리포트**
포함된 시각화
경보 레벨 분포
세대별 고객 변화
시계열 추이 그래프
월별 백테스트 성능
SHAP Summary / Force Plot
Export
CSV: 전체 가맹점 위험 등급
PDF/PNG: 월별 성능 그래프
LLM 자연어 분석 리포트
<img width="874" height="493" alt="image" src="https://github.com/user-attachments/assets/4be78151-3e1e-4346-a341-f858c551b706" />
<img width="833" height="446" alt="image" src="https://github.com/user-attachments/assets/a9a6dc86-a4fb-4055-abf9-7ae4929d02a9" />
<img width="861" height="500" alt="image" src="https://github.com/user-attachments/assets/78fe8962-35fb-4b65-ac84-db6c1bd7257c" />


**🧠 고찰**
✔ 주요 학습 사항

Time Series CV 중요성

특성 엔지니어링이 성능의 핵심 (60+ 특성 생성)

SHAP + LLM → AI 예측을 실행 가능한 지식으로 변환

규칙 기반 → 100% 데이터 기반 판단 시스템 전환

✔ 기술적 도전 해결

클래스 불균형 → is_unbalance

SHAP Multiclass 오류 → model_output=raw

시계열 누수 방지 → 시간 순서 분할

18개월 백테스트 안정성 확보(F1±1%)

📌 한계 & 개선 방향
현재 한계	향후 개선안
월 단위 예측	주·일 단위 세밀화
정적 과거 데이터	실시간 데이터 스트리밍
수동 보고서	자동 알림 및 리포트 자동화
단일 LightGBM 모델	XGBoost/CatBoost 앙상블
```
📁 프로젝트 구조
EWS-Project/
│
├── data/
│   └── merged_dataset.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_train.ipynb
│   ├── 04_shap_analysis.ipynb
│   └── 05_backtest.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── model.py
│   ├── shap_utils.py
│   ├── llm_analysis.py
│   └── backtest.py
│
├── results/
│   ├── shap_plots/
│   ├── backtest/
│   └── final_report.csv
│
└── README.md
```
🔚 결론

본 프로젝트는

정확한 위험 탐지(86% F1)

시계열 기반 백테스트 검증

SHAP 해석성

LLM 기반 자동 분석

을 결합하여 실제 비즈니스 현장에서 바로 활용 가능한 EWS 시스템을 구축했습니다.

데이터 기반 의사결정을 통해
가맹점의 매출 하락을 “사후 대응” → “사전 예방”으로 전환할 수 있는
실질적 솔루션을 제시합니다.
