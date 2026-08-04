# kaggle titanic classification
## 문제 정의
주어진 데이터셋을 가지고 학습한 뒤 승객의 정보를 보고 생존 여부를 예측하기

* 특이사항
    * 핸즈온 머신러닝 2장의 프로세스를 따라가며 적용하기

## 성능 기록
### 실험 1

* 사용 피쳐
  * sex, pclass, fare
* 사용 모델
  * 로지스틱 회귀
* 특이사항
  * 결측치 처리만 함
  * 책 읽기 전에 간단하게 맛만 본 버전
* 성능
  * CV (5-fold): -
  * Holdout: -
  * Kaggle LB: 0.76555


### 실험 2
* 특이사항
  * 핸즈온 머신러닝 2장 적용

| 버전                       | 사용 피처                   | CV(k-fold) | LB |
|--------------------------|-------------------------|------------|----|
| v1 - logistic regression | Age, Parch, Pclass, Sex | 0.7893     | -  |
| v1 - random forest       | 위와 동일                   | 0.7893     | -  |
| v2 - logistic regression | v1에서 Age를 범주형으로 변경      | 0.8019     | -  |
| v2 - random forest       | 위와 동일                   | 0.8047     | -  |
| v3 - logistic regression | v2에서 Parch를 범주형으로 변경    | 0.7977     |    |
| v3 - random forest | 위와 동일                   | 0.8005| |
| v4 - logistic regression | v3에서 이름 경칭 특성 추가        | 0.8159 ||
| v4 - random forest | 위와 동일                   | 0.7864 ||
| v4-2 - randomforest | Age를 숫자형으로 사용           | 0.7879 ||
| v5 - logistic regression | has_cabin 이진 범주형 추가     | 0.8229 ||
| v5 - random forest | 위와 동일                   |0.7879 ||
|                          |                         |            |    |


