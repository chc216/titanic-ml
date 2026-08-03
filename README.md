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

| 버전                       | 사용 피처                  | CV(k-fold) | LB |
|--------------------------|------------------------|------------|----|
| v1 - logistic regression | Age, Parch, Pclass, Sex | 0.7893     | -  |
| v1 - random forest       | 위와 동일                  | 0.7893     | -  |
| v2 - logistic regression | v1에서 Age를 범주형으로 변경     | 0.8019     | -  |
| v2 - random forest       | 위와 동일                  | 0.8047     | -  |
|                          |                        |            |    |

