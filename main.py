import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)


# 데이터 불러오기
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(train.head())
print(train.info())

# 사용할 변수 선택
features = ["Sex", "Pclass", "Fare", "Age"]

X = train[features].copy()
y = train["Survived"]

X_test = test[features].copy()


# 데이터 확인
print(X.head())
print(X.describe())
X.info()

f, axes = plt.subplots(2,1, sharey=True)

X["Age"].hist(ax=axes[0])



# Fare 결측치 처리
fare_median = X["Fare"].median()
age_mode = X["Age"].mode()[0]

X["Fare"] = X["Fare"].fillna(fare_median)
X["Age"] = X["Age"].fillna(age_mode)
X_test["Fare"] = X_test["Fare"].fillna(fare_median)
X_test["Age"] = X_test["Age"].fillna(age_mode)


X["Age"].hist(ax=axes[1])

plt.show()
# Sex 열 인코딩
X = pd.get_dummies(
    X,
    columns=["Sex"],
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=["Sex"],
    drop_first=True
)

print(X.head())
print(X_test.head())


# 학습 데이터와 검증 데이터 분리
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 검증용 모델 학습
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)


# 검증 데이터 예측
valid_pred = model.predict(X_valid)
print("Validation accuracy:", accuracy_score(y_valid, valid_pred))


# 전체 train 데이터로 최종 모델 다시 학습
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X, y)


# 실제 test.csv 예측
test_pred = final_model.predict(X_test)

print("test 데이터 행 개수:", len(test))
print("예측 결과 개수:", len(test_pred))


# 제출 파일 생성
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})

print(submission.head())
print(submission.shape)
print(submission.isna().sum())


submission.to_csv("submission.csv", index=False)
