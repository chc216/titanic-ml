from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np



# 사용특성: Parch, Age, Pclass(숫자형), Sex(범주형)
def prep_baseline():
    numbers = ["Parch", "Age", "Pclass"]
    num_imputer = SimpleImputer(strategy="median")
    num_pipeline = make_pipeline(num_imputer, StandardScaler())

    categories = ["Sex"]
    cat_imputer = SimpleImputer(strategy="most_frequent")
    sex_encoder = OneHotEncoder()
    cat_pipeline = make_pipeline(cat_imputer, sex_encoder)

    return ColumnTransformer(
        [
            ("cat", cat_pipeline, categories),
            ("num", num_pipeline, numbers),
        ]
    )


# Age -> 그룹 범주형으로 나눠서 파이프라인 적용

def bin_age(X):
    return pd.DataFrame(pd.cut( X.squeeze(), bins=[-1,20,40,100], labels=False))

def prep_v2():
    age = ["Age"]
    number = ["Pclass", "Parch"]
    cat = ["Sex"]

    age_transformer = FunctionTransformer(bin_age)
    age_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("binner", age_transformer)])

    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder())
    ])
    return ColumnTransformer([
        ("num", num_pipeline, number),
        ("cat", cat_pipeline, cat),
        ("age", age_pipeline, age)
    ])




def to_binary_cat(X):
    return pd.DataFrame(pd.cut(X.squeeze(), bins=[-1,0,14], labels=[0,1]))
# 기존 Age 변경에 더하여 Parch특성을 이진 범주로 가공하는 파이프라인
def prep_v3():
    age = ["Age"]
    parch = ["Parch"]
    number = ["Pclass"]
    cat = ["Sex"]

    age_transformer = FunctionTransformer(bin_age)
    age_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("binner", age_transformer)])

    parch_transformer = FunctionTransformer(to_binary_cat)
    parch_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", parch_transformer)
    ])
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder())
    ])
    return ColumnTransformer([
        ("num", num_pipeline, number),
        ("parch", parch_pipeline, parch),
        ("cat", cat_pipeline, cat),
        ("age", age_pipeline, age)
    ])



# 이름의 경칭을 추출하여 특성으로 활용하는 파이프라인
def extract_title(X):
    title = X.squeeze().str.extract(r",\s*([^\.]+)\.")
    common = ["Mr", "Miss", "Mrs", "Master"]
    return pd.DataFrame(title.where(title.isin(common), "Rare"))

def prep_v4():
    age = ["Age"]
    parch = ["Parch"]
    pclass = ["Pclass"]
    cat = ["Sex"]
    cat_name = ["Name"]

    name_transformer = FunctionTransformer(extract_title)
    name_pipeline = Pipeline([
        ("transformer", name_transformer),
        ("onehot", OneHotEncoder())
    ])

    age_transformer = FunctionTransformer(bin_age)
    age_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("binner", age_transformer)])

    parch_transformer = FunctionTransformer(to_binary_cat)
    parch_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", parch_transformer)
    ])
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder())
    ])
    return ColumnTransformer([
        ("num", num_pipeline, pclass),
        ("parch", parch_pipeline, parch),
        ("cat", cat_pipeline, cat),
        ("age", age_pipeline, age),
        ("name", name_pipeline, cat_name)
    ])


def prep_v4_2():
    age = ["Age"]
    parch = ["Parch"]
    pclass = ["Pclass"]
    cat = ["Sex"]
    cat_name = ["Name"]

    name_transformer = FunctionTransformer(extract_title)
    name_pipeline = Pipeline([
        ("transformer", name_transformer),
        ("onehot", OneHotEncoder())
    ])

    age_transformer = FunctionTransformer(bin_age)
    age_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler()),])

    parch_transformer = FunctionTransformer(to_binary_cat)
    parch_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", parch_transformer)
    ])
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder())
    ])
    return ColumnTransformer([
        ("num", num_pipeline, pclass),
        ("parch", parch_pipeline, parch),
        ("cat", cat_pipeline, cat),
        ("age", age_pipeline, age),
        ("name", name_pipeline, cat_name)
    ])


def extract_cabin_to_binary(X):
    filled = X.fillna('N')
    df =  pd.DataFrame(
        np.where(filled == 'N', 'N', 'Y'),
    )
    return df


def prep_v5():
    age = ["Age"]
    parch = ["Parch"]
    pclass = ["Pclass"]
    cat = ["Sex"]
    cat_name = ["Name"]
    cabin = ["Cabin"]


    ## 객실 특성 변환 과정
    # 1. 결측치는 모두 N으로 처리한다. 2. 나머지는 모두 1로 처리한다. -> 원핫 인코딩을 해본다.
    cabin_transformer = FunctionTransformer(extract_cabin_to_binary)
    cabin_pipeline = Pipeline([
        ("transformer", cabin_transformer),
        ("onehot", OneHotEncoder())
    ])

    name_transformer = FunctionTransformer(extract_title)
    name_pipeline = Pipeline([
        ("transformer", name_transformer),
        ("onehot", OneHotEncoder())
    ])

    age_transformer = FunctionTransformer(bin_age)
    age_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler()), ])

    parch_transformer = FunctionTransformer(to_binary_cat)
    parch_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("transformer", parch_transformer)
    ])
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder())
    ])
    return ColumnTransformer([
        ("num", num_pipeline, pclass),
        ("parch", parch_pipeline, parch),
        ("cat", cat_pipeline, cat),
        ("age", age_pipeline, age),
        ("name", name_pipeline, cat_name),
        ("has_cabin", cabin_pipeline, cabin)
    ])