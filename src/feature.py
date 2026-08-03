from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd



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


def bin_age(X):
    return pd.DataFrame(pd.cut( X.squeeze(), bins=[-1,20,40,100], labels=[0,1,2]))

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
