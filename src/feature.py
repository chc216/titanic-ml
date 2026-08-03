from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


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
