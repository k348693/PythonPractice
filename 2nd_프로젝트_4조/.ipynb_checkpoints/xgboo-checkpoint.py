import pandas as pd
import numpy as np
import pandas

from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import numpy as np

# 1. 데이터 가져오기
def load_dataset():
    # 데이터 load
    data = pd.read_csv("data/credit_card_churn.csv", na_values='Unknown')
    # 컬럼명 소문자로 변경
    data.columns = data.columns.str.lower()
    data.rename(columns={'attrition_flag': 'churn'}, inplace=True)
    
    ## 불필요 칼럼 삭제
    data.drop(
        columns=[
            'clientnum',
            'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_1',
            'naive_bayes_classifier_attrition_flag_card_category_contacts_count_12_mon_dependent_count_education_level_months_inactive_12_mon_2'
        ], 
        inplace=True
    )
    return data

df = load_dataset()

# 2. 데이터 전처리 (이상치, 결측치, Feature Engineering)

# 2-1. 데이터 전처리 > 이상치(Outlier) 
# - IQR 식별 -> 극단치 제거
#  2-1-1. Outlier 식별: IQR(Inter quantile Range) 을 이용해 Outlier 식별 
def find_outliers(df, column_name, whis=1.5):
    """
    분위수 기준으로 이상치를 찾는 함수

    Parameters:
    df (pd.DataFrame): 데이터프레임
    column_name (str): 이상치를 찾을 컬럼명

    Returns:
    pd.Series: 이상치 값들
    """
    q1, q3 = df[column_name].quantile(q=[0.25, 0.75])
    iqr = q3 - q1
    iqr *= whis
    return df.loc[~df[column_name].between(q1 - iqr, q3 + iqr)]

# ==> ["customer_age", "total_trans_ct"] 두 칼럼의 이상치를 삭제하기로 결정

#  2-1-2. Outlier 제거: 삭제할 이상치 index를 찾아서 drop
def delete_outliers(df, columns, whis=1.5):
    index_list = []
    _df = df.copy()
    
    for col in columns: 
        outliers_column_index = find_outliers(df, col, whis=whis)
        index_list.extend(outliers_column_index.index)
        
        
    _df = _df.drop(index=index_list)
        
    _df.reset_index(drop=True, inplace=True)
    
    return _df

outlier_columns = ["customer_age", "total_trans_ct"]
df = delete_outliers(df, outlier_columns)

# 2-2. 데이터 전처리 > 결측치 대체(imputation) 
# - SimpleImputer: 최빈값으로 대치(column 2개) + 사용자 정의 imputer: 비율에 따른 대치(column 1개)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

#  2-2-1. SimpleImputer(strategy=최빈값) 
simple_imputer = SimpleImputer(strategy='most_frequent')

#  2-2-2. ProportionalImputer(사용자 정의)
class ProportionalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.fill_values = {}

    def fit(self, X, y=None):
        for column in self.columns:
            value_counts = X[column].value_counts(normalize=True)
            self.fill_values[column] = (value_counts.index, value_counts.values)
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:

            nan_count = X[column].isna().sum()
            if nan_count > 0:
                fill_values = np.random.choice(
                    self.fill_values[column][0], size=nan_count, p=self.fill_values[column][1]
                )
                X.loc[X[column].isna(), column] = fill_values
        return X

def simple_impute_most_frequent(df, columns):
    imputer = SimpleImputer(strategy='most_frequent')
    df[columns] = imputer.fit_transform(df[columns])
    return df
def proportional_impute(df, columns):
    imputer = ProportionalImputer(columns=columns)
    imputer.fit(df)
    return imputer.transform(df)

df = simple_impute_most_frequent(df, ['education_level', 'marital_status'])
df = proportional_impute(df, ['income_category'])

# 2-3. 데이터 전처리 > Feature Engineering
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 2-3-1. 라벨 인코딩(Label Encoding) - 'gender'
# 이유: 이진 변수의 경우 모델 성능에 큰 차이가 없으므로, 간단히 라벨 인코딩을 사용하기로 함.
label_encoder = LabelEncoder()

df['gender'] = label_encoder.fit_transform(df['gender'])

# 2-3-2. 순서 인코딩 (Ordinal Encoding) - 'education_level'
education_order = {"Uneducated": 0, "High School": 1, "College": 2, "Graduate": 3, "Post-Graduate": 4, "Doctorate": 5}
df['education_level'] = df['education_level'].map(education_order)

# 2-3-3. mapping - 'gender'
df['churn'] = df['churn'].map({"Existing Customer": 0, "Attrited Customer": 1})

# 2-3-4. 원핫 인코딩(One-Hot encoding) - 'marital_status', 'card_category', 'income_category'
# 이유: 순서가 없고 각 값이 독립적인 범주형 데이터으로서 순서나 크기 정보 없이 각각 독립적인 특성으로 변환되므로, 머신러닝 모델에서 더 잘 해석될 가능성이 있다고 보아 원핫 인코딩 하기로 결정.
ohe_encoder = OneHotEncoder(drop='first', sparse_output=False)

columns_to_ohe_encode = [ 'marital_status', 'card_category', 'income_category']
encoded_data = ohe_encoder.fit_transform(df[columns_to_ohe_encode])

# 인코딩된 데이터를 dataframe으로 변환
encoded_df = pd.DataFrame(encoded_data, columns=ohe_encoder.get_feature_names_out(columns_to_ohe_encode))
# 기존 df와 인코딩된 dataframe을 병합하고 원본 열 삭제
df = pd.concat([df.drop(columns=columns_to_ohe_encode), encoded_df], axis=1)

## 데이터 스플릿

from sklearn.model_selection import train_test_split
X = df.drop(["churn"], axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


# # XGBoostClassifier 모델 생성
# xgb = XGBClassifier()

# xgb.fit(X_train, y_train)

# y_pred_xgb = xgb.predict(X_test)
# y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# # 성능 지표 계산
# accuracy = accuracy_score(y_test, y_pred_xgb)
# auc = roc_auc_score(y_test, y_pred_proba_xgb)
# precision = precision_score(y_test, y_pred_xgb)
# recall = recall_score(y_test, y_pred_xgb)
# f1 = f1_score(y_test, y_pred_xgb)

# print(f"XGBoost - Accuracy: {accuracy:.4f}, recall: {recall:.4f} ,AUC: {auc:.4f}, Precision: {precision:.4f}, F1 Score: {f1:.4f}")



def train_and_evaluate_xgb(X_train, y_train, X_test, y_test):
    # XGBClassifier 모델 정의 및 학습
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    
    # 예측 및 예측 확률
    y_pred_xgb = xgb.predict(X_test)
    y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    # 성능 지표 계산
    accuracy = accuracy_score(y_test, y_pred_xgb)
    auc = roc_auc_score(y_test, y_pred_proba_xgb)
    precision = precision_score(y_test, y_pred_xgb)
    recall = recall_score(y_test, y_pred_xgb)
    f1 = f1_score(y_test, y_pred_xgb)
    
    # 성능 지표 반환
    return {
        "model": xgb,
        "y_pred_proba_xgb": y_pred_proba_xgb,
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def load_df():
    return df





