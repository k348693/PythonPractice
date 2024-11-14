import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboo

# Load the churn dataset
st.header("CSV Read")
# df = pd.read_csv("credit_card_churn.csv") 

df = xgboo.load_df()   # 전처리 완료된 데이터 
st.dataframe(df)


# Streamlit app
st.title("Churn Prediction for Credit Card")

# Sidebar for user input
#############


# Sidebar for user input
st.sidebar.header("고객 정보")

def user_input_features():
    age = st.sidebar.slider('customer_age', min_value=int(df['customer_age'].min()), max_value=int(df['customer_age'].max()), value=int(df['customer_age'].mean()))
    gender = st.sidebar.selectbox('gender', options=[0, 1],
                                  format_func=lambda x: 'Male' if x == 1 else 'Female')
    dependents = st.sidebar.selectbox('dependent_count', options=list(range(6)))

    education_level_options = {
        0: 'Uneducated',
        1: 'Graduate',
        2: 'High_school',
        3: 'College',
        4: 'Post_Graduate',
        5: 'Doctorate'
    }
    education_level = st.sidebar.selectbox('education_level', options=list(education_level_options.keys()),
                                           format_func=lambda x: education_level_options[x])
    marital_status_options = {
        0: 'Married',
        1: 'Single',
        2: 'Divorced'
    }
    marital_status = st.sidebar.selectbox('marital_status', options=list(marital_status_options.keys()),
                                          format_func=lambda x: marital_status_options[x])
    income_category_options = {
        0: 'Less_than_$40K',
        1: '$40K_$60K',
        2: '$60K_$80K',
        3: '$80K_$120K',
        4: 'Over_$120K'
    }
    income_category = st.sidebar.selectbox('income_category', options=list(income_category_options.keys()),
                                           format_func=lambda x: income_category_options[x])
    card_category_options = {
        0: 'Blue',
        1: 'Silver',
        2: 'Gold',
        3: 'Platinum',
    }
    card_category = st.sidebar.selectbox('card_category', options=list(card_category_options.keys()),
                                         format_func=lambda x: card_category_options[x])



# Collect remaining numerical inputs
    months_on_book = st.sidebar.slider('months_on_book', int(df['months_on_book'].min()), int(df['months_on_book'].max()), int(df['months_on_book'].mean()))
    total_relationship_count = st.sidebar.slider('total_relationship_count', int(df['total_relationship_count'].min()), int(df['total_relationship_count'].max()), int(df['total_relationship_count'].mean()))
    months_inactive_12_mon = st.sidebar.slider('months_inactive_12_mon', int(df['months_inactive_12_mon'].min()), int(df['months_inactive_12_mon'].max()), int(df['months_inactive_12_mon'].mean()))
    contacts_count_12_mon = st.sidebar.slider('contacts_count_12_mon', int(df['contacts_count_12_mon'].min()), int(df['contacts_count_12_mon'].max()), int(df['contacts_count_12_mon'].mean()))
    credit_limit = st.sidebar.slider('credit_limit', float(df['credit_limit'].min()), float(df['credit_limit'].max()), float(df['credit_limit'].mean()))
    total_revolving_bal = st.sidebar.slider('total_revolving_bal', int(df['total_revolving_bal'].min()), int(df['total_revolving_bal'].max()), int(df['total_revolving_bal'].mean()))
    avg_open_to_buy = st.sidebar.slider('avg_open_to_buy', int(df['avg_open_to_buy'].min()), int(df['avg_open_to_buy'].max()), int(df['avg_open_to_buy'].mean()))
    total_amt_chng_q4_q1 = st.sidebar.slider('total_amt_chng_q4_q1', float(df['total_amt_chng_q4_q1'].min()), float(df['total_amt_chng_q4_q1'].max()), float(df['total_amt_chng_q4_q1'].mean()))
    total_trans_amt = st.sidebar.slider('total_trans_amt', int(df['total_trans_amt'].min()), int(df['total_trans_amt'].max()), int(df['total_trans_amt'].mean()))
    total_trans_ct = st.sidebar.slider('total_trans_ct', int(df['total_trans_ct'].min()), int(df['total_trans_ct'].max()), int(df['total_trans_ct'].mean()))
    total_ct_chng_q4_q1 = st.sidebar.slider('total_ct_chng_q4_q1', float(df['total_ct_chng_q4_q1'].min()), float(df['total_ct_chng_q4_q1'].max()), float(df['total_ct_chng_q4_q1'].mean()))
    avg_utilization_ratio = st.sidebar.slider('avg_utilization_ratio', float(df['avg_utilization_ratio'].min()), float(df['avg_utilization_ratio'].max()), float(df['avg_utilization_ratio'].mean()))

    data = {
        'customer_age': age,
        'gender': gender,
        'dependent_count': dependents,
        'education_level': education_level,
        'marital_status': marital_status,       ####
        'income_category': income_category,
        'card_category': card_category,         ####
        'months_on_book': months_on_book,
        'total_relationship_count': total_relationship_count,
        'months_inactive_12_mon': months_inactive_12_mon,
        'contacts_count_12_mon': contacts_count_12_mon,
        'credit_limit': credit_limit,
        'total_revolving_bal': total_revolving_bal,
        'avg_open_to_buy': avg_open_to_buy,
        'total_amt_chng_q4_q1': total_amt_chng_q4_q1,
        'total_trans_amt': total_trans_amt,
        'total_trans_ct': total_trans_ct,
        'total_ct_chng_q4_q1': total_ct_chng_q4_q1,
        'avg_utilization_ratio': avg_utilization_ratio
    }
    # features = pd.DataFrame(data, index=[0])
    return data

data_change = user_input_features()

input_df = df.loc[[0]].drop(columns=['churn'])

for column, value in data_change.items():
    input_df.at[0, column] = value
    # input_df.drop[
# customer_age,gender,dependent_count,education_level,months_on_book,total_relationship_count,months_inactive_12_mon,contacts_count_12_mon,credit_limit,total_revolving_bal,avg_open_to_buy,total_amt_chng_q4_q1,total_trans_amt,total_trans_ct,total_ct_chng_q4_q1,avg_utilization_ratio,marital_status_Married,marital_status_Single,card_category_Gold,card_category_Platinum,card_category_Silver,income_category_$40K - $60K,income_category_$60K - $80K,income_category_$80K - $120K,income_category_Less than $40K


st.dataframe(input_df)


################################## One Hot ##############################
# ohe_encoder = OneHotEncoder(drop='first', sparse_output=False)

# columns_to_ohe_encode = [ 'marital_status', 'card_category']
# encoded_data = ohe_encoder.fit_transform(input_df[columns_to_ohe_encode])

# # 인코딩된 데이터를 dataframe으로 변환
# encoded_df = pd.DataFrame(encoded_data, columns=ohe_encoder.get_feature_names_out(columns_to_ohe_encode))
# # 기존 df와 인코딩된 dataframe을 병합하고 원본 열 삭제
# input_df = pd.concat([input_df.drop(columns=columns_to_ohe_encode), encoded_df], axis=1)



###############################################################
input_df2 = df.loc[[39]].drop(columns=['churn'])
# st.dataframe(input_df)


# Display user input
# st.subheader("Customer Input Information")
# st.write(input_df)

# Splitting data for training and testing
X = df.drop(columns=['churn'])
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model Training
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#####################################################################################################


performance_metrics = xgboo.train_and_evaluate_xgb(X_train, y_train, X_test, y_test)
st.write("Model Performance Metrics:")

results = xgboo.train_and_evaluate_xgb(X_train, y_train, X_test, y_test)
for metric, value in results.items():
    if metric in ["model", "y_pred_proba_xgb"]:
        continue  # 모델 객체와 확률 예측은 제외하고 성능 지표만 출력
    # st.write(f"{metric.capitalize()}: {value:.4f}")


# st.write("Predicted Churn Probabilities for Test Set:")
# st.write(results["y_pred_proba_xgb"])

st.dataframe(input_df2)
st.header("이탈률 예측")
y_pred_proba = results["model"].predict_proba(input_df2)[:, 1]
st.write(y_pred_proba)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Prediction
# input_scaled = scaler.transform(input_df)
# prediction = model.predict(input_scaled)
# prediction_proba = model.predict_proba(input_scaled)

# # Display Prediction
# st.subheader("Prediction")
# if prediction[0] == 1:
#     st.write("This customer is likely to churn.")
# else:
#     st.write("This customer is not likely to churn.")

# # Display Prediction Probability
# st.subheader("Prediction Probability")
# st.write(f"Probability of Not Churning: {prediction_proba[0][0]:.2f}")
# st.write(f"Probability of Churning: {prediction_proba[0][1]:.2f}")

# # Model Accuracy
# st.subheader("Model Accuracy")
# y_pred = model.predict(X_test_scaled)
# st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
