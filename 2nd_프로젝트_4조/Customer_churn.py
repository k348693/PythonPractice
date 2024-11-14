import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.markdown('''
Customer churn built to analyse Churn in Companies.

**Credits**
- App built by [K. Abhi Ram Reddy](https://www.linkedin.com/in/k-abhi-ram-reddy-561742332/) 
''')

# Load the customer churn dataset
data = pd.read_csv("credit_card_churn.csv")

# Select relevant features for the model
df = data[["Attrition_Flag","Customer_Age","Gender","Dependent_count","Education_Level","Marital_Status","Income_Category","Card_Category","Months_on_book","Total_Relationship_Count","Months_Inactive_12_mon","Contacts_Count_12_mon","Credit_Limit","Total_Revolving_Bal","Avg_Open_To_Buy","Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Trans_Ct","Total_Ct_Chng_Q4_Q1","Avg_Utilization_Ratio"
]]

# Streamlit app
st.title("Churn Prediction for Credit Card")

# Sidebar for user input
st.sidebar.header("Customer Information")


def user_input_features():
    age = st.sidebar.slider('Age', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()),
                            value=int(df['Age'].mean()))
    total_purchase = st.sidebar.slider('Total Purchase ($)', min_value=float(df['Total_Purchase'].min()),
                                       max_value=float(df['Total_Purchase'].max()),
                                       value=float(df['Total_Purchase'].mean()))
    account_manager = st.sidebar.selectbox('Account Manager', options=[0, 1],
                                           format_func=lambda x: 'Yes' if x == 1 else 'No')
    years = st.sidebar.slider('Years with the company', min_value=float(df['Years'].min()),
                              max_value=float(df['Years'].max()), value=float(df['Years'].mean()))
    num_sites = st.sidebar.slider('Number of sites', min_value=int(df['Num_Sites'].min()),
                                  max_value=int(df['Num_Sites'].max()), value=int(df['Num_Sites'].mean()))

    data = {
        'Age': age,
        'Total_Purchase': total_purchase,
        'Account_Manager': account_manager,
        'Years': years,
        'Num_Sites': num_sites
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Display user input
st.subheader("Customer Input Information")
st.write(input_df)

# Splitting data for training and testing
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Prediction
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display Prediction
st.subheader("Prediction")
if prediction[0] == 1:
    st.write("This customer is likely to churn.")
else:
    st.write("This customer is not likely to churn.")

# Display Prediction Probability
st.subheader("Prediction Probability")
st.write(f"Probability of Not Churning: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Churning: {prediction_proba[0][1]:.2f}")

# Model Accuracy
st.subheader("Model Accuracy")
y_pred = model.predict(X_test_scaled)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
