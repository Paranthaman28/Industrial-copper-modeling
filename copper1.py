import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle

df = pd.read_excel(r"C:\Users\Paranthaman\Downloads\Copper_Set.xlsx")
st.set_page_config(
    page_title='CarDheko price pridection',
    layout='wide'
)
#dasboard title
st.title('CarDheko Price Prediction')
select=option_menu("main menu",options=['Home','Regression prediction','classification prediction'],
                   icons=['house','pencil-square','phone'],
                   styles={'container':{'padding':'10!important','width':'100'},"icon": {"color": "black", "font-size": "20px"}})
df['item_year'] = pd.DatetimeIndex(df['item_date']).year
df['item_month'] = pd.DatetimeIndex(df['item_date']).month
df['deliver_year'] = pd.DatetimeIndex(df['delivery date']).year
df['deliver_month'] = pd.DatetimeIndex(df['delivery date']).month
df = df.drop(['id','item_date', 'delivery date'], axis=1)
df=df.ffill()
# Input fields
if select=="Regression prediction":
    col1,col2=st.columns(2)
    with col1:
        
        quantity = st.number_input("Quantity Tons", min_value=0.0)
        customer = st.selectbox("Customer", sorted(df['customer'].unique()))
        country = st.selectbox("Country", sorted(df['country'].unique()))
        status = st.selectbox("Status", df['status'].unique())
        item_type = st.selectbox("Item Type", df['item type'])
        material_ref = st.selectbox("Material Ref", df['material_ref'].unique())
    
    with col2:

        application = st.selectbox("Application", sorted(df['application'].unique()))
        thickness = st.number_input("Thickness", min_value=0.0)
        width = st.number_input("Width", min_value=0.0)
        product_ref = st.selectbox("Product Ref", sorted(df['product_ref'].unique()))
        item_year = st.number_input("Item Year", min_value=2000, max_value=2100)
        item_month = st.number_input("Item Month", min_value=1, max_value=12)
        deliver_year = st.number_input("Delivery Year", min_value=2000, max_value=2100)
        deliver_month = st.number_input("Delivery Month", min_value=1, max_value=12)

    def enco(df):
        df = df.dropna(subset=['selling_price'])

        # Encode categorical columns
        encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df,encoders
    df1, encoders = enco(df)
    # Save encoders
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    # Prepare data for training
    target = 'selling_price'
    X = df1.drop(columns=target)
    y = df1[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Save model
    with open("reg_model.pkl", "wb") as f:
        pickle.dump(model, f)


    if st.button("Predict Selling Price"):
            input_dict = {
                'quantity tons': [quantity],
                'country': [country],
                'customer':[customer],
                'status': [encoders['status'].transform([status])[0]],
                'item type': [encoders['item type'].transform([item_type])[0]],
                'material_ref': [encoders['material_ref'].transform([material_ref])[0]],
                'application': [application],
                'thickness': [thickness],
                'width': [width],
                'product_ref': [product_ref],
                'item_year': [item_year],
                'item_month': [item_month],
                'deliver_year': [deliver_year],
                'deliver_month': [deliver_month]
    }
            # Load encoders and scaler
            with open("encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("reg_model.pkl", "rb") as f:
                model = pickle.load(f)
            st.write(encoders)

            input_df = pd.DataFrame(input_dict)

            # Apply the same label encoding
            def encode(input_df):
                for col in input_df.select_dtypes(include='object').columns:
                    if col in encoders:
                        le = encoders[col]
                        input_df[col] = le.transform(input_df[col].astype(str))
                    else:
                        st.error(f"Encoder for {col} not found.")
                        st.stop()
                return input_df
            input_df1=input_df
            feature_names = X.columns.tolist()

            # Save the feature names used
            import pickle
            with open("features.pkl", "wb") as f:
                pickle.dump(feature_names, f)
            
            with open("features.pkl", "rb") as f:
                expected_columns = pickle.load(f)

            input_df2 = input_df1[expected_columns]

            # Scale the input features
            input_scaled = scaler.transform(input_df2)

            # Predict the selling price
            predicted_price = model.predict(input_scaled)
            st.success(f"Predicted Selling Price: ₹ {predicted_price.item():,.2f}")
if select=='classification prediction':
    quantity = st.number_input("Quantity Tons", min_value=0.0)
    customer = st.selectbox("Customer", sorted(df['customer'].unique()))
    country = st.selectbox("Country", sorted(df['country'].unique()))
    selling_price = st.selectbox("Status", df['selling_price'].unique())
    item_type = st.selectbox("Item Type", df['item type'])
    material_ref = st.selectbox("Material Ref", df['material_ref'].unique())
    application = st.selectbox("Application", sorted(df['application'].unique()))
    thickness = st.number_input("Thickness", min_value=0.0)
    width = st.number_input("Width", min_value=0.0)
    product_ref = st.selectbox("Product Ref", sorted(df['product_ref'].unique()))
    item_year = st.number_input("Item Year", min_value=2000, max_value=2100)
    item_month = st.number_input("Item Month", min_value=1, max_value=12)
    deliver_year = st.number_input("Delivery Year", min_value=2000, max_value=2100)
    deliver_month = st.number_input("Delivery Month", min_value=1, max_value=12)

    def enco(df):
        df = df.dropna(subset=['status'])

        # Encode categorical columns
        encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df,encoders
    df1, encoders = enco(df)
    # Save encoders
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    # Prepare data for training
    target = 'status'
    X = df1.drop(columns=target)
    y = df1[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Save scaler
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    model1=RandomForestClassifier()
    model1.fit(X_scaled, y)

        # Save regression model
    with open("class_model.pkl", "wb") as f:
        pickle.dump(model1, f)

    if st.button("Predict Status"):
            input_dict = {
                'quantity tons': [quantity],
                'country': [country],
                'customer':[customer],
                'selling_price':[selling_price],
                'item type': [encoders['item type'].transform([item_type])[0]],
                'material_ref': [encoders['material_ref'].transform([material_ref])[0]],
                'application': [application],
                'thickness': [thickness],
                'width': [width],
                'product_ref': [product_ref],
                'item_year': [item_year],
                'item_month': [item_month],
                'deliver_year': [deliver_year],
                'deliver_month': [deliver_month]
    }
            # Load encoders and scaler
            with open("encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open("class_model.pkl","rb") as f:
                class_moidel=pickle.load(f)
            st.write(encoders)

            input_df = pd.DataFrame(input_dict)

            # Apply the same label encoding
            def encode(input_df):
                for col in input_df.select_dtypes(include='object').columns:
                    if col in encoders:
                        le = encoders[col]
                        input_df[col] = le.transform(input_df[col].astype(str))
                    else:
                        st.error(f"Encoder for {col} not found.")
                        st.stop()
                return input_df
            input_df1=input_df
            feature_names = X.columns.tolist()

            # Save the feature names used
            import pickle
            with open("features.pkl", "wb") as f:
                pickle.dump(feature_names, f)
            
            with open("features.pkl", "rb") as f:
                expected_columns = pickle.load(f)

            input_df2 = input_df1[expected_columns]

            # Scale the input features
            input_scaled = scaler.transform(input_df2)

            # Predict the selling price
            predicted_status = class_moidel.predict(input_scaled)
            st.success(f"Predicted Status: {predicted_status[0]}")
            decoded_status = encoders['status'].inverse_transform(predicted_status)
            st.success(f"Predicted Status: {decoded_status[0]}")
