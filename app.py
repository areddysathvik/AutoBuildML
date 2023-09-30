import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline_code import Pipeline


st.title("**Machine Learning Pipeline**")
st.write('**Build Your ML model in minutes**')
st.success('📊 Only CSV data format is supported for now 📊')

st.header("Upload a Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
st.set_option('deprecation.showPyplotGlobalUse', False)

if uploaded_file is not None:
    # Load the dataset
    df_ = pd.read_csv(uploaded_file)
    df = df_.copy()

    # Display dataset info
    is_fitted = False  # Initialize is_fitted to False

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    target_column = st.sidebar.selectbox("Select the target column", df_.columns)
    k_features = st.sidebar.slider("Select the number of features (K_features)", min_value=1, max_value=len(df_.columns) - 1, value=5)
    classification = st.sidebar.checkbox("Classification Task", value=True)
    
    # Display dataset details
    st.markdown('<hr>', unsafe_allow_html=True)
    st.header("DataSet Details")
    st.write("Sample Data")
    st.write(df_.head())
    st.write('Statistics')
    st.write(df_.describe())

    if target_column and k_features:
        if classification:
            pipeline = Pipeline(df_, target_column, k_features, classification=True)
            button_text = "Run Classification"
            is_fitted = True  
        else:
            pipeline = Pipeline(df_, target_column, k_features, classification=False)
            is_fitted = True  
            button_text = "Run Regression"
            
        if st.sidebar.button(button_text):
            # Display results DataFrame
            st.header("Results DataFrame")
            st.write(pipeline.results_df)

            # Display selected features
            st.header("Selected Features")
            st.write(pipeline.selected_features.tolist())

            # Plot feature importance (for classification)
            if classification:
                st.header("Feature Importance Plot")
                plt.figure(figsize=(10, 6))
                sns.barplot(x=pipeline.results_df['Model'], y=pipeline.results_df['Precision'])
                plt.title("Feature Importance")
                plt.xticks(rotation=45)
                st.pyplot()

    if is_fitted:
        st.sidebar.header("Test Custom Record")
        st.markdown('<hr>', unsafe_allow_html=True)
        inputs = {}
        for column_name, dtype_ in zip(df.columns, df.dtypes):
            if column_name != target_column:
                if dtype_ == 'int64' or dtype_ == 'int32':
                    inputs[column_name] = st.sidebar.number_input(f"Enter {column_name}", value=0)
                elif dtype_ == 'bool':
                    inputs[column_name] = st.sidebar.checkbox(f"Select {column_name}", value=False)
                elif dtype_ == 'float32' or dtype_ == 'float64':
                    inputs[column_name] = st.sidebar.number_input(f"Enter {column_name}", value=0.0, step=0.01)
                else:
                    if len(df[column_name].unique()) <= 10:
                        inputs[column_name] = st.sidebar.selectbox(f"Select {column_name}", df[column_name].unique())
                    else:
                        inputs[column_name] = st.sidebar.text_input(f"Enter {column_name}", "")

        if st.sidebar.button("Predict Custom Record"):
            custom_record_df = pd.DataFrame([inputs])
            custom_predictions, target_encoder = pipeline.custom_Record_Prediction(custom_record_df)

            # Display custom record predictions
            st.header("Custom Record Predictions")
            st.write(custom_predictions)

            # Display the mapping of classes to numbers
            if target_encoder is not None:
                mapping_df = pd.DataFrame({'Class': target_encoder.classes_, 'Number': target_encoder.transform(target_encoder.classes_)})
                st.header("Mapping of Classes to Numbers")
                st.dataframe(mapping_df)

            # Plot feature importance (for classification)
            if classification:
                st.header("Feature Importance Plot (Custom Record)")
                plt.figure(figsize=(10, 6))
                sns.barplot(x=pipeline.results_df['Model'], y=pipeline.results_df['Precision'])
                plt.title("Feature Importance")
                plt.xticks(rotation=45)
                st.pyplot()