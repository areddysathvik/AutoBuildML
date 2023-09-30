import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score

class Pipeline:
    def __init__(self, dataframe, Target, K_features, classification):
        self.df = dataframe
        self.Target = Target
        self.k = K_features
        self.classification = classification
        self.identify_Cols()
    
    def identify_Cols(self):
        # Identify categorical and continuous columns
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns
        self.continuos_cols = self.df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        
        self.handle_NA()
    
    def handle_NA(self):
        # Handle missing values
        self.need_to_fill_na_cols = self.df.columns[self.df.isna().sum() > self.df.shape[0] / 10]
        
        cat_na = self.df[self.need_to_fill_na_cols].select_dtypes(include=['category', 'object']).columns
        cont_na = self.df[self.need_to_fill_na_cols].select_dtypes(include=['int64', 'int32', 'float32', 'float64']).columns

        # Fill continuous columns with mean
        for col in cont_na:
            self.df[col] = self.df[col].fillna(self.df[col].mean())

        # Fill categorical columns with mode
        for col in cat_na:
            self.df[col] = self.df[col].fillna(self.df[col].mode())
        
        # Drop remaining rows with missing values
        self.df.dropna(inplace=True)
        
        self.handle_Duplicated()
    
    def handle_Duplicated(self):
        # Handle duplicated rows
        self.df.drop_duplicates(inplace=True)
        
        self.scale_Numerical()
    
    def scale_Numerical(self):
        # Standardize numerical columns
        self.scaler = StandardScaler()
        
        self.continuos_cols_without_y = self.continuos_cols.to_list()
        
        if self.Target in self.continuos_cols_without_y:
            self.continuos_cols_without_y.remove(self.Target)

        self.cat_cols_without_y = self.categorical_cols.to_list()
        
        if self.Target in self.cat_cols_without_y:
            self.cat_cols_without_y.remove(self.Target)

        self.df[self.continuos_cols_without_y] = self.scaler.fit_transform(self.df[self.continuos_cols_without_y])
    
        self.encode_Categorical()
    
    def encode_Categorical(self):
        # Encode categorical columns
        self.label_encoders = {}  
        if self.classification:
            self.Target_encoder = LabelEncoder()
            self.df[self.Target] = self.Target_encoder.fit_transform(self.df[self.Target])

        for col in self.cat_cols_without_y:
            label_encoder = LabelEncoder()
            self.df[col] = label_encoder.fit_transform(self.df[col])

            self.label_encoders[col] = label_encoder

        self.select_Best_Features()
    
    def select_Best_Features(self):
        # Select K best features using ANOVA F-value
        X = self.df.drop(self.Target, axis=1)
        self.y = self.df[self.Target]
        self.y = self.y.astype(int)
        
        k_best = SelectKBest(score_func=f_classif, k=self.k) 

        X_new = k_best.fit_transform(X, self.y)

        self.selected_feature_indices = k_best.get_support(indices=True)

        self.selected_features = X.columns[self.selected_feature_indices]
        
        self.df_To_Train = X.iloc[:, self.selected_feature_indices]
        self.prepare_TrainTest_Sets()
    
    def prepare_TrainTest_Sets(self):
        # Split dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                self.df_To_Train, self.y, test_size=0.4, random_state=1042)
        
        self.fit_Models()
    
    def fit_Models(self):
        # Fit classification or regression models
        results = []

        if self.classification:
            self.classification_models = [
                ('Logistic Regression', LogisticRegression()),
                ('Decision Tree', DecisionTreeClassifier()),
                ('Random Forest', RandomForestClassifier()),
                ('SVC', SVC()),
            ]

            for model_name, model in self.classification_models:
                model.fit(self.X_train, self.y_train)

                predicted = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, predicted)
                report = classification_report(self.y_test, predicted, output_dict=True)

                result = {
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': report['weighted avg']['precision'],
                    'Recall': report['weighted avg']['recall'],
                    'F1-score': report['weighted avg']['f1-score'],
                }

                results.append(result)

        else:
            self.regression_models = [
                ('Linear Regression', LinearRegression()),
                ('Decision Tree', DecisionTreeRegressor()),
                ('Random Forest', RandomForestRegressor()),
                ('SVR', SVR()),
            ]

            for model_name, model in self.regression_models:
                model.fit(self.X_train, self.y_train)

                predicted = model.predict(self.X_test)
                mae = mean_absolute_error(self.y_test, predicted)
                mse = mean_squared_error(self.y_test, predicted)
                rmse = mean_squared_error(self.y_test, predicted, squared=False)
                r2 = r2_score(self.y_test, predicted)

                result = {
                    'Model': model_name,
                    'Mean Absolute Error (MAE)': mae,
                    'Mean Squared Error (MSE)': mse,
                    'Root Mean Squared Error (RMSE)': rmse,
                    'R-squared (R2)': r2,
                }

                results.append(result)

        self.results_df = pd.DataFrame(results)
    
    def custom_Record_Prediction(self, pred_df):
        # Predict custom records
        pred_df[self.continuos_cols_without_y] = self.scaler.transform(pred_df[self.continuos_cols_without_y])
        
        for col, encoder in self.label_encoders.items():
            pred_df[col] = encoder.transform(pred_df[col])

        final_df = pred_df.iloc[:, self.selected_feature_indices]
        
        results = []

        if self.classification:
            for model_name, model in self.classification_models:
                predicted = model.predict(final_df)
                result = {
                    'Model': model_name,
                    'Predicted': predicted
                }

                results.append(result)
        else:
            for model_name, model in self.regression_models:
                predicted = model.predict(final_df)
                result = {
                    'Model': model_name,
                    'Predicted': predicted
                }

                results.append(result)
        
        if self.Target_encoder is not None:
            return pd.DataFrame(results), self.Target_encoder
        else:
            return pd.DataFrame(results), None
