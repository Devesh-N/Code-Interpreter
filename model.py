import joblib
import pandas as pd

# Load the model from the file
model = joblib.load('model.pickle')

# Now you can use the model to make predictions. For example:
X_new = pd.DataFrame({
    'area': [5000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': ['yes'],
    'guestroom': ['no'],
    'basement': ['yes'],
    'hotwaterheating': ['no'],
    'airconditioning': ['yes'],
    'parking': [1],
    'prefarea': ['no'],
    'furnishingstatus': ['furnished']
})

y_pred = model.predict(X_new)
print(y_pred)
#end The code below is for generating the Pickle file if it does not work for you
# import pandas as pd
# data = pd.read_csv('Housing.csv')
# X = data.drop('price', axis=1)
# y = data['price']
# from sklearn.model_selection import train_test_split
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import GradientBoostingRegressor
# 
# numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())])
# 
# categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# 
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])
# 
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('regressor', GradientBoostingRegressor(random_state=42))])
# 
# model.fit(X_train, y_train)
# import pickle
# 
# with open('model.pickle', 'wb') as f:
#     pickle.dump(model, f)
# with open('model.pickle', 'rb') as f:
#     model = pickle.load(f)
#     