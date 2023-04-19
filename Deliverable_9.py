import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# load the dataset
boston = load_boston()

# create features and response data frames
features = pd.DataFrame(boston.data, columns=boston.feature_names)
response = pd.DataFrame(boston.target, columns=['MEDV'])

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, response, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# create the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# fit the model to the training data
rf_model.fit(X_train, y_train.values.ravel())

# generate predictions over the test data
y_pred = rf_model.predict(X_test)

# evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

