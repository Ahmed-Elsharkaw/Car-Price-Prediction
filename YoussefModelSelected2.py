import matplotlib.pyplot as plt
import pandas as p
import seaborn as sns
import sklearn.preprocessing
from matplotlib.pyplot import figure
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from pre_youssef import *
from sklearn.preprocessing import StandardScaler
from math import *

data = p.read_csv('cars-train.csv')
cols = ['condition', 'fuel_type', 'transmission', 'drive_unit']
X, y = feature_engineering(data)
# X = Feature_Encoder(X, cols)
X = Feature_Encoder1(X, cols)

# figure(figsize=(8, 6), dpi=100)
# sns.heatmap(X.corr(), annot=True)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=9)

standardScalar = StandardScaler()
X_train[['mean price', 'mileage(kilometers)', 'volume(cm3)']] = standardScalar.fit_transform(
    X_train[['mean price', 'mileage(kilometers)', 'volume(cm3)']])
X_test[['mean price', 'mileage(kilometers)', 'volume(cm3)']] = standardScalar.fit_transform(
    X_test[['mean price', 'mileage(kilometers)', 'volume(cm3)']])

poly_feature = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly_feature.fit_transform(X_train)
model1 = linear_model.LinearRegression()
model = linear_model.Ridge()
model1.fit(X_train_poly, y_train)
model.fit(X_train_poly, y_train)

scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
model_score = abs(scores.mean())
print("model cross validation score is " + str(model_score))

y_train_predicted = model.predict(poly_feature.fit_transform(X_test))
print('Mean Square Error ridge', (metrics.mean_squared_error(y_test, y_train_predicted)))

y_train_predicted = model1.predict(poly_feature.fit_transform(X_test))
print('Mean Square Error', (metrics.mean_squared_error(y_test, y_train_predicted)))
