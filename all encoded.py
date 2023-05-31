import matplotlib.pyplot as plt
import pandas as p
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from preprocessing import *

data = p.read_csv('cars-train.csv')

cols = ['model', 'brand', 'condition', 'fuel_type', 'transmission', 'drive_unit', 'year', 'mileage(kilometers)',
        'volume(cm3)']
X, y = feature_engineering(data)
X = func(X)
X['year'] = X['year'].apply(func)
X = Feature_Encoder(X, cols)

# figure(figsize=(8, 6), dpi=500)
# sns.heatmap(X.corr(), annot=True)
# plt.show()
# X.to_csv('submission.csv', encoding='utf-8', index=False)
# df = pd.DataFrame(X)
# col = df.columns.get_indexer(['year', 'mileage(kilometers)', 'volume(cm3)'])
# X = featureScaling(X, col)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=9)

poly_feature = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly_feature.fit_transform(X_train)
model1 = linear_model.LinearRegression()
model = linear_model.Ridge()
model1.fit(X_train_poly, y_train)
model.fit(X_train_poly, y_train)
# model = make_pipeline(PolynomialFeatures(6), Ridge(alpha=0.01, solver='cholesky'))
# model.fit(X_train, y_train)

scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
model_score = abs(scores.mean())
print("model cross validation score is " + str(model_score))

y_train_predicted = model.predict(poly_feature.fit_transform(X_test))
print('Mean Square Error', metrics.mean_squared_error(y_test, y_train_predicted))

y_train_predicted = model1.predict(poly_feature.fit_transform(X_test))
print('Mean Square Error', metrics.mean_squared_error(y_test, y_train_predicted))
