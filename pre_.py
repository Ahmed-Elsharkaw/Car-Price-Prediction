import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def func(x):
    x = x.replace('[', '')
    x = x.replace('(', '')
    x = x.replace(')', '')
    x = x.replace(']', '')
    return x


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def Feature_Encoder1(X, cols):
    newDataFrame = pd.get_dummies(X, columns=cols)
    newDataFrame.drop(['fuel_type_electrocar'], axis=1, inplace=True)
    return newDataFrame


brandModel = {}
brandModelMean = {}
priceMean = 0
brand = {}
brandMean = {}


def feature_engineering(X):
    li = []
    df = pd.DataFrame(X)
    priceMean = df['price(USD)'].mean()
    df.drop('car_id', axis='columns', inplace=True)
    new = df["car-info"].str.split(",", n=3, expand=True)
    for n in range(df.shape[0]):
        current = df['car-info'][n]
        currentSplit = current.split(",", 3)
        if brand.get(currentSplit[1], -1) == -1:
            brand.update({currentSplit[1]: 1})
            price = float(df['price(USD)'][n])
            brandMean.update({currentSplit[1]: price})
        else:
            freq = float(brand.get(currentSplit[1]))
            freq = freq + 1
            brand.update({currentSplit[1]: freq})
            oldPrice = float(brandMean.get(currentSplit[1]))
            newPrice = float(df['price(USD)'][n])
            oldPrice = oldPrice + newPrice
            brandMean.update({currentSplit[1]: oldPrice})
    for item in brandMean.items():
        price = float(item[1])
        str = item[0]
        freq = float(brand.get(str))
        target = float(price / freq)
        brandMean.update({str: target})
    for n in range(df.shape[0]):
        current = df['car-info'][n]
        if brandModel.get(current, -1) == -1:
            brandModel.update({current: 1})
            price = float(df['price(USD)'][n])
            brandModelMean.update({current: price})
        else:
            freq = float(brandModel.get(current))
            freq = freq + 1
            brandModel.update({current: freq})
            oldPrice = float(brandModelMean.get(current))
            newPrice = float(df['price(USD)'][n])
            oldPrice = oldPrice + newPrice
            brandModelMean.update({current: oldPrice})
    for item in brandModelMean.items():
        price = float(item[1])
        str = item[0]
        freq = float(brandModel.get(str))
        target = float(price / freq)
        brandModelMean.update({str: target})
    for n in range(X.shape[0]):
        str = X['car-info'][n]
        li.append(float(brandModelMean.get(str)))
    df['fuel_type'] = df['fuel_type'].str.lower()
    df['drive_unit'] = df['drive_unit'].str.lower()
    df['condition'] = df['condition'].str.lower()
    df.drop('car-info', axis='columns', inplace=True)
    df['mean price'] = li
    df['volume(cm3)'].fillna(df['volume(cm3)'].mean(), inplace=True)
    df['drive_unit'].fillna(df['drive_unit'].mode()[0], inplace=True)
    df.drop(['color'], axis=1, inplace=True)
    df.drop(['segment'], axis=1, inplace=True)
    df.dropna(inplace=True)
    y = df['price(USD)']
    df.drop('price(USD)', axis='columns', inplace=True)
    return df, y
