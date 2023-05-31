from YoussefModelSelected2 import *
from pre_youssef import *

data = pd.read_csv('cars-test.csv')
df = pd.DataFrame(data)


def stringMatchScore(str1, str2, str3, key1, key2, key3):
    if str1 == key1 and str2 == key2:
        return True
    else:
        return False


def stringMatchBrand(str1, key1):
    if str1 == key1:
        return True
    else:
        return False


def mostLikeStr(str1, str2, str3):
    flag = 0
    years = []
    for x in brandModelMean:
        newString = x.split(",", 3)
        score = stringMatchScore(str1, str2, str3, newString[0], newString[1], newString[2])
        if score:
            years.append(newString[2])
            flag = 1
    min = 100000000000
    if flag:
        for year in range(len(years)):
            word2 = func(str3)
            word1 = func(years[year])
            diff = abs(int(word1) - int(word2))
            if diff < min:
                min = diff
                newstr3 = years[year]
        str3 = newstr3
    if flag:
        return str1 + "," + str2 + "," + str3
    return "modelNotFound"


def mostLikeBrand(str1):
    if brandMean.get(str1, -1) != -1:
        return float(brandMean.get(str1))
    return "modelNotFound"


df['volume(cm3)'].fillna(df['volume(cm3)'].mean(), inplace=True)
df['drive_unit'].fillna(df['drive_unit'].mode()[0], inplace=True)
df['segment'].fillna(df['segment'].mode()[0], inplace=True)
li = []
df.drop('car_id', axis='columns', inplace=True)
new = df["car-info"].str.split(",", n=3, expand=True)

for n in range(df.shape[0]):
    str = df['car-info'][n]
    strSplit = str.split(",", 3)
    if brandModelMean.get(str, -1) == -1:
        str = mostLikeStr(strSplit[0], strSplit[1], strSplit[2])
        if str == "modelNotFound":
            str = mostLikeBrand(strSplit[1])
            if str != "modelNotFound":
                li.append(float(str))
                continue
            else:
                li.append(float(priceMean))
                continue
    li.append(float(brandModelMean.get(str)))

df.drop('car-info', axis='columns', inplace=True)
df.drop('color', axis='columns', inplace=True)
df.drop('segment', axis='columns', inplace=True)
df['fuel_type'] = df['fuel_type'].str.lower()
df['drive_unit'] = df['drive_unit'].str.lower()
df['condition'] = df['condition'].str.lower()
df['mean price'] = li
cols = ['condition', 'fuel_type', 'transmission', 'drive_unit']
df = Feature_Encoder1(df, cols)
standardScalar = StandardScaler()
df[['mean price', 'mileage(kilometers)', 'volume(cm3)']] = standardScalar.fit_transform(
    df[['mean price', 'mileage(kilometers)', 'volume(cm3)']])
df[['mean price', 'mileage(kilometers)', 'volume(cm3)']] = standardScalar.fit_transform(
    df[['mean price', 'mileage(kilometers)', 'volume(cm3)']])

target = model.predict(poly_feature.fit_transform(df))

ta = pd.read_csv('sample_submission.csv')
ta['price(USD)'] = target
ta.to_csv('feature_eng_selected2.csv', encoding='utf-8', index=False)
