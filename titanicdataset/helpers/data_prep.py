
# DATA PRE-PROCESSING
# eda: keşifsel veri analizi- exploratory data analysis
# # 1. Outliers (Aykırı Değerler)
# # 2. Missing Values (Eksik Değerler)
# # 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# # 4. Feature Scaling (Özellik Ölçeklendirme)


# Import Libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


#import os
#import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

# EDA modülü:
#from helpers.eda import *

#df = load_dataset("titanic")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

############## ############## ############## ################
############## 1.Outliers (Aykırı Değerler) #################

# Outlier Threshold'ların belirlenmesi:
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    q1 = dataframe[col_name].quantile(q1)  # 1.Çeyrek
    q3 = dataframe[col_name].quantile(q3)  # 3.Çeyrek
    interquantile_range = q3 - q1  # range'i hesaplayalım
    low_limit = q1 - 1.5 * interquantile_range # low & up limit:
    up_limit = q3 + 1.5 * interquantile_range
    return low_limit, up_limit

# Outlier Değer Var mı Yok Mu:
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Outlier Değerlere Erişmek:
def grab_outliers(dataframe, col_name, index=False):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].head())
    else:
        print(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)])
    if index:
        outlier_index = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].index
        return outlier_index

# Outlier Değer Problemini Çözme:

# Silme ile remove_outlierçözümleme:
def remove_outlier (dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

# Baskılama yöntemi:
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

############## 2.Missing Values (Eksik Değerler)  #################

# Eksik Değerlerin Yakalanması
def missing_values_table(dataframe, na_name=False):
    na_column = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] # missing değer iceren kolon adı
    n_miss = dataframe[na_column].isnull().sum().sort_values(ascending=False) # boş gözlem sayısı
    ratio = (dataframe[na_column].isnull().sum() * 100/ dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.DataFrame({"n_miss":n_miss, "n_miss_ratio":ratio})
    # missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_column

# Eksik değerler ile target arasındaki ilişki:
def missing_vs_target(dataframe, target):
    temp_df = dataframe.copy()
    na_cols = [var for var in dataframe.columns if dataframe[var].isnull().sum() > 0]
    for col in na_cols:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)# case when
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


# Kategorik değişkenleri mode ile doldurma:
def missing_cat_cols_fill(dataframe, show_na_col_name=False):
    cat_cols = grab_col_names(dataframe)[0]
    for col in cat_cols:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
    return dataframe


############## 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding) #################

# Label Encoding & Binary Encoding
#2 sınıf ya da cok sınıflı ordinal

def binary_cols(dataframe, target):
    binary_col_names = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                  and target not in col
                  and dataframe[col].nunique() == 2]
    return binary_col_names

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# One-Hot-Encoding:
def ohe_cols(dataframe):
    ohe_col_names = [col for col in dataframe.columns if (dataframe[col].dtype == "O" and 10 >= dataframe[col].nunique() > 2)]
    return ohe_col_names


def one_hot_encoder(dataframe, ohe_col_names, drop_first=True):
    dms = pd.get_dummies(dataframe[ohe_col_names], drop_first=drop_first)    # dummy değişken üretilecek kategorik değişkenler (ohe cols)
    df_ = dataframe.drop(columns=ohe_col_names, axis=1)              # dummy değişkene çevirdiğimiz değişkeni dışarıda tutacak yeni df'i oluşturalım
    dataframe = pd.concat([df_, dms],axis=1)                      # 1.ve 2.adımdaki dataframe'i birleştirelim
    return dataframe


# Rare Analyzing & Encoding
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc=0.01):

    # temp_df = dataframe.copy() # orjinal veri setini bozmamak için copy aldık

    # kategorik değişkenlerin frekanslarına bak, rare_perc altında kalan sınıflar için
    # dönen boolean değerleri topla, 1'den büyükse yani en az th altında kalan min 2 sınıflı varsa
    # (birleştirmek içi bu koşula bakıyoruz) rare_columns olarak etiketle
    cat_cols = grab_col_names(dataframe)[0]
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) <= rare_perc).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe) # th altında kalan sınıfı olan değişkenlerin sınıf frekanslarından olusan df yarat
        rare_labels = tmp[tmp <= rare_perc].index # sınıf frekansı < th olanların indexlerini bul
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col]) # th altında kalan değerleri Rare olarak grupla

    return dataframe

# Useless_Cols:
def useless_cols(dataframe, rare_perc=0.01):
    cat_cols = grab_col_names(dataframe)[0]
    useless_col_names = [col for col in cat_cols if dataframe[col].nunique() == 2 and
                        (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]
    return dataframe

##############  4. Scaling ##############

def StandartScaling(dataframe, col_name):
    ss = StandardScaler()
    dataframe[col_name] = ss.fit_transform(dataframe[col_name])
    return dataframe

def MinMaxScaling(dataframe, col_name):
    mms = MinMaxScaler()
    dataframe[col_name] = mms.fit_transform(dataframe[col_name])
    return dataframe

def RobustScaling(dataframe, col_name):
    rs = RobustScaler()
    dataframe[col_name] = rs.fit_transform(dataframe[col_name])
    return dataframe

def Scaling(dataframe, method):
    numerical_cols = grab_col_names(dataframe)[1]
    if method == "StandartScaling":
        StandartScaling(dataframe, numerical_cols)
    elif method == "MinMaxScaling":
        MinMaxScaling(dataframe, numerical_cols)
    else:
        RobustScaling(dataframe, numerical_cols)
    return dataframe