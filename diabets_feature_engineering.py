# BUSINESS PROBLEM
# It is required to develop a machine learning model that can predict whether individuals are diabetic based on their features.
# Before building the model, you are expected to perform the necessary data analysis and feature engineering steps.

# DATASET STORY
# The dataset is part of a larger dataset maintained by the National Institute of Diabetes and Digestive and Kidney Diseases in the United States.
# It contains data used for a diabetes study conducted on Pima Indian women aged 21 and above, living in Phoenix, Arizona—the fifth-largest city in the state.
# The target variable is labeled as "outcome", where 1 indicates a positive diabetes test result and 0 indicates a negative result.



# 9 Variables, 768 Observations, 24 KB

# | Variable                     | Description                                                                  |
# | ---------------------------- | ---------------------------------------------------------------------------- |
# | **Pregnancies**              | Number of pregnancies                                                        |
# | **Glucose**                  | Plasma glucose concentration after 2 hours in an oral glucose tolerance test |
# | **BloodPressure**            | Diastolic blood pressure (mm Hg)                                             |
# | **SkinThickness**            | Skinfold thickness                                                           |
# | **Insulin**                  | 2-hour serum insulin level (mu U/ml)                                         |
# | **DiabetesPedigreeFunction** | A function representing diabetes pedigree (based on glucose test)            |
# | **BMI**                      | Body Mass Index (weight in kg / (height in m)^2)                             |
# | **Age**                      | Age (in years)                                                               |
# | **Outcome**                  | Indicates whether the person has diabetes (1) or not (0)                     |

# PROJECT TASKS
# TASK 1: Exploratory Data Analysis (EDA)
#       -✅ Step 1: Examine the Overall Picture
#               Review the general structure of the dataset, including dimensions, column types, and sample records.
#
#       -✅ Step 2: Identify Numerical and Categorical Variables
#               Detect which variables are numerical and which ones are categorical based on data types and unique values.
#
#       -✅ Step 3: Analyze Numerical and Categorical Variables
#               Perform summary statistics and visualizations to better understand the distribution and characteristics of each variable type.
#
#       -✅ Step 4: Analyze the Target Variable
#               Calculate the mean of the target variable (Outcome) for each categorical variable.
#               Calculate the mean of numerical variables grouped by the target variable.
#
#       -✅ Step 5: Perform Outlier Analysis
#               Identify and examine outlier values using statistical methods (e.g., IQR, Z-score).
#
#       -✅ Step 6: Perform Missing Value Analysis
#               Detect missing or invalid values (e.g., zero values in medical data) and decide how to handle them.
#
#       -✅ Step 7: Perform Correlation Analysis
#               Analyze the relationships between numerical variables and visualize them using a correlation matrix.


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Step 1: Examine the Overall Picture
df = pd.read_csv("datasets/diabetes.csv")
df.head()
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
# Step 2: Identify Numerical and Categorical Variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}') # Burası zaten cat_cols'un içerisindedir. Sadece bilgi amaçlıdır.
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols
num_cols
cat_but_car

# Step 3: Analyze Numerical and Categorical Variables
# Categorical Variables Analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "Outcome", True)

# Numerical Variables Analysis
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Numerical Variables Analysis a Single Figure
df[num_cols].describe().T
df[num_cols].hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.show(block=True)

# Step 4: Analyze the Target Variable
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Step 5: Perform Outlier Analysis

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

# We see that there are outliers in all columns.
# We just analyzed the outliers, but we did not remove them yet.

# Step 6: Perform Missing Value Analysis

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)
df.isnull().sum()

df.head()

# We see that there are no missing values in the dataset.

# Step 7: Perform Correlation Analysis
# Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show(block=True)

# ‘Outcome’ and Other Variables:
# •   ‘Glucose’ and ‘Outcome’: 0.47 — Strong positive correlation.
# •   ‘BMI’ and ‘Outcome’: 0.29 — Moderate positive correlation.
# •   ‘Age’ and ‘Outcome’: 0.24 — Moderate positive correlation.
# •   ‘Pregnancies’ and ‘Outcome’: 0.22 — Moderate positive correlation.

# Correlations Between Variables:
# •   ‘Pregnancies’ and ‘Age’: 0.54 — Strong positive correlation.
# •   ‘SkinThickness’ and ‘Insulin’: 0.44 — Moderate positive correlation.
# •   ‘BMI’ and ‘SkinThickness’: 0.39 — Moderate positive correlation.

# By examining this correlation matrix, we can see that the ‘Glucose’ variable likely plays an important role in diabetes diagnosis (Outcome). Additionally, variables such as ‘BMI’ and ‘Age’ may also be associated with diabetes diagnosis.

# KORELASYON'DA BİRBİRİYLE DOĞRU ORAN İLİŞKİ VARSA POZİTİF YÖNLÜ KORELASYON
# KORELASYONDA BİRBİRİYLE TERS ORAN İLİŞKİ VARSA NEGATİF YÖNLÜ KORELASYON
# BİR DEĞİŞKEN AZALIRKEN BİR DEĞİŞKENİN ARTMASI TERS ORANTILI KORELASYON
# BİR DEĞİŞKEN ARTARKEN BİR DEĞİŞKENİN ARTMASI DOĞRU ORANTILI KORELASYON


# TASK 2 : Feature Engineering
#   -✅Step 1: Handle Missing and Outlier Values
#           There are no missing values in the dataset by default. \
#           However, in variables such as Glucose and Insulin, observations with a value of 0 may actually represent missing data. \
#           For example, it is physiologically impossible for a person to have a glucose or insulin level of zero. \
#           Considering this, you can replace zero values in the relevant variables with NaN and then apply appropriate missing value treatments.

#   -✅Step 2: Create New Features

#   -✅Step 3:  Perform Encoding Operations

#   -✅Step 4: Standardize Numerical Variables

#   -✅Step 5: Build a Machine Learning Model


# Step 1: Handle Missing and Outlier Values

# The values of variables other than Pregnancies and Outcome cannot be 0.
# Therefore, 0 values can be replaced with NaN

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.head()

na_columns = missing_values_table(df, na_name=True)

# We can see that there are missing values in the dataset now. So, we can apply missing value treatments.

# Analyze the relationship between missing values and the target variable:
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

df.isnull().sum()
missing_vs_target(df, "Outcome", na_columns)

# Filling missing values
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()
df.isnull().sum()

for col in num_cols:
    print(col, check_outlier(df, col))

# Step 2: Create New Features

# Create a new age variable by categorizing the age variable
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

# BMI below 18.5 is considered underweight, between 18.5 and 24.9 is normal, between 24.9 and 29.9 is overweight, and above 30 is obese
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Convert the glucose value to a categorical variable.
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Create a categorical variable by considering age and BMI together
df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "under_weight_mature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "under_weight_senior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthy_mature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthy_senior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "over_weight_mature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "over_weight_senior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obese_mature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obese_senior"

# Create a categorical variable by considering age and glucose levels together
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "low_mature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "low_senior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normal_mature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normal_senior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hidden_mature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hidden_senior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "high_mature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "high_senior"
df.head()

# Create a categorical variable based on insulin values
def set_insulin(dataframe, col_name="Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Anormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]


df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

df.columns = [col.upper() for col in df.columns]
df.head()


# Step 3: Perform Encoding Operations
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Label encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# One hot encoding
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()



# Step 4: Standardize Numerical Variables
num_cols
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()
df.shape


# Step 5: Build a Machine Learning Model
y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")


# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)






