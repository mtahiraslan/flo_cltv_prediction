##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma Model
##############################################################

###############################################################
# TASKS
###############################################################

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
import numpy as np

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

###############################################################
# Task 1: Preparing data
###############################################################

# 1. Read Flo_data_20K.CSV dataset. Create a copy of the DataFrame.

main_df = pd.read_csv('CLTV/flo_data_20k.csv')
df = main_df.copy()
df.head()


def missing_values_analysis(df):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=True)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df


def check_dataframe(df, row_num=5):
    print("*************** Dataset Shape ***************")
    print("No. of Rows:", df.shape[0], "\nNo. of Columns:", df.shape[1])
    print("*************** Types of Columns ***************")
    print(df.dtypes)
    print(f"*************** First {row_num} Rows ***************")
    print(df.head(row_num))
    print(f"*************** Last {row_num} Rows ***************")
    print(df.tail(row_num))
    print("*************** Summary Statistics of The Dataset ***************")
    print(df.describe().T)
    print("*************** Dataset Missing Values Analysis ***************")
    print(missing_values_analysis(df))


check_dataframe(df)


# # 2. Define Outlier_thresholds and Replace_with_thresholds functions that are required to suppress contradictory
# values. Note: Frequency values must be integer when calculating CLTV. Therefore, round the upper and lower
# limits with round ().


def outlier_thresholds(dataframe, column):
    quartile_1 = dataframe[column].quantile(0.01)
    quartile_3 = dataframe[column].quantile(0.99)
    iqr_range = quartile_3 - quartile_1
    up_limit = quartile_3 + 1.5 * iqr_range
    low_limit = quartile_1 - 1.5 * iqr_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


# 3. Determine if variables "order_num_total_ever_online", "order_num_total_ever_offline",
# "customer_value_total_ever_offline" and "customer_value_total_ever_online" have contradictory values


threshold_columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
                     "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in threshold_columns:
    replace_with_thresholds(df, col)

check_dataframe(df)

# 4. Omnichannel states that customers shop both online and offline platforms. Create new variables for
# the total number of shopping and spending of each customer.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Examine the variable types. Turn the type of variables containing "date" to the datetime type.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

###############################################################
# Task 2: Creation of CLTV data structure
###############################################################

# 1.Take 2 days after the last shopping on the data set as an analysis date.

df["last_order_date"].max()
last_date = dt.datetime(2021, 6, 1)

# 2.Create a new cltv dataframe with customer_id, recency_cltv_weekly, T_weekly, Frequency and Monetary_CLTV_AVG values.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["T_weekly"] = ((last_date - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()

###############################################################
# Task 3: Establishing BG/NBD and Gamma-Hamma models, calculation of 6 months of CLTV
###############################################################

# 1. Create the BG/NBD model

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Estimate the expected purchases from customers in 3 months and add to the CLTV DataFrame as exp_sales_3_month.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# Estimate the expected purchases from customers in 6 months and add to the CLTV DataFrame as exp_sales_6_month.

cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# We examine the 10 people who will make the most purchase in the 3rd and 6th month.

cltv_df.groupby('customer_id').agg({'exp_sales_3_month': 'sum',
                                    'exp_sales_6_month': 'sum'}).sort_values(by='exp_sales_6_month',
                                                                             ascending=False).head(10)

# 2. Fit the gamma-gamma model. Estimates the average value of customers and add to CLTV DataFrame as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()

# 3.Calculate 6-month CLTV and add dataframe with the name 'cltv'.

cltv_df['cltv'] = ggf.customer_lifetime_value(bgf,
                                              cltv_df['frequency'],
                                              cltv_df['recency_cltv_weekly'],
                                              cltv_df['T_weekly'],
                                              cltv_df['monetary_cltv_avg'],
                                              time=6,
                                              freq="W",
                                              discount_rate=0.01)

# Observe 20 people at the highest value of CLTV.

cltv_df.sort_values("cltv", ascending=False).head(20)

###############################################################
# Task 4: Creation of segments according to CLTV
###############################################################

# 1. According to 6 -month CLTV, separate all your customers into 4 groups (segment) and add the group names to
# the data set. Assign with the name CLTV_segment.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# 2.Check out the averages of the segments of the Recency, Frequency and Monetary.

cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})
