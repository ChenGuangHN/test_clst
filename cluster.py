# -*- coding:utf8 -*-
'''
 程序：自动广告报告关键词聚类分析
 作者：陈广
 版本：20171110
'''

import pandas as pd
import time
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_table("AUTOADS_ALLDEUS_2017-11-05.txt", encoding="utf8")
data.columns = [
    "Campaign_Name",
    "Ad Group_Name",
    "Customer_Search_Term",
    "Keyword",
    "Match_Type",
    "First_Day_of_Impression",
    "Last_Day_of_Impression",
    "Impressions",
    "Clicks",
    "CTR",
    "Total_Spend",
    "Average_CPC",
    "ACoS",
    "Currency",
    "Orders_placed_within_1_week_of_a_click",
    "Product_Sales_within_1_week_of_a_click",
    "Conversion_Rate_within_1_week_of_a_click",
    "Same_SKU_units_Ordered_within_1_week_of_click",
    "Other_SKU_units_Ordered_within_1_week_of_click",
    "Same_SKU_units_Product_Sales_within_1_week_of_click",
    "Other_SKU_units_Product_Sales_within_1_week_of_click"
]

# def rep(string):
#     return string.replace(',', '.')
#
# data['Total_Spend'] = data['Total_Spend'].apply(rep).apply(float)
# data['Product_Sales_within_1_week_of_a_click'] = data['Product_Sales_within_1_week_of_a_click'].apply(rep).apply(float)

# data['First_Day_of_Impression'] = pd.to_datetime(data['First_Day_of_Impression'], dayfirst=True)
# data['Last_Day_of_Impression'] = pd.to_datetime(data['Last_Day_of_Impression'], dayfirst=True)

data['First_Day_of_Impression'] = pd.to_datetime(data['First_Day_of_Impression'])
data['Last_Day_of_Impression'] = pd.to_datetime(data['Last_Day_of_Impression'])

data["period"] = data["Last_Day_of_Impression"] - data["First_Day_of_Impression"]

# data["Impression_level"] = 0
# data["Impression_level"].loc[data["Impressions" <= 500]] = 0
# data["Impression_level"].loc[data["Impressions" <= 1000] & data["Impressions"] > 500] = 1
# data["Impression_level"].loc[data["Impressions" <= 1500] & data["Impressions"] > 1000] = 2
# data["Impression_level"].loc[data["Impressions" <= 2000] & data["Impressions"] > 1500] = 3
# data["Impression_level"].loc[data["Impressions" <= 2500] & data["Impressions"] > 2000] = 4
# data["Impression_level"].loc[data["Impressions" <= 3000] & data["Impressions"] > 2500] = 5
# data["Impression_level"].loc[data["Impressions" <= 3500] & data["Impressions"] > 3000] = 6
# data["Impression_level"].loc[data["Impressions" <= 4000] & data["Impressions"] > 3500] = 7
# data["Impression_level"].loc[data["Impressions" <= 4500] & data["Impressions"] > 4000] = 8


def timedelta2int(timedel):
    return timedel.days
data["period"] = data["period"].apply(timedelta2int)
bool_index = (data["Keyword"] != "*") & (data["period"] >= 10) & (data["Impressions"] <= 4500) & (data["Clicks"] <= 35)
data1 = data[bool_index]

# -------------------------------------------------------------------------------------------
# Kmeans 模型
x = data1[["Impressions",
           "Clicks",
           "Total_Spend",
           "Orders_placed_within_1_week_of_a_click",
           "Product_Sales_within_1_week_of_a_click",
           "period"]]

x_scaled = preprocessing.StandardScaler().fit(x)
# kmeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_scaled.transform(x))
data1["label"] = kmeans.labels_

# 轮廓系数
silhouette_avg = silhouette_score(x, kmeans.labels_)
print "The average silhouette_score is : ", silhouette_avg
tm = time.strftime("%Y%m%d%H%M%S", time.localtime())
# file_name = "kmeans_cluster_%s.xlsx" %tm
# data1.to_excel(file_name, encoding="utf8")

# -------------------------------------------------------------------------------------------
# 层次聚类 Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)
    clustering.fit(x)
    silhouette_avg = silhouette_score(x, clustering.labels_)
    print "The average silhouette_score is : ", silhouette_avg
    file_name = "agglomerative_%s_cluster_%s.xlsx" %(linkage,tm)
    data1.to_excel(file_name, encoding="utf8")



