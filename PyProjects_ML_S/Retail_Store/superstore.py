# -*- coding: utf-8 -*-
"""
Content for Retail Sales Analysis
    0. Clear Memory
    1. Import
    2. Read data
    3. Function
    4. Explore data
    5. Visualization
    6. Data Analysis
"""
# 0 Clear memory
reset -f

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

import os
path  = os.path.abspath("Retail_Store/Data/superstore_data.csv")
pd.options.display.max_columns = 300

df = pd.read_csv(path, encoding = "ISO-8859-1")

#==============================================================================
#EDA
df.shape
df.head()
df.columns
df.dtypes

def print_rows(name_column):
    return df[name_column][0:5]

def describe_col(name_column):
    return df[name_column].describe()

df.info()


df.isnull().sum()

print_rows("Customer Name")
describe_col("Customer Name")

df['Order Date'] = pd.to_datetime(df['Order Date'])
describe_col('Order Date')

len(df["Customer Name"].unique())


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()
    
plotPerColumnDistribution(df, 10, 5)   
df.dataframeName = "SuperStore"
plotCorrelationMatrix(df, 8)

plotScatterMatrix(df, 20, 10)
#==============================================================================
#1. Who are the top-20 most profitable customers. Show them through plots.
#==============================================================================

result = df.groupby(["Customer Name"])['Profit'].aggregate(np.sum).reset_index().sort_values('Profit',ascending = False).head(20)

result.head()
type(result)
sns.barplot(x= "Customer Name", y='Profit',data = result)

fig = plt.figure(figsize = (5,5))

ax1 = fig.add_subplot(111)
sns.barplot(x = "Customer Name",y= "Profit",
            data=result,
            ax = ax1
             )
ax1.set_ylabel("Profit", fontname="Arial", fontsize=12)

# Set the title to Comic Sans
ax1.set_title("Top 20 Customers", fontname='Comic Sans MS', fontsize=18)

# Set the font name for axis tick labels to be Comic Sans
for tick in ax1.get_xticklabels():
    tick.set_fontname("Comic Sans MS")
    tick.set_fontsize(12)
for tick in ax1.get_yticklabels():
    tick.set_fontname("Comic Sans MS")
    tick.set_fontsize(12)

# Rotate the labels as the Customer names overwrites on top of each other
ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)
plt.show()
    
#=================================================================================
# 2. What is the distribution of our customer segment
#=================================================================================
descending_order = df['Segment'].value_counts().index
df.Segment.value_counts()
df['Segment'].unique()

sns.countplot("Segment", data = df, order = descending_order)

#---Observations
# Segment is categorical attribute with 3 levels - Consumer, Corporate & Home Office. The distribution is highest in Consumer
# followed by Corporate and Home Office
#=====================================================================================
#3. Who are our top-20 oldest customers
#=====================================================================================
df.dtypes.value_counts()
df['Order Date'] = pd.to_datetime(df['Order Date'])

oldest = pd.DataFrame({'Count' : df.groupby(["Order Date","Customer Name"]).size()}).reset_index()

oldest.head()

#=========================================================================================
#4. Which customers have visited this store just once
#==========================================================================================
Customers_visit = pd.DataFrame({'Count' : df.groupby(["Customer Name"]).size()}).reset_index()

Customers_visit[Customers_visit['Count'] == 1]

#==========================================================================================
#5. Relationship of Order Priority and Profit
df['Order Priority'].value_counts()

sns.boxplot(
            "Order Priority",
            "Profit",
             data= df
             )

#6. What is the distribution of customers Market wise?
df.shape
df['Market'].value_counts()
Customers_market = pd.DataFrame({'Count' : df.groupby(["Market","Customer Name"]).size()}).reset_index()
Customers_market.shape

sns.barplot(x = "Market",     # Data is groupedby this variable
             y= "Count",    # Aggregated by this variable
             data=Customers_market
             )

sns.countplot("Market",        # Variable whose distribution is of interest
                data = Customers_market)

#7. What is the distribution of customers Market wise and Region wise
df['Region'].value_counts()
Customers_market_region = pd.DataFrame({'Count' : df.groupby(["Market","Region","Customer Name"]).size()}).reset_index()

sns.countplot("Market",        # Variable whose distribution is of interest
              hue= "Region",    # Distribution will be gender-wise
              data = Customers_market_region)

#8.Distribution of  Customers by Country & State - top 15
Customers_Country = pd.DataFrame({'Count' : df.groupby(["Country","State"]).size()}).reset_index().sort_values('Count',ascending = False).head(15)
Customers_Country

sns.barplot(x = "Country",     # Data is groupedby this variable
            y= "Count",  
            hue="State",
            data = Customers_Country.sort_values('Country')
            )
## US has the largest number of customers -California being the largest followed by New York, Washington, Illinois & Ohio
## UK has the next largest population of Customers -England

# Top 20 Cities by Sales Volume
sale_cities = df.groupby(["City"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False).head(20)
sns.barplot(x = "City",     # Data is groupedby this variable
            y= "Quantity",          
            data=sale_cities,
            )

# top 10 products
sale_Products = df.groupby(["Product Name"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False).head(20)
sns.barplot(x = "Product Name",     # Data is groupedby this variable
            y= "Quantity",          
            data=sale_Products)
#Staples is the largest selling product

# top selling products by countries (in US)
df.columns
sale_Products_Country = df.groupby(["Product Name","Country"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)
sale_Products_Country = df.groupby(["Product Name","Country"])['Quantity'].sum().reset_index().sort_values('Quantity',ascending = False)
sale_Products_Country
type(sale_Products_Country)
spc = sale_Products_Country[sale_Products_Country['Country'] == "United States"].sort_values('Quantity',ascending = False).head(20)
sns.barplot(x = "Product Name",     # Data is groupedby this variable
            hue="Country",
            y= "Quantity",          
            data=spc)
# top selling products by countries (in US)
df.columns
sale_Products_Country = df.groupby(["Product Name","Country"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)
sale_Products_Country = df.groupby(["Product Name","Country"])['Quantity'].sum().reset_index().sort_values('Quantity',ascending = False)
sale_Products_Country
type(sale_Products_Country)
spc = sale_Products_Country[sale_Products_Country['Country'] == "United States"].sort_values('Quantity',ascending = False).head(20)
sns.barplot(x = "Product Name",     # Data is groupedby this variable
            hue="Country",
            y= "Quantity",          
            data=spc)

# sales by product Category, Sub-category
sale_category = df.groupby(["Category","Sub-Category"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)
sale_category
sns.barplot(x = "Category",     # Data is groupedby this variable
            hue="Sub-Category",
            y= "Quantity",          
            data=sale_category)
#==========================================================================================
#Data Visualization
plt.figure(figsize=(16,8))
df['Market'].value_counts().plot.bar()
plt.title('Market Wise Sales')
plt.ylabel('Count')
plt.xlabel('Market Region')
plt.show()
#Top 20 Countries in sales
plt.figure(figsize=(16,8))
top20countries = df.groupby('Country')['Row ID'].count().sort_values(ascending=False)
top20countries = top20countries [:20]
top20countries.plot(kind='bar', color='green')
plt.title('Top 20 Countries in Sales')
plt.ylabel('Count')
plt.xlabel('Countries')
plt.show()
# United States as a Country tops all the Countries in Sales
#Top 20 States in Sales
plt.figure(figsize=(16,8))
top20states = df.groupby('State')['Row ID'].count().sort_values(ascending=False)
top20states = top20states [:20]
top20states.plot(kind='bar', color='blue')
plt.title('Top 20 States in Sales')
plt.ylabel('Count')
plt.xlabel('States')
plt.show()
# California as a State tops all the States in Sales

#==========================================================================================
#Customers visited only once Which customers have visited this store just once
Visit=df.groupby('Customer ID').apply(lambda x: pd.Series(dict(visit_count=x.shape[0])))
Visit.loc[(Visit.visit_count==1)]