#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[281]:


import pandas as pd
import numpy as np
import pylab as plt


# In[3]:


#new_header = df.iloc[0] #grab the first row for the header
#df = df[1:] #take the data less the header row
#df.columns = new_header #set the header row as the df header


# # Cleaning

# In[282]:


df9 = pd.read_csv("./S07/datasets/UN/SYB62_T30_201904_Tourist-Visitors Arrival and Expenditure.csv", encoding = "ISO-8859-1", thousands=',')
new_header=df9.iloc[0]
df9=df9[1:]
df9.columns=new_header
#df9=df9.drop('Footnotes',1)
#df9=df9.drop('Source',1)
#df9=df9.drop('Tourism arrivals series type footnote',1)


df9=df9.rename(columns={np.nan:'Country'})
df9=df9.rename(columns={'Value':'Tourist'})
df9=df9[['Country','Year','Series','Tourist']]
#df9
#print(df9.columns)

#df91=df9.loc[df9['Series'] == 'Tourism expenditure (millions of US dollars)']
#df91=df91.rename(columns={'Tourist':'Tourism expenditure (millions of US dollars)'})
#df91=df91.drop('Series',1)
#df91=df91.reset_index()

df92=df9.loc[df9['Series'] == 'Tourist/visitor arrivals (thousands)']
df92=df92.rename(columns={'Tourist':'Tourist/visitor arrivals (thousands)'})
df92=df92.drop('Series',1)
df92=df92.reset_index(drop=True)
df92.shape

#df91
#df92
#df92


# In[ ]:





# In[5]:





# In[283]:


df8 = pd.read_csv("./S07/datasets/UN/SYB62_T29_201904_Internet Usage.csv", encoding = "ISO-8859-1",thousands=',')
new_header=df8.iloc[0]
df8=df8[1:]
df8.columns=new_header

#df8=df8.drop('Footnotes',1)                                     ##Totall countries dare
#df8=df8.drop('Source',1)

df8=df8.rename(columns={np.nan:'Country'})
df8=df8.rename(columns={'Value':'Percentage of individuals using the internet'})
df8=df8[['Country','Year','Percentage of individuals using the internet']]
df8=df8.reset_index(drop=True)
df8.shape

#print(df8.columns)
#df8

#df8


# In[128]:





# In[ ]:





# In[284]:


df7 = pd.read_csv("./S07/datasets/UN/SYB62_T13_201904_GDP and GDP Per Capita.csv", encoding = "ISO-8859-1" ,thousands=',')
new_header=df7.iloc[0]
df7=df7[1:]
df7.columns=new_header


#df7

df7=df7.rename(columns={np.nan:'Country'})

df7=df7[['Country','Year','Series','Value']]


#df71=df7.loc[df7['Series'] == 'GDP in current prices (millions of US dollars)']
#df71=df71.rename(columns={'Value':'GDP in current prices (millions of US dollars)'})
#df71=df71.drop('Series',1)
#df71=df71.reset_index()

df72=df7.loc[df7['Series'] == 'GDP per capita (US dollars)']
df72=df72.rename(columns={'Value':'GDP per capita (US dollars)'})
df72=df72.drop('Series',1)
df72=df72.reset_index(drop=True)

df72.shape

#df73=df7.loc[df7['Series'] == 'GDP in constant 2010 prices (millions of US dollars)']
#df73=df73.rename(columns={'Value':'GDP in constant 2010 prices (millions of US dollars)'})
#df73=df73.drop('Series',1)
#df73=df73.reset_index()

#df74=df7.loc[df7['Series'] == 'GDP real rates of growth (percent)']
#df74=df74.rename(columns={'Value':'GDP real rates of growth (percent)'})
#df74=df74.drop('Series',1)
#df74=df74.reset_index()

#df74

#GDP in current prices (millions of US dollars)
#GDP per capita (US dollars)
#GDP inGDP in constant 2010 prices (millions of US dollars)
#GDP real rates of growth (percent)


# In[ ]:





# In[285]:


df6 = pd.read_csv("./S07/datasets/UN/SYB62_T12_201904_Intentional homicides and Other Crimes.csv", encoding = "ISO-8859-1",thousands=',')
new_header=df6.iloc[0]
df6=df6[1:]
df6.columns=new_header

#df6=df6.drop('Footnotes',1)
#df6=df6.drop('Source',1)

#df6

df6=df6.rename(columns={np.nan:'Country'})

df6=df6[['Country','Year','Series','Value']]

print(df6.columns)
df61=df6.loc[df6['Series'] == 'Intentional homicide rates per 100,000']
df61=df61.rename(columns={'Value':'Intentional homicide rates per 100,000'})
df61=df61.drop('Series',1)
df61=df61.reset_index(drop=True)

df61.shape


#df62=df6.loc[df6['Series'] == 'Percentage of male and female intentional homicide victims, Male']
#=df62.rename(columns={'Value':'Percentage of male and female intentional homicide victims, Male'})
#df62=df62.drop('Series',1)
#df62=df62.reset_index()

#error#df63=df6.loc[df6['Series'] == 'Percentage of male and female intentional homicide victims, Female']
      #df63=df63.rename(columns={'Percentage of male and female intentional homicide victims, Female'})
      #df63=df63.drop('Series',1)
      #df63=df63.reset_index()

#df64=df6.loc[df6['Series'] == 'Assault rate per 100,000 population']
#df64=df64.rename(columns={'Value':'Assault rate per 100,000 population'})
#df64=df64.drop('Series',1)
#df64=df64.reset_index()

df65=df6.loc[df6['Series'] == 'Theft at the national level, rate per 100,000 population']
df65=df65.rename(columns={'Value':'Theft at the national level, rate per 100,000 population'})
df65=df65.drop('Series',1)
df65=df65.reset_index(drop=True)

#df66=df6.loc[df6['Series'] == 'Robbery at the national level, rate per 100,000 population']
#df66=df66.rename(columns={'Value':'Robbery at the national level, rate per 100,000 population'})
#df66=df66.drop('Series',1)
#df66=df66.reset_index()


#df67=df6.loc[df6['Series'] == 'Total Sexual Violence at the national level, rate per 100,000']
#df67=df67.rename(columns={'Value':'Total Sexual Violence at the national level, rate per 100,000'})
#df67=df67.drop('Series',1)
#df67=df67.reset_index()

#df68=df6.loc[df6['Series'] == 'Kidnapping at the national level, rate per 100,000']
#df68=df68.rename(columns={'Value':'Kidnapping at the national level, rate per 100,000'})
#df68=df68.drop('Series',1)
#df68=df68.reset_index()

#Intentional homicide rates per 100,000
#Percentage of male and female intentional homicide victims, Male
#Percentage of male and female intentional homicide victims, Female
#Assault rate per 100,000 population
#Theft at the national level, rate per 100,000 population
#Robbery at the national level, rate per 100,000 population
#Total Sexual Violence at the national level, rate per 100,000
#Kidnapping at the national level, rate per 100,000

df61.shape


# In[286]:


df5 = pd.read_csv("./S07/datasets/UN/SYB62_T09_201905_Public Expenditure on Education.csv", encoding = "ISO-8859-1",thousands=',')
new_header=df5.iloc[0]
df5=df5[1:]
df5.columns=new_header

#df5=df5.drop('Footnotes',1)
#df5=df5.drop('Source',1)


df5=df5.rename(columns={np.nan:'Country'})
df5=df5[['Country','Year','Series','Value']]

print(df5.columns)


df51=df5.loc[df5['Series'] == 'Current expenditure other than staff compensation as % of total expenditure in public institutions']
df51=df51.rename(columns={'Value':'Current expenditure other than staff compensation as % of total expenditure in public institutions'})
df51=df51.drop('Series',1)
df51=df51.reset_index()

#df52=df5.loc[df5['Series'] == 'All staff compensation as % of total expenditure in public institutions (%)']
#df52=df52.rename(columns={'Value':'All staff compensation as % of total expenditure in public institutions (%)'})
#df52=df52.drop('Series',1)
#df52=df52.reset_index()

#df53=df5.loc[df5['Series'] == 'Capital expenditure as % of total expenditure in public institutions (%)']
#df53=df53.rename(columns={'Value':'Capital expenditure as % of total expenditure in public institutions (%)'})
#df53=df53.drop('Series',1)
#df53=df53.reset_index()

df54=df5.loc[df5['Series'] == 'Expenditure by level of education: primary (as % of government expenditure)']
df54=df54.rename(columns={'Value':'Expenditure by level of education: primary (as % of government expenditure)'})
df54=df54.drop('Series',1)
df54=df54.reset_index()

#df55=df5.loc[df5['Series'] == 'Expenditure by level of education: secondary (as % of government expenditure)']
#df55=df55.rename(columns={'Value':'Expenditure by level of education: secondary (as % of government expenditure)'})
#df55=df55.drop('Series',1)
#df55=df55.reset_index()

#df56=df5.loc[df5['Series'] == 'Expenditure by level of education: tertiary (as % of government expenditure)']
#df56=df56.rename(columns={'Value':'Expenditure by level of education: tertiary (as % of government expenditure)'})
#df56=df56.drop('Series',1)
#df56=df56.reset_index()

df57=df5.loc[df5['Series'] == 'Public expenditure on education (% of government expenditure)']
df57=df57.rename(columns={'Value':'Public expenditure on education (% of government expenditure)'})
df57=df57.drop('Series',1)
df57=df57.reset_index()

df58=df5.loc[df5['Series'] == 'Public expenditure on education (% of GDP)']
df58=df58.rename(columns={'Value':'Public expenditure on education (% of GDP)'})
df58=df58.drop('Series',1)
df58=df58.reset_index(drop=True)

df58.shape


#Current expenditure other than staff compensation as % of total expenditure in public institutions (%)
#All staff compensation as % of total expenditure in public institutions (%)
#Capital expenditure as % of total expenditure in public institutions (%)
#Expenditure by level of education: primary (as % of government expenditure)
#Expenditure by level of education: secondary (as % of government expenditure)
#Expenditure by level of education: tertiary (as % of government expenditure)
#Public expenditure on education (% of government expenditure)
#Public expenditure on education (% of GDP)



# In[287]:


df4 = pd.read_csv("./S07/datasets/UN/SYB61_T21_Total Imports, Exports and Balance of Trade.csv", encoding = "ISO-8859-1",thousands=',')
new_header=df4.iloc[0]
df4=df4[1:]
df4.columns=new_header

#df4=df4.drop('Footnotes',1)
#df4=df4.drop('Source',1)
#df4=df4.drop('System of trade footnote',1)


df4=df4.rename(columns={np.nan:'Country'})
df4=df4[['Country','Year','Series','Value']]

#print(df4.columns)


#df41=df4.loc[df4['Series'] == 'Imports CIF (millions of US dollars)']
#df41=df41.rename(columns={'Value':'Imports CIF (millions of US dollars)'})
#df41=df41.drop('Series',1)
#df41=df41.reset_index()

df42=df4.loc[df4['Series'] == 'Exports FOB (millions of US dollars)']
df42=df42.rename(columns={'Value':'Exports FOB (millions of US dollars)'})
df42=df42.drop('Series',1)
df42=df42.reset_index(drop=True)


df43=df4.loc[df4['Series'] == 'Balance imports/exports (millions of US dollars)']
df43=df43.rename(columns={'Value':'Balance imports/exports (millions of US dollars)'})
df43=df43.drop('Series',1)
df43=df43.reset_index(drop=True)
df42.shape


#Imports CIF (millions of US dollars)
#Exports FOB (millions of US dollars)
#Balance imports/exports (millions of US dollars)


# In[288]:


df3 = pd.read_csv("./S07/datasets/UN/SYB61_T19_Consumer Price Index.csv", encoding = "ISO-8859-1",thousands=',')
new_header=df3.iloc[0]
df3=df3[1:]
df3.columns=new_header


df3=df3.rename(columns={np.nan:'Country'})
df3=df3[['Country','Year','Series','Value']]

#print(df3.columns)

df31=df3.loc[df3['Series'] == 'Consumer price index: General']
df31=df31.rename(columns={'Value':'Consumer price index: General'})
df31=df31.drop('Series',1)
df31=df31.reset_index(drop=True)

df31.shape

df32=df3.loc[df3['Series'] == 'Consumer price index: Food']
df32=df32.rename(columns={'Value':'Consumer price index: Food'})
df32=df32.drop('Series',1)
df32=df32.reset_index()
#df32.shape
#Consumer price index: General
#Consumer price index: Food


# In[289]:


df2 = pd.read_csv("./S07/datasets/UN/SYB61_T04_International Migrants and Refugees.csv", encoding = "ISO-8859-1",thousands=',')

new_header=df2.iloc[0]
df2=df2[1:]
df2.columns=new_header

df2=df2.rename(columns={np.nan:'Country'})
df2=df2[['Country','Year','Series','Value']]

df21=df2.loc[df2['Series'] == 'International migrant stock: Both sexes (number)']
df21=df21.rename(columns={'Value':'International migrant stock/Both sexes (number)'})
df21=df21.drop('Series',1)
df21=df21.reset_index(drop=True)

df21.shape

#df22=df2.loc[df2['Series'] == 'International migrant stock: Both sexes (% total population)']
#df22=df22.rename(columns={'Value':'International migrant stock: Both sexes (% total population)'})
#df22=df22.drop('Series',1)
#df22=df22.reset_index()

#df23=df2.loc[df2['Series'] == 'International migrant stock: Male (% total Population)']
#df23=df23.rename(columns={'Value':'International migrant stock: Male (% total Population)'})
#df23=df23.drop('Series',1)
#df23=df23.reset_index()

#df24=df2.loc[df2['Series'] == 'International migrant stock: Female (% total Population)']
#df24=df24.rename(columns={'Value':'International migrant stock: Female (% total Population)'})
#df24=df24.drop('Series',1)
#df24=df24.reset_index()

#df25=df2.loc[df2['Series'] == 'Total refugees and people in refugee-like situations (number)']
#df25=df25.rename(columns={'Value':'Total refugees and people in refugee-like situations (number)'})
#df25=df25.drop('Series',1)
#df25=df25.reset_index()

#df26=df2.loc[df2['Series'] == 'Asylum seekers, including pending cases (number)']
#df26=df26.rename(columns={'Value':'Asylum seekers, including pending cases (number)'})
#df26=df26.drop('Series',1)
#df26=df26.reset_index()

#df27=df2.loc[df2['Series'] == 'Other of concern to UNHCR (number)']
#df27=df27.rename(columns={'Value':'Other of concern to UNHCR (number)'})
#df27=df27.drop('Series',1)
#df21=df27.reset_index()

#df28=df2.loc[df2['Series'] == 'Total population of concern to UNHCR (number)']
#df28=df28.rename(columns={'Total population of concern to UNHCR (number)'})
#df28=df28.drop('Series',1)
#df28=df28.reset_index()


#International migrant stock: Both sexes (number)
#International migrant stock: Both sexes (% total population)
#International migrant stock: Male (% total Population)
#International migrant stock: Female (% total Population)
#Total refugees and people in refugee-like situations (number)
#Asylum seekers, including pending cases (number)
#Other of concern to UNHCR (number)
#Total population of concern to UNHCR (number)


# In[290]:


df1 = pd.read_csv("./S07/datasets/UN/SYB58_T25 Index of industrial production.csv", encoding = "ISO-8859-1",thousands=',')

new_header=df1.iloc[0]
df1=df1[1:]
df1.columns=new_header

df1=df1.rename(columns={np.nan:'Country'})
df1=df1[['Country','Year','Series','Value']]

df11=df1.loc[df1['Series'] == 'Index of industrial production: Total industry - Mining; manufacturing; electricity, gas and water (Index base: 2005=100)']
df11=df11.rename(columns={'Value':'Index of industrial production: Total industry'})
df11=df11.drop('Series',1)
df11=df11.reset_index(drop=True)


#Index of industrial production: Total industry - Mining; manufacturing; electricity, gas and water (Index base: 2005=100)
#Index of industrial production: Mining (Index base: 2005=100)
#Index of industrial production: Manufacturing (Index base: 2005=100)
#Index of industrial production: Food, beverages and tobacco (Index base: 2005=100)
#Index of industrial production: Textiles, wearing apparel, leather, footwear (Index base: 2005=100)
#Index of industrial production: Chemicals, petroleum, rubber and plastic products (Index base: 2005=100)
#Index of industrial production: Metal products and machinery (Index Base: 2005=100)
#Index of industrial production: Electricity, gas, steam (Index base: 2005=100)
#Index of industrial production: Water and waste management (Index base: 2005=100)


# # Just for Test

# In[40]:



#dftest=pd.merge(df1,df2,how='outer',on='Unnamed: 2')


# In[98]:




#dftest=pd.merge(df8,df92,how='outer',on=['Country','Year'])
#dftest

#pd.DataFrame.to_csv(df_merged, 'merged.txt', sep=',', na_rep='.', index=False)
#writer = csv.writer(open("C:/Users/Arghavan/Desktop/S07/S07/csv/file", 'w'))


# 

# In[96]:


#dfcpi = df8.merge(df92,on=['Country', 'Year'])
#dfcpi
#plt.plot(dfcpi)
#plt.show()
#dfcpi=dfcpi.astype(float)
#dfcpi['Tourist/visitor arrivals (thousands)'] = dfcpi['Tourist/visitor arrivals (thousands)'].str.replace(',', '').astype(float)
#dfcpi['Percentage of individuals using the internet'] = dfcpi['Percentage of individuals using the internet'].str.replace(',', '').astype(float)
#dfcpi.astype({'Internet': 'float','Tourist':'float'}).dtypes
#dfcpi.plot(x='Internet',y='Tourist', style='-')

#x=dfcpi['Tourist/visitor arrivals (thousands)'].values
#y=dfcpi['Percentage of individuals using the internet'].values

#matrix=np.corrcoef(x,y)

#print(matrix)

#Tlist=list(dfcpi['Tourist'])
#mean=Tlist.mean()


# In[ ]:


#SYB62_T12_201904_Intentional homicides and Other Crimes.csv
#Unnamed 4=value
#a=df.isna().sum(axis = 0)
#a


# In[12]:


#b=df.isna()
#b


# # Merging Dataframes

# In[291]:


from functools import reduce
data_frames = [df72,df43, df8, df92,df61,df65,df58,df31,df21,df11]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Country','Year'],
                                            how='outer'), data_frames)
df_merged

df_mergedin= reduce(lambda  left,right: pd.merge(left,right,on=['Country','Year'],
                                            how='inner'), data_frames)

#a=df_merged.isna().sum(axis = 0)

df=df_merged

#df=df.dropna(thresh=11)
#df=df.dropna(subset=['GDP per capita (US dollars)'],inplace=True)
#df=df.reset_index(drop=True)
df=df[df['GDP per capita (US dollars)'].notnull()]
df=df[df['International migrant stock/Both sexes (number)'].notnull()]
dfc=df[['Country','Year']]

dfnan = pd.concat([dfc,dfO], axis=1).reindex(dfc.index)
dfnan.to_csv('nan.csv')
#df=df.dropna(thresh=11, axis=0, inplace=True)
#DataFrameName.dropna(axis=0, how='any', thresh=2, subset=None, inplace=False)
#threshHold=2
#def find_rows_with_nan(df, threshHold):

    #return df.dropna(thresh=threshHold)


# In[292]:


a=df.isna().sum(axis = 0)
a


# # Correaltion Matrix

# In[293]:


data=df_mergedin
data = data.drop(columns=['Country','Year'])
cols = data.columns
nn = len(cols)
for i in range(nn):
    data[cols[i]]=data[cols[i]].str.replace(',','').astype(float)


correlation=data.corr(method ='pearson') 

#data
#correlation
#data.columns
correlation


# In[294]:


dfO = df.drop(columns=['Country','Year'])


# # Regression

# In[295]:


from sklearn import linear_model



# we just consider correlations more than 0.5


col = dfO.columns
n = len(col)
for i in range(n):
    dfO[cols[i]]=dfO[cols[i]].str.replace(',','').astype(float)


GDP = data[['GDP per capita (US dollars)']]#1~3&6
GDPO = dfO[['GDP per capita (US dollars)']]

Internet = data[['Percentage of individuals using the internet']]#3~1&6
InternetO = dfO[['Percentage of individuals using the internet']]

Theft = data[['Theft at the national level, rate per 100,000 population']]#6~1&3
TheftO = dfO[['Theft at the national level, rate per 100,000 population']]#6~1&3

Tourist = data[['Tourist/visitor arrivals (thousands)']]#4~9
Migrant = data[['International migrant stock/Both sexes (number)']]#9~4
MigrantO = dfO[['International migrant stock/Both sexes (number)']]#9~4

reg13 = linear_model.LinearRegression()
reg13.fit(GDP,Internet)

#reg49 = linear_model.LinearRegression()
#reg49.fit(Tourist,Migrant)

reg94 = linear_model.LinearRegression()
reg94.fit(Migrant,Tourist)

#reg36 = linear_model.LinearRegression()
#reg36.fit(Internet,Theft)

reg16 = linear_model.LinearRegression()
reg16.fit(GDP,Theft)

#Rows = dfO.iterrows

#for i in Rows:

InternetF=reg13.predict(GDPO)
columns=['Internet']
df3=pd.DataFrame(InternetF,columns=columns)

TheftF=reg16.predict(GDPO)
columns=['Theft']
df6=pd.DataFrame(TheftF,columns=columns)


TouristF=reg94.predict(MigrantO)
columns=['Tourist']
df4=pd.DataFrame(TouristF,columns=columns)





# In[296]:



df1=dfO[['GDP per capita (US dollars)']]
df9=dfO[['International migrant stock/Both sexes (number)']]


# In[297]:


df2=dfO[['Balance imports/exports (millions of US dollars)']]
df5=dfO[['Intentional homicide rates per 100,000']]
df7=dfO[['Public expenditure on education (% of GDP)']]
df8F=dfO[['Consumer price index: General']]
df10=dfO[['Index of industrial production: Total industry']]


# In[298]:


mean_value=dfO['Balance imports/exports (millions of US dollars)'].mean()
df2['Balance imports/exports (millions of US dollars)']=df2['Balance imports/exports (millions of US dollars)'].fillna(mean_value)

mean_value=dfO['Intentional homicide rates per 100,000'].mean()
df5['Intentional homicide rates per 100,000']=df5['Intentional homicide rates per 100,000'].fillna(mean_value)

median_value=dfO['Public expenditure on education (% of GDP)'].median()
df7['Public expenditure on education (% of GDP)']=df7['Public expenditure on education (% of GDP)'].fillna(median_value)

median_value=dfO['Public expenditure on education (% of GDP)'].median()
df8F['Consumer price index: General']=df8F['Consumer price index: General'].fillna(median_value)

mean_value=dfO['Index of industrial production: Total industry'].mean()
df10['Index of industrial production: Total industry']=df10['Index of industrial production: Total industry'].fillna(mean_value)


# In[299]:


result = pd.concat([df1, df2,df3,df4,df5,df6,df7,df8F,df9,df10], axis=1).reindex(df1.index)

result=result.rename(columns={"Internet": "Percentage of individuals using the internet", "Theft": "Theft at the national level, rate per 100,000 population","Tourist":"Tourist/visitor arrivals (thousands)"})


# In[300]:



dfO=dfO.reset_index(drop=True)
#dfO[dfO.isnull()] = result
result=result.reset_index(drop=True)


# In[301]:


dfO[dfO.isnull()]=result


# In[277]:





# In[302]:


Final = pd.concat([dfc,dfO], axis=1).reindex(dfc.index)


Final.to_csv('Final.csv')

#=result.reset_index(drop=True)


# In[303]:


Final


# In[ ]:


##########
حامد فراهانی

