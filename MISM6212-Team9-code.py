
###############################################
##########Group Project Code###################
''' 
MISM 6212 Final Project: Sharing Bike Demand in Seoul
Team 9: Haozhe Liu, Xuanjie Ma, Ruji Zhang, Shiming Zhao
'''
    
###############################################
##########Data profiling and cleasing##########

#Importing the needed packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importing the SeoulBikeData dataset
df = pd.read_csv("SeoulBikeData.csv", encoding= 'unicode_escape')

#check data profile
df.shape
# (8760, 14)

#learn about the content of data
df.head()

#The name and data type of each data column
df.info()
''' Features
- Date
- Rented Bike Count
- Hour
- Temperature(°C)
- Humidity(%)
- Wind speed 
- Visibility
- Dew point temperature(°C)
- Solar Radiation(MJ/m2)
- Rainfall(mm)
- Snowfall (cm)
- Seasons
- Holiday
- Functioning Day
'''

#The number of missing values of each column
df.isnull().sum()
# No missing value

#Before changing the value in each columns, we check their unique value of them
cols = df.columns
# for each column
for col in cols:
    print(col)
    # get a list of unique values
    unique = df[col].unique()

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
    if len(unique)<30:
        print(unique, '\n====================================\n\n')
    else:
        print(str(len(unique)) + ' unique values', '\n====================================\n\n')

#The number of fully duplicated data rows.
df.duplicated()
# Selecting duplicate based on all columns
duplicate = df[df.duplicated()]
print("Duplicate Rows :")
duplicate
#No duplicated rows

#formatted the column names in lowercase and replaced spaces with underscores
df.rename(columns=lambda x:x.lower().strip().replace(' ','_'),inplace=True) 

#After changing the value of some columns, we check their unque value again
cols = df.columns
# for each column
for col in cols:
    print(col)
    # get a list of unique values
    unique = df[col].unique()

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values
    if len(unique)<30:
        print(unique, '\n====================================\n\n')
    else:
        print(str(len(unique)) + ' unique values', '\n====================================\n\n')




###############################################
##########Data Visualization###################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##heatmap for overview the big picture 
df_heatmap = df.drop(columns = ["date"])
correlation=df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cmap='Greens',annot=True, square=True)

#scatterplots for all the numerical variables 
plt.scatter(df["temperature(°c)"], df["rented_bike_count"],c="green")
plt.xlabel("temperature(°c)")
plt.ylabel("rented_bike_count")
plt.title("The relationship between temperature and number of bikes rented", y=1.02)

plt.scatter(df["humidity(%)"], df["rented_bike_count"],c="green")
plt.xlabel("humidity(%)")
plt.ylabel("rented_bike_count")
plt.title("The relationship between humidity of weather and number of bikes rented", y=1.02)

plt.scatter(df["wind_speed_(m/s)"], df["rented_bike_count"],c="green")
plt.xlabel("wind_speed_(m/s)")
plt.ylabel("rented_bike_count")
plt.title("The relationship between wind speed and number of bikes rented", y=1.02)

plt.scatter(df["visibility_(10m)"], df["rented_bike_count"],c="green")
plt.xlabel("visibility_(10m)")
plt.ylabel("rented_bike_count")
plt.title("The relationship between visibility and number of bikes rented", y=1.02)

plt.scatter(df["solar_radiation_(mj/m2)"], df["rented_bike_count"],c="green")
plt.xlabel("solar_radiation_(mj/m2)")
plt.ylabel("rented_bike_count")
plt.title("The relationship between solar radiation and number of bikes rented", y=1.02)

plt.scatter(df["rainfall(mm)"], df["rented_bike_count"],c="green")
plt.xlabel("rainfall(mm)")
plt.ylabel("rented_bike_count")
plt.title("The relationship between rainfall and number of bikes rented", y=1.02)

plt.scatter(df["snowfall_(cm)"], df["rented_bike_count"],c="green")
plt.xlabel("snowfall_(cm)")
plt.ylabel("rented_bike_count")
plt.title("The relationship between snowfall and number of bikes rented", y=1.02)

plt.scatter(df["temperature(°c)"], df["dew_point_temperature(°c)"],c="green")
plt.xlabel("temperature(°c)")
plt.ylabel("dew_point_temperature(°c)")
plt.title("The relationship between temperature and dew_point_temperature", y=1.02)

#Bar chart (categorical variables) and line chart 
#Season VS Rented Bikes
sns.barplot(data=df, x='seasons', y='rented_bike_count')
#Holiday VS Rented Bikes
sns.barplot(data=df, x='holiday', y='rented_bike_count')

#Convert the Date column in Datetime Dtype
df_time = df
df_time['date']=pd.to_datetime(df_time['date'])
#Breaking Down the Date into 3 Components
df_time['Day']=df_time['date'].dt.day
df_time['Month']=df_time['date'].dt.month
df_time['Year']=df_time['date'].dt.year

#Year VS Rented Bikes
sns.barplot(data=df, x='Year', y='rented_bike_count')

#hour VS Rented Bikes
sns.pointplot(data=df, x="hour", y="rented_bike_count")


###############################################
##########Model Analysis-1#####################

#Importing the needed packages
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor 

from sklearn import metrics
from sklearn.metrics import(confusion_matrix,precision_score,recall_score,accuracy_score,f1_score)
from sklearn.ensemble import RandomForestRegressor

''' Data Preprocessing'''
df1 = df.drop(columns = ["date"])
df1_dummy = pd.get_dummies(df1)
df1_dummy.columns
x= df1_dummy.drop(columns = "rented_bike_count")
y= df1_dummy["rented_bike_count"]
#split the data into training data (70%) and testing data (30%). 
#We train the model on training data and see its performance on the testing data. 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state = 1)


'''Decision Tree'''
dt = DecisionTreeRegressor(random_state = 1)
dt.fit(x_train,y_train) 
###train
dt_pred_train = dt.predict(x_train)
rmse_train = mean_squared_error(dt_pred_train, y_train,squared = False)
###test
dt_pred_test = dt.predict(x_test)
rmse_test = mean_squared_error(dt_pred_test, y_test,squared = False)

#Measure the accuracy of decision tree
from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(dt_pred_test, y_test))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(dt_pred_test, y_test))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(dt_pred_test, y_test))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(dt_pred_test, y_test)))
#R^2: 0.744
#(MAE)Mean Absolute Error: 192.260
#(MSE)Mean Squared Error: 104818.140
#(RMSE)Root Mean Squared Error: 323.756

#Draw the scatter plot between predicted and actual value
plt.scatter(dt_pred_test,y_test,color='b')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Decision Tree")


'''Random Forest'''
rfc = RandomForestRegressor(random_state = 1,n_estimators =50)
rfc.fit(x_train,y_train)
###training 
rfc_pred_train=rfc.predict(x_train)
metrics.r2_score(y_train,rfc_pred_train)
###testing 
rfc_pred_test=rfc.predict(x_test)
metrics.r2_score(y_test,rfc_pred_test)

#Measure the accuracy of Random forecast¶
from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(y_test, rfc_pred_test))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(y_test, rfc_pred_test))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test, rfc_pred_test))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(y_test, rfc_pred_test)))
#R^2: 0.868
#(MAE)Mean Absolute Error: 140.583
#(MSE)Mean Squared Error: 53557.956
#(RMSE)Root Mean Squared Error: 231.426

#Draw the scatter plot between predicted and actual value
plt.scatter(rfc_pred_test,y_test,color='orange')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Random Forest")


'''Multiple Linear Regression'''
lr = LinearRegression()
###train
lr.fit(x_train,y_train)
###test
lr_pred_test = lr.predict(x_test)

#Measure the accuracy of Random forecast¶
from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(y_test,lr_pred_test))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(y_test,lr_pred_test))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test,lr_pred_test))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(y_test,lr_pred_test)))
#R^2: 0.551
#(MAE)Mean Absolute Error: 319.708
#(MSE)Mean Squared Error: 182073.535
#(RMSE)Root Mean Squared Error: 426.701

#Draw the scatter plot between predicted and actual value
plt.scatter(lr_pred_test,y_test,color='red')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Multipule Linear Regression")






###############################################
##########Model Analysis-2#####################
'''Model Improvement'''
df = pd.read_csv("SeoulBikeData.csv", encoding= 'unicode_escape')
#rename the column and capitalize the first letter
df2 = df
df2.columns=['Date','Rent_Bike_Count', 'Hour', 'Temperature', 'Humidity', 
            'Wind_speed', 'Visibility', 'DPT',
            'Solar_Radiation', 'Rainfall', 'Snowfall', 'Seasons',
            'Holiday', 'Functioning_Day']

##1
#Convert the Date column in Datetime Dtype
df2['Date']=pd.to_datetime(df2['Date'])
#Breaking Down the Date into 3 Components
df2['Day']=df2['Date'].dt.day
df2['Month']=df2['Date'].dt.month
df2['Year']=df2['Date'].dt.year

##2
#Converting Contionus variable to categorical variable for ease in prediction
df2['IsVisibility']=df2['Visibility'].apply(lambda x: 1 if x>=2000 else 0)
df2['IsRainfall']=df2['Rainfall'].apply(lambda x:1 if x>=0.148687 else 0)
df2['IsSnowfall']=df2['Snowfall'].apply(lambda x:1 if x>=0.075068 else 0)
df2['IsSolar_Radiation']=df2['Solar_Radiation'].apply(lambda x:1 if x>=0.56911 else 0)

##3
#Mapping the Variables
df2['Functioning_Day']=df2['Functioning_Day'].map({'Yes':1,'No':0})
df2['IsHoliday']=df2['Holiday'].map({'No Holiday':0,'Holiday':1})

##4
#Ater Conversion of numerical variable to categorical droping the original columns to avoid ambiguity
df2.drop(['Date','Visibility','Rainfall','Snowfall','Solar_Radiation','Holiday'],axis=1,inplace=True)

##5
#Since there was no bike rented in Non Functioning Day we gonna drop the rows 
df2=df2[df2['Functioning_Day']!=0]

##6
#Finally, we transform the Season column into dummy variables. 
Seasons=pd.get_dummies(df2['Seasons'],drop_first=True)
final_df=df2.append(Seasons)
final_df.fillna(0,inplace=True)
final_df.drop('Seasons',axis=1,inplace=True)
#################################################################




#Training Test Split
x=final_df.drop(columns = 'Rent_Bike_Count')
y=final_df['Rent_Bike_Count']
#split the data into training data (70%) and testing data (30%). 
#We train the model on training data and see its performance on the testing data. 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
    



'''Multiple Linear Regression'''
lr = LinearRegression()
###train
lr.fit(x_train,y_train)
###test
lr_pred_test = lr.predict(x_test)

#Measure the accuracy of Random forecast¶
from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(y_test,lr_pred_test))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(y_test,lr_pred_test))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test,lr_pred_test))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(y_test,lr_pred_test)))
#R^2: 0.714
#(MAE)Mean Absolute Error: 159.630
#(MSE)Mean Squared Error: 96164.219
#(RMSE)Root Mean Squared Error: 310.104

#Draw the scatter plot between predicted and actual value
plt.scatter(lr_pred_test,y_test,color='red')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Multipule Linear Regression-Improvement")

'''backward selection method'''
model_b = LinearRegression()
sfs_b = SFS(model_b, 
          k_features=(1,17), 
          forward=False, 
          scoring='neg_root_mean_squared_error',
          cv=5)
sfs_b.fit(x_train, y_train)
sfs_b.k_feature_names_
#the column number:('0', '1', '2', '5', '7', '8', '9', '10', '11', '13', '14', '16')


X_train_sfs = sfs_b.transform(x_train)
X_test_sfs = sfs_b.transform(x_test)
model_b.fit(X_train_sfs, y_train)
y_pred = model_b.predict(X_test_sfs)

from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(y_test,y_pred))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(y_test,y_pred))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test,y_pred))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#R^2: 0.713
#(MAE)Mean Absolute Error: 159.984
#(MSE)Mean Squared Error: 96386.729
#(RMSE)Root Mean Squared Error: 310.462

'''forward selection method'''
model_f = LinearRegression()
sfs_f = SFS(model_f, 
          k_features=(1,17), 
          forward=True, 
          scoring='neg_root_mean_squared_error',
          cv=5)
sfs_f.fit(x_train, y_train)
sfs_f.k_feature_names_
# the column number:('0', '1', '2', '4', '5', '7', '8', '9', '10', '11', '13', '14', '15', '16')

X_train_sfs = sfs_f.transform(x_train)
X_test_sfs = sfs_f.transform(x_test)

model_f.fit(X_train_sfs, y_train)
y_pred = model_f.predict(X_test_sfs)

from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(y_test,y_pred))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(y_test,y_pred))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test,y_pred))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#R^2: 0.713
#(MAE)Mean Absolute Error: 159.923
#(MSE)Mean Squared Error: 96352.106
#(RMSE)Root Mean Squared Error: 310.406




'''Decision Tree'''
dt = DecisionTreeRegressor(random_state = 1)
dt.fit(x_train,y_train) 
###train
dt_pred_train = dt.predict(x_train)
rmse_train = mean_squared_error(dt_pred_train, y_train,squared = False)
###test
dt_pred_test = dt.predict(x_test)
rmse_test = mean_squared_error(dt_pred_test, y_test,squared = False)

#Measure the accuracy of decision tree
from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(dt_pred_test, y_test))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(dt_pred_test, y_test))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(dt_pred_test, y_test))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(dt_pred_test, y_test)))
#R^2: 0.838
#(MAE)Mean Absolute Error: 95.881
#(MSE)Mean Squared Error: 54321.763
#(RMSE)Root Mean Squared Error: 233.070

#Draw the scatter plot between predicted and actual value
plt.scatter(dt_pred_test,y_test,color='b')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Decision Tree-Improvement")





'''Random Forest'''
rfc = RandomForestRegressor(random_state = 1,n_estimators =50)
rfc.fit(x_train,y_train)
###training 
rfc_pred_train=rfc.predict(x_train)
metrics.r2_score(y_train,rfc_pred_train)
###testing 
rfc_pred_test=rfc.predict(x_test)
metrics.r2_score(y_test,rfc_pred_test)

#Measure the accuracy of Random forecast¶
from sklearn.metrics import r2_score
print('R^2: %.3f' % r2_score(y_test, rfc_pred_test))
print('(MAE)Mean Absolute Error: %.3f' % metrics.mean_absolute_error(y_test, rfc_pred_test))
print('(MSE)Mean Squared Error: %.3f' % metrics.mean_squared_error(y_test, rfc_pred_test))
print('(RMSE)Root Mean Squared Error: %.3f' % np.sqrt(metrics.mean_squared_error(y_test, rfc_pred_test)))
#R^2: 0.915
#(MAE)Mean Absolute Error: 72.640
#(MSE)Mean Squared Error: 28610.641
#(RMSE)Root Mean Squared Error: 169.147

#Draw the scatter plot between predicted and actual value
plt.scatter(rfc_pred_test,y_test,color='orange')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Random Forest-Improvement")






