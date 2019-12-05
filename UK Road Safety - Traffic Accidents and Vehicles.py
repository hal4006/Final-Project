#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Hide input.
from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
CSS = """#notebook div.output_subarea {max-width:100%;}"""
HTML('<style>{}</style>'.format(CSS))
# Hide warnings.
import warnings
warnings.filterwarnings('ignore')


# # UK Road Safety: Traffic Accidents and Vehicles
# [View Data Sourse Here](https://www.kaggle.com/tsiaras/uk-road-safety-accidents-and-vehicles)

# ## Abstract
# 
# This project is the exploration into the road safety of the United Kingdom from 2010 to 2016. based the traffic accident information and vehicle information, which are the two datasets that can be merged by the common variable, accident index. Besides the basic analysis methods, I also used heatmap, treemap, ratio calculation, etc to show the relationship between variables. The unsupervised methods used in the project was K-means, in order to give suggestions to the Transport Department in London City, to set the optimal quantity of traffic control stations to correspond to accidents better. According to the analysis, it is found that the accident distribution over weekdays and weekends are diffrent; one of the main reason why the proportion of serious or fatal accidents increased from 2014 is the increasement of illegal driving in teenagers under the age of 16. Finally, 6 points of centers was decided to set stations to promote the efficacy of traffic control.

# ## Introduction and Background
# 
# ### Background Information
# 
# The datasets were published by [Thanasis](https://www.kaggle.com/tsiaras) from [Kaggle](https://www.kaggle.com/). The data come from the [Open Data](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) website of the UK government, where they have been published by the Department of Transport.
# 
# These files provide detailed data about the circumstances of personal injury road accidents in Great Britain from 2005 onwards, the types of vehicles involved and the consequential casualties. The statistics relate only to personal injury accidents on public roads that are reported to the police, and subsequently recorded, using the STATS19 accident reporting form. Information on damage-only accidents, with no human casualties or accidents on private roads or car parks are not included in this data.
# 
# As well as giving details of date, time and location, the accident file gives a summary of all reported vehicles and pedestrians involved in road accidents and the total number of casualties, by severity. Details in the casualty and vehicle files can be linked to the relevant accident by the “Accident_Index” field. The Longitude and Latitude data is based on WGS 1984.
# 
# ### Content
# 
# There are two csv files:
# 
# ***Accident_Information.csv:*** every line in the file represents a unique traffic accident (identified by the Accident_Index column), featuring various properties related to the accident as columns. Date range: 2005-2017
# 
# ***Vehicle_Information.csv:*** every line in the file represents the involvement of a unique vehicle in a unique traffic accident, featuring various vehicle and passenger properties as columns. Date range: 2004-2016

# In[2]:


# Import the following libraries for analysis:
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import seaborn as sns
import squarify
sns.set()
from datetime import datetime as dt
from sklearn.cluster import KMeans


# In[3]:


# Load data sets:
acc = pd.read_csv('Accident_Information.csv', encoding = 'latin', low_memory = False)
veh = pd.read_csv('Vehicle_Information.csv', encoding = 'latin', low_memory = False)


# Check data samples for ***Accident Information:***

# In[4]:


acc.head()


# Check data samples for ***Vehicle Information:***

# In[5]:


veh.head()


# To be more focused, I will only look into the traffic accidents happened in the common years in 2010s, 2010-2016, for these two datasets. After filter the data, the records for traffic accidents and vehicles are as follow:

# In[6]:


# Filter accident information from 2010 to 2016
# acc = acc.copy()
acc = acc[(acc.Year >= 2010) & (acc.Year <= 2016)]
veh = veh[(veh.Year >= 2010) & (veh.Year <= 2016)]
print("Records of Accidents from 2010 to 2016:", acc.shape[0])
print("Records of Vehicles from 2010 to 2016:", veh.shape[0])


# ### Introduction of Questions
# 
# #### Question 1.
# 
# The first question is going to ***explore the periodic patterns of the accident data***. As there are several datetime variables in the Accident Information, such as *Date, Day_of_Week,* and *Time*, we can look into the accident distribution over the day, the week, and also the year, also considering other components, such as *Road_Surface_Conditions* at the same time.
# 
# #### Question 2.
# 
# The second question comes up based on the situation that ***the proportion of serious or fatal accidents began to increase from 2014***. Here, I would like to combine the two datasets together, and ***give a reason to such situation in terms of vehicle information***.
# 
# 
# #### Question 3.
# 
# The third question is about ***giving suggestions to the Transport Department in London City, to set the optimal quantity of traffic control stations to correspond to accidents better***. K-means clustering will be performed.

# ## Methods
# 
# ### Data Cleaning
# 
# The major data cleaning stuffs are as follows:
# 
# 1. Transform datetime format. For example, I subtracted Hours from Time and generated 5 traffic periods over the day for easier understanding of the analysis. Besides, I add the Month variable to the accident information for exploration to the accident distribution over the year.
# 
# 2. There are many categorical variables in both accident information and vehicle information. When I was exploring these variables, I would unique the column first to see the factors and filter out things like 'Data Missing'.
# 
# ### Python Libraries for Analysis
# 
# I basicly imported *numpy, pandas,* as well as *matplotlib* and *seaborn.* 
# 
# What's more, I imported *datetime* for date / time data processing in question 1, *squarify* for plot treemap in question 2, and *KMeans from sklearn.cluster* library for clustering job in question 3.
# 
# ### Analysis Methods
# 
# I used bar plots or lines to show the trend over time and the stacked bar plot to show the proportion of different stratification.
# 
# To combine two variables together, I used heatmap in Question 1; to compare the distribution of age band in Question 2, I calculated the ratio between 2013 and 2016 and displayed the table.
# 
# Finally, in Question 3, I used K-means clustering. To decide an ideal cluster value, I used Elbow Curve first to compare the efficacy of different cluster numbers.

# ## Results
# ### Question 1. Explore the periodic patterns of the accident data.
# #### a. Accident distribution over the day and the week.
# 1. According to Figure 1, there are two modes of the accident distribution over the day, the first one is at around 8:00, and the second is at around 17:00, both of which are in rush hour.
# 2. Looking into Figure 2, the numbers of accidents in Tuesday, Wednesday, and Thursday are quite close to each other. On Friday, the numbers of accidents has its maximum value in the distribution, but after that the number goes down and increased on Monday close to Tuesday.
# 3. Based on above, the number of traffic accidents has strong association with it whether the day is weekday or not or whether the time is in rush hours.
# 4. According to Figure 3, it is obvious that the accident distribution on weekdays are quite similar. On weekdays, afternoon rush is the traffic period that has the most traffic accidents, while it is much less likely that traffic accidents happen during night.
# 5. However, the rules do not work for weekends. On weekends, the distribution curve should be much more gentle than that of weekdays. The frequency of accidents during the afternoon rush decreased, while the probability of traffic accidents in night grows.

# In[7]:


# There are 3 plots included in this cell.

# 1. Distribution of accidents over the day.
plt.figure(figsize=(20, 5))
# Get Hour in which an accident happened from Time variable.
acc['Hour'] = pd.to_datetime(acc.Time, format = '%H:%M').dt.hour
# Draw traffic accident distribution over the day.
plt.subplot(1,3,1)
acc.Hour.value_counts().sort_index().plot(kind = 'area', alpha = 0.6)
plt.xlabel('Hours')
plt.ylabel('Number of Traffic Accidents')
plt.title('Figure 1. Accident Distribution over the Day', fontsize = 15)

# 2. Distribution of accidents over the week.
plt.subplot(1,3,2)
# Set the Day_of_Week variable in the weekday order.
acc.Day_of_Week = pd.Categorical(acc.Day_of_Week, ordered=True,
                                categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'])
# Draw traffic accident distribution over the week.
acc.Day_of_Week.value_counts().sort_index().plot.bar(alpha = 0.6)
pl.xticks(rotation = 360)
plt.ylabel('Number of Traffic Accidents')
plt.title('Figure 2. Accident Distribution over the Week', fontsize = 15)

# 3. Distribution of accidents over the week stratified by traffic period.
plt.subplot(1,3,3)
# Write a function to define the traffic period based on hour.
def traffic_period(hour):
    if hour >= 5 and hour < 10:
        return '1-morning rush'
    elif hour >= 10 and hour < 15:
        return '2-office hours'
    elif hour >= 15 and hour < 19:
        return '3-afternoon rush'
    elif hour >= 19 and hour < 23:
        return '4-evening'
    else:
        return '5-night'
# Add a new variable to the acc data set.
acc['Traffic_Period'] = acc.Hour.apply(traffic_period)
# Draw the plot.
day_of_week = acc.groupby(['Day_of_Week','Traffic_Period'])['Accident_Index'].count().unstack()
day_of_week_log = np.log10(day_of_week).round(2)
sns.heatmap(day_of_week_log, annot=True, cmap='Blues', alpha=0.9)
pl.xticks(rotation = 30)
plt.xlabel('')
plt.ylabel('')
plt.title('Figure 3. Heatmap for Accident Distribution (Log)', fontsize = 15)
plt.show()


# #### b. Accident distribution over the year in terms of road surface conditions.
# 1. According to Figure 4, November is the month that has the greatest scale of traffic accidents in total. The total level of accidents is higher in the second half year than the first half.
# 2. In terms of seasonal factors, accidents in each month were further divided by road surface conditions, which is the one not only related to weather condition but also has less factors for easier interpretation. From April to September, the distribution of road surface conditions are quite similar, as there is little possibility that it gets snowy.
# 3. Combine Figure 4 and 5, it can be concluded that road surface conditions related to weather conditions is the main reason that the numbers of traffic accidents keep in a relative high level in the second half year.
# 4. However, I have to admit that the Figures below do not make sense enough. The reason is that I failed to combine the accident information with the total traffic scale, so the risk of traffic in each month could not be calculated. This is the limitation of the analyasis.

# In[8]:


# There are 2 plots included in this cell.

# 1. Accident Distribution over the Year.
plt.figure(figsize=(18, 5))
plt.subplot(1,2,1)
# Subtract Month from variable Date.
acc['Month'] = pd.to_datetime(acc.Date, format = '%Y-%m-%d').dt.month
acc.Month.value_counts().sort_index().plot.bar(alpha=0.6)
pl.xticks(rotation = 360)
plt.ylabel('Number of Traffic Accidents')
plt.title('Figure 4. Accidents Distribution over the Year', fontsize = 15)

# 2. Proportion of accidents by road surface condition.
plt.subplot(1,2,2)
road_surface_conditions = acc.Road_Surface_Conditions.unique()[0:5]
month_and_road_surface = acc.groupby(['Month','Road_Surface_Conditions'])['Accident_Index'].count().unstack()[road_surface_conditions]
# Calculte proportions.
accident_month = month_and_road_surface.sum(axis=1)
for x in road_surface_conditions:
    month_and_road_surface[x] = month_and_road_surface[x]/accident_month
    month_and_road_surface[x] = round(month_and_road_surface[x], 4)
# Draw a stacked bar plot.
wet = month_and_road_surface['Wet or damp']
dry = month_and_road_surface['Dry']
frost = month_and_road_surface['Frost or ice']
snow = month_and_road_surface['Snow']
flood = month_and_road_surface['Flood over 3cm. deep']
wet.plot.bar(color = '#30a2da', alpha = 0.8)
dry.plot.bar(bottom = wet, color = '#e5ae38', alpha = 0.8)
frost.plot.bar(bottom = wet+dry, color = '#fc4f30', alpha = 0.8)
snow.plot.bar(bottom = wet+dry+frost, color = '#2fbabd', alpha = 0.8)
flood.plot.bar(bottom = wet+dry+frost+snow, color = '#da2f4c', alpha = 0.8)
plt.xlabel('')
pl.xticks(rotation = 360)
plt.ylabel('Proportion of Accidents by Road Surface Conditions')
plt.title('Figure 5. Proportion of Accidents by Road Surface Condition over the Year', fontsize = 15)
plt.legend(fontsize = 12, ncol = 3, loc = 9, bbox_to_anchor = (0.5, -0.1))


# ### Question 2. Give reasons to the situation that the proportion of sever accidents began to increase from 2014.
# 
# 1. Based on the Figure 6, we can see that the total numbers of traffic accidents show an overall decline trend. However, although the number of slight accidents goes down, the number of serious and fatal ones keep stable or even go up.
# 2. The proportion of fatal or serious traffic accidents goes up again from 2014. Especially in 2016, the proportion of serious accidents has grown according to Figure 7.

# In[9]:


plt.figure(figsize=(18, 5))
plt.subplot(1,2,1)

# Make a bar plot show the tendency of total number of traffic accidents from 2010 to 2016.
acc.Year.value_counts().sort_index().plot.bar(alpha=0.6)
pl.xticks(rotation = 360)
plt.ylabel('Number of Traffic Accidents')
plt.title('Figure 6. Traffic Accidents by Year', fontsize = 15)

# Make a table of the distribution of accident severity by year.
acc_se = acc.groupby(['Year','Accident_Severity'])['Accident_Index'].count().unstack()

# Calculate Proportions.
acc_year = acc_se.sum(axis = 1)
for se in acc.Accident_Severity.unique():
    acc_se[se] = acc_se[se] / acc_year
    acc_se[se] = round(acc_se[se], 4)
fatal = acc_se['Fatal']
serious = acc_se['Serious']
slight = acc_se['Slight']

# Draw the stacked bar plot for the proportion of accidents with different severity levels by years.
plt.subplot(1,2,2)
slight.plot.bar(color = '#30a2da', alpha = 0.8)
serious.plot.bar(bottom = slight, color = '#e5ae38', alpha = 0.8)
fatal.plot.bar(bottom = slight+serious, color = '#fc4f30', alpha = 0.8)
pl.xticks(rotation = 360)
plt.ylabel('Number of Traffic Accidents')
plt.title('Figure 7. Traffic Accidents of Different Severity by Year', fontsize = 15)
plt.legend(fontsize = 12, ncol = 3, loc = 9, bbox_to_anchor=(0.5, -0.15))


# As the table below shows, the ratio of the proportion of age band 0-15 group between 2016 to 2013 is 33.33. So one of the reason for the increasement of the proportion of serious accidents in 2016 is that the total age level for driving has been lower than before that even some teenagers tend to drive even at the illegal age.

# In[10]:


# Merge the two data sets.
acc_veh = pd.merge(veh, acc, how = 'inner', on = 'Accident_Index')
# Select accidents that are serious or fatal.
acc_veh = acc_veh[acc_veh.Accident_Severity.isin(['Serious','Fatal'])]


# In[11]:


# Summary accident distribution by year and age band of drivers
acc_age = acc_veh.groupby('Year_x')['Age_Band_of_Driver'].value_counts().sort_index().unstack()
acc_age['0 - 15'] = np.nansum(acc_age[['0 - 5','6 - 10','11 - 15']], axis = 1)
acc_age_1 = acc_age[['0 - 15', '16 - 20', '21 - 25', '26 - 35', '36 - 45', '46 - 55', '56 - 65', '66 - 75', 'Over 75']]
# Calculate the proportion in each age band.
acc_se_sum = acc_age_1.sum(axis = 1)
for x in ['0 - 15', '16 - 20', '21 - 25', '26 - 35', '36 - 45', '46 - 55', '56 - 65', '66 - 75', 'Over 75']:
    acc_age_1[x] = acc_age_1[x]/acc_se_sum
    acc_age_1[x] = round(acc_age_1[x], 4)
# Show the comparison between 2013 and 2016 by calculating ratios.
compare = acc_age_1.iloc[[3,6],:].T
compare['ratio'] = compare.iloc[:,1]/compare.iloc[:,0]
compare['ratio'] = round(compare['ratio'],4)
compare.T


# ### Question 3. K-means clustering: give suggestions to the Transport Department in London City, to set the optimal quantity of traffic control stations to correspond to accidents better.
# 1. As the elbow curve shows in Figure 8, the curve is close to be horizontal after the cluster number reached 6, so the optimal quantity of traffic control stations should be 6.
# 2. Figure 9 shows the result the distribution of serious or fatal accident distribution colored by the predicted cluster label.

# In[12]:


london_acc = acc_veh[acc_veh['Local_Authority_(District)'].isin(['City of London'])]
X = london_acc[['Longitude', 'Latitude']]


# In[51]:


plt.figure(figsize=(15, 5))

plt.subplot(1,2,1)
cost = []
for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(X)
    cost.append(kmeans.inertia_)
plt.plot(range(1, 10), cost)
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances from Samples to Clusters')
plt.title('Figure 8. Elbow Curve', fontsize = 15)
   

plt.subplot(1,2,2)
kmeans = KMeans(n_clusters = 6)
kmeans.fit(X)
y = kmeans.predict(X)
X['cluster'] = y
centers = kmeans.cluster_centers_
plt.scatter(X.Longitude, X.Latitude, c = X.cluster, s = 50, cmap = 'viridis', alpha = 0.4)
plt.scatter(centers[:,0], centers[:,1], c = 'darkgrey', s = 200)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Figur 9. Clustered Scatter Plot with Centers', fontsize = 15)
plt.xlim(-0.114,-0.07)
plt.ylim(51.504, 51.522)
for i in range(0,6):
    label = "(" + str(round(centers[i,0],4)) + "," + str(round(centers[i,1],4)) + ")"
    plt.annotate(s = label, xy = centers[i,[0,1]], fontsize = 12, c = 'darkblue')


# # Conclusions
# 
# Based on the results above, I have a few suggestions as follows:
# 
# 1. Pay attention to the traffic control during the afternoon rush, which is the traffic period that has the highest level of traffic accident even if on weekends.
# 
# 2. Enhance the education of the severity of illegal driving, as the result shows a tendancy that the number of serious or fatal accident caused by teenagers under 16 is increasing.
# 
# 3. According to Figure 9, there are 6 important points that the transpose department should pay strong attention to, so as to reply better to the traffic emergencies.
# 
# # Appendix
# 
# Please refer to the detailed codes [here](https://github.com/hal4006/Final-Project)
# 
# Part of EDA not included in the Result part is now shown as follow:

# I. (Question 1) The original summary table of the accident distribution by day of week and traffic periods before log-transform.

# In[52]:


day_of_week


# II. (Question 1) The original summary table of the accident distribution by month and road surface distributions.

# In[53]:


month_and_road_surface


# III. (Question 2) The original summary table of the accident distribution by year and severity.

# In[54]:


acc_se


# IV. (Question 2) The original summary table of the accident distribution by year and age band of the driver.

# In[55]:


acc_age


# V. (Question 2) Treemap of Vehicle Manoeuvre (since it does not show strong relationship with the severity distribution, so it is not included in the result part).

# In[56]:


manoeuvre = acc_veh.groupby('Vehicle_Manoeuvre').size()                                                .reset_index(name='counts')                                                    .sort_values(by='counts', ascending=False)

plt.figure(figsize=(12, 6))
#  prepare plot
labels = manoeuvre.iloc[0:10,:].apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
sizes = manoeuvre['counts'].values.tolist()
colors = [plt.cm.Pastel1(i/float(len(labels))) for i in range(len(labels))]

squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

# Decorate
plt.title('Treemap of Vehicle Manoeuvre', fontsize=15)
plt.axis('off')


# VI. (Question 2) The original serious or fatal accident distribution by year and manoeuvre.

# In[57]:


# Have a brief look at the distribution of vehicle manoeuvre.
acc_veh.groupby('Year_x')['Vehicle_Manoeuvre'].value_counts().unstack()


# VII. (Question 3) The data frame for the 6 cluster centers.

# In[58]:


centerDf = pd.DataFrame(centers[:,0:2], columns = ['Longitude','Latitude'])
centerDf

