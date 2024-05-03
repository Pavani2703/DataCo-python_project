#!/usr/bin/env python
# coding: utf-8

# #  Data Analysis on Cricket Bowlers Statistics

# 1.) Performing descriptive analytics calculate the measures of central tendency such as mean, median, and mode for  wickets taken, bowling average, economy rate, strike rate to understand the typical performance of bowlers.
# 
# 2.) Visualizing data through Scatter plots & barcharts to visualize the relationships between balls bowled and wickets taken 
# and visualizing to compare the performance of different bowlers and wickets taken respectively.
# 
# 3.) Running regression to investigate the relationship between factors like average, strike rate, number of innings player and the number of matches played,to predict the number of wickets a bowler is likely to take in future.

# In[1]:


import requests, re
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


#Website to web crawl data to run analysis
url = requests.get("https://www.espncricinfo.com/records/team/averages-bowling/australia-2/combined-test-odi-and-t20i-records-11")
html_code = str(url.content)

#regular expression to extract data 
regex_pattern = r'<tr class=.*?><td class=.*?><a href=.*?><span class=.*?>(.*?)</span></a></td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td><td class=.*?>(.*?)</td></tr>'
matches = re.compile(regex_pattern, re.S | re.I).findall(html_code)

#removing html tags 
def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

# Connect to database 
conn = sqlite3.connect("Bowling_data_stats.db")
cursor = conn.cursor()

# Create a new table for inserting data
cursor.execute('''
CREATE TABLE IF NOT EXISTS Bowling_data (
    id INTEGER PRIMARY KEY,
    Player_name TEXT,
    Span TEXT,
    Matches INTEGER,
    Innings INTEGER,
    Balls INTEGER,
    Maidens INTEGER,
    Runs INTEGER,
    Wickets INTEGER,
    BBI INTEGER,
    BBM INTEGER,
    Average REAL,
    Economy REAL,
    Strike_rate REAL
)
''')

# Insert the non-html tags data into the table
for m in matches:
    sorted_data = tuple(remove_tags(value) for value in m)
    cursor.execute("INSERT INTO Bowling_data (Player_name, Span, Matches, Innings, Balls, Maidens, Runs, Wickets, BBI, BBM, Average, Economy, Strike_rate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sorted_data)

# Commit the changes and close the connection
conn.commit()
conn.close()


# In[4]:


# Connecting to the database 
conn = sqlite3.connect("Bowling_data_stats.db")

# Read the data from the table 
df = pd.read_sql_query("SELECT * FROM Bowling_data", conn)

#Closing database connection
conn.close()

#view data
df


# In[5]:


#Cleaning data by replacing null values with zero

updated_column = ["id","Player_name", "Matches", "Innings", "Balls", "Maidens", "Runs", "Wickets", "BBI", "BBM", "Average", "Economy", "Strike_rate"]
df = df[updated_column].replace('-', '0', regex=True)
df


# # Descriptive Analytics

# In[6]:


#List of columns to run descriptive analysis
column_names = ['Matches', 'Innings', 'Balls', 'Maidens', 'Runs', 'Wickets', 'Average', 'Economy', 'Strike_rate']

#Converting columns to numeric values
for col in column_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#Removing rows with null values
df = df.dropna(subset=column_names)

#Calculate descriptive statistics for the numeric columns
descriptive_data = df[column_names].describe()

# Print the summary statistics
print(descriptive_data)


# # Data Visualization
# 

# In[7]:


df['Wickets'] = pd.to_numeric(df['Wickets'], errors='coerce')

#Sorting the Dataframe by wickets taken  in descending order and select the top 10
top_wicket_taker = df.sort_values(by='Wickets', ascending=False).head(10)

#Bar Chart for best top 10 bowlers
plt.figure(figsize=(12, 6))
plt.bar(top_wicket_taker['Player_name'], top_wicket_taker['Wickets'])
plt.xlabel('Player Name')
plt.ylabel('Wickets')
plt.title('Top 10 Players with Most Wickets')
plt.xticks(rotation=45)
plt.show()


# In[8]:


#Scatter plot to show the relation between balls bowled and wickets taken

plt.figure(figsize=(12, 6))
plt.scatter(df['Balls'], df['Wickets'])
plt.xlabel('Balls')
plt.ylabel('Wickets')
plt.title('Scatter plot of Balls Bowled against Wickets Taken')
plt.show()


# # OLS Regression

# In[17]:


#Deleting null values 
df = df.dropna(subset=['Wickets', 'Balls', 'Average', 'Matches', 'Strike_rate', 'Innings'])

#defining dependent and independent variables
X = df[['Average', 'Matches', 'Strike_rate', 'Innings']]
y = df['Wickets']

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)

#Adding a constant to independent variables for OLS regression
X_ols_train = sm.add_constant(X_train)

#Fit the OLS model to the training data
ols_model = sm.OLS(y_train, X_ols_train).fit()

#Printing the summary of OLS regression model
print(ols_model.summary())

#Adding a constant to independent variables for making predictions
X_ols_test = sm.add_constant(X_test)

#Predicting using the testing data
y_pred = ols_model.predict(X_ols_test)

#Calculating the mean squared error and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#plotting scatter plot to show regression line
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', label='Actual vs Predicted')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='red', label='Regression Line')
plt.xlabel('Actual Values (Wickets)')
plt.ylabel('Predicted Values (Wickets)')
plt.title('Actual vs Predicted Wickets (OLS Regression)')
plt.legend()

plt.show()


# # Conclusion 

# An R-squared score of 0.901 indicates that the model can account for 90.1% of the variation in the "Wickets." This suggests a significant correlation between the variables ('Average', 'Matches', 'Strike_rate', 'Innings') and a suitable model.
