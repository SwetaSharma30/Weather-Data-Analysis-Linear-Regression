

# # Weather data analysis and linear regression

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,RobustScaler
from sklearn.metrics import mean_squared_error, r2_score


# # Load the Dataset

# In[3]:


pd.read_csv("weather_data_extended.csv")


# In[4]:


df= pd.read_csv("weather_data_extended.csv")


# In[5]:


df.head()


# In[6]:


df.info


# # Check for Missing Values

# In[7]:


df.isnull().sum()


# In[10]:


df.isnull().sum()


# In[9]:


#dropping missing values
df=df.dropna()


# # Statistics Summary

# In[11]:


df.describe()


# # String Indexing

# In[ ]:


# String indexing on 'Location' column: Converting Location column from string to integer
df.loc[:, 'Location_index'] = df['Location'].astype('category').cat.codes

# Get unique values
unique_locations = df['Location'].drop_duplicates().reset_index(drop=True)
unique_location_indices = df['Location_index'].drop_duplicates().reset_index(drop=True)

# Create a new DataFrame
unique_df = pd.DataFrame({
    'Unique_Locations': unique_locations,
    'Unique_Location_Indices': unique_location_indices
})

unique_df


# # Correlation Heatmap

# In[14]:


# Select only numeric columns before correlation
numeric_df = df.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# # Box Plot: Humidity

# In[15]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df['Pressure (mb)'])
plt.title('Boxplot of Humidity')
plt.show()


# # Define Features and Target Variable

# In[16]:


X = df[['Humidity (%)', 'Wind Speed (kph)', 'Pressure (mb)', 'Visibility (km)', 'Location_index']] #features
y = df['Temperature (°C)'] #target variable or label


# # Standard Scaling

# In[17]:


# Standard scaling the values in features
scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(df_scaled.describe())


# # Train-Test Split

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train Linear Regression Model

# In[19]:


model = LinearRegression()
model.fit(X_train, y_train)


# # Predictions

# In[20]:


y_pred = model.predict(X_test)


# # Model Evaluation

# In[21]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared Score: {r2:.2f}')


# # Scatter Plot: Actual vs Predicted Temperature

# In[22]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Actual vs Predicted Temperature')
plt.show()


# # Types of Linear Regression & Accuracy Comparison
# In this section, we will compare different types of linear regression models:
# 
# Lasso Regression: L1 regularization (Lasso) encourages sparsity in the model by forcing some coefficients to be exactly zero, effectively selecting only a subset of features.
# Ridge Regression: L2 regularization (Ridge) discourages large coefficients by shrinking them toward zero but usually does not force them to be exactly zero.
# We will evaluate these models based on their Mean Squared Error (MSE) and R-squared Score(R²).

# In[23]:


# Define models
models = {
    "Simple Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
}

# Store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R² Score": r2}

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).T
print(results_df)


# In[24]:


# plot comparison
import matplotlib.pyplot as plt
import numpy as np

mse = results_df['MSE']
r2_score = results_df['R² Score']

positions = np.arange(len(mse))
bar_width = 0.2

plt.bar(positions - bar_width, mse, width = bar_width, label = 'MSE')
plt.bar(positions + bar_width, r2_score, width = bar_width, label = 'R2_Score')

# adding labels and title
plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Comparison of Regression Metrics')

# adding the legend
plt.legend()
plt.xticks(positions, ['Regression', 'Lasso', 'Ridge'])
plt.show()





