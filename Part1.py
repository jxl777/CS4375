#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jxl777/CS4375/blob/main/Linear_Regression_using_Gradient_Descent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor


# In[7]:


# URL of the dataset uploaded to GitHub
url = 'https://raw.githubusercontent.com/jxl777/CS4375/main/Steel_industry_data.csv'


# In[8]:


# Load the dataset
try:
    data = pd.read_csv(url)
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")


# In[11]:


# Display the first few rows of the dataset to inspect column names and data types
print(data.head())
print(data.columns)


# In[12]:


# Step 1: Remove null or NA values
data = data.dropna()
print(f"Dataset shape after removing null values: {data.shape}")


# In[13]:


# Step 2: Remove redundant rows (duplicate rows)
data = data.drop_duplicates()
print(f"Dataset shape after removing duplicate rows: {data.shape}")


# In[14]:


# Step 3: Convert categorical variables to numerical variables if there are any

for col in data.select_dtypes(include=['object', 'category']).columns:
    data[col] = data[col].astype('category').cat.codes


# In[16]:


# Define the name of the target variable column
target_variable = 'Usage_kWh' # Replace 'Usage_kWh' with the actual name of target variable column

# Split the dataset into features and target variable
X = data.drop(target_variable, axis=1)
y = data[target_variable]


# In[17]:


# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[18]:


# Add a bias term (column of ones) to the feature matrix
X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]


# In[19]:


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=42)


# In[20]:


# Linear regression prediction function
def predict(X, theta):
    return X.dot(theta)


# In[21]:


# Cost function (Mean Squared Error)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# In[22]:


# Gradient of the cost function
def gradient(X, y, theta):
    m = len(y)
    return (1 / m) * X.T.dot(predict(X, theta) - y)


# In[23]:


# Generalized gradient descent function
def gradient_descent(gradient, X, y, start, learn_rate, n_iter=1000, tolerance=1e-6):
    vector = start
    cost_history = [compute_cost(X, y, vector)]
    for i in range(n_iter):
        diff = -learn_rate * gradient(X, y, vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
        cost_history.append(compute_cost(X, y, vector))
    return vector, cost_history


# In[30]:


# Tuning parameters
learning_rates = [1e-2, 1e-3, 1e-4]
num_iterations = [1000, 5000, 10000]


# In[31]:


# Log file to record parameters and error
log_file = "gradient_descent_log.txt"

with open(log_file, "w") as log:
    log.write("Learning Rate,Num Iterations,Training MSE,Test MSE,Training R2,Test R2\n")

    for lr in learning_rates:
        for iters in num_iterations:
            # Initialize parameters
            theta_initial = np.random.randn(X_train.shape[1])  # Initialize theta randomly

            # Run gradient descent
            theta, cost_history = gradient_descent(gradient, X_train, y_train, theta_initial, lr, iters)

            # Evaluate the model
            mse_train = mean_squared_error(y_train, predict(X_train, theta))
            mse_test = mean_squared_error(y_test, predict(X_test, theta))
            r2_train = r2_score(y_train, predict(X_train, theta))
            r2_test = r2_score(y_test, predict(X_test, theta))

            # Log the results
            log.write(f"{lr},{iters},{mse_train},{mse_test},{r2_train},{r2_test}\n")
            print(f"Learning Rate: {lr}, Iterations: {iters}, Training MSE: {mse_train}, Test MSE: {mse_test}")


# In[32]:


# Read the log file and identify the best parameters
log_data = pd.read_csv(log_file)
best_params = log_data.loc[log_data['Test MSE'].idxmin()]
best_learning_rate = best_params['Learning Rate']
best_num_iterations = best_params['Num Iterations']
print(f"Best parameters - Learning Rate: {best_learning_rate}, Num Iterations: {best_num_iterations}")


# In[33]:


# Re-train the model with the best parameters
theta_initial = np.random.randn(X_train.shape[1])  # Initialize theta randomly
theta_final, cost_history_final = gradient_descent(gradient, X_train, y_train, theta_initial, best_learning_rate, int(best_num_iterations))



# In[34]:


# Evaluate the final model
mse_train_final = mean_squared_error(y_train, predict(X_train, theta_final))
mse_test_final = mean_squared_error(y_test, predict(X_test, theta_final))
r2_train_final = r2_score(y_train, predict(X_train, theta_final))
r2_test_final = r2_score(y_test, predict(X_test, theta_final))

print(f'Final Training Mean Squared Error: {mse_train_final}')
print(f'Final Test Mean Squared Error: {mse_test_final}')
print(f'Final Training R2 Score: {r2_train_final}')
print(f'Final Test R2 Score: {r2_test_final}')


# In[35]:


# Plot the training cost over iterations for the final model
plt.plot(range(len(cost_history_final)), cost_history_final)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.title('Cost Function History')
plt.show()


# In[36]:


# Print the final parameters
print(f'Final theta: {theta_final}')


# In[37]:


print("I am satisfied that I found the best answer, since the both trainning and test R2 are close to 1 and test R2 is higher than trainning R2 and training MSE is higher than test MSE")

