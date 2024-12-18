import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
sars_df = pd.read_csv("/mnt/data/input_sars.csv")
bcell_df = pd.read_csv("/mnt/data/input_bcell.csv")
covid_df = pd.read_csv("/mnt/data/input_covid.csv")

# Display basic information
print("SARS Data\n", sars_df.head(), "\n")
print("B-Cell Data\n", bcell_df.head(), "\n")
print("COVID Data\n", covid_df.head(), "\n")

# Check for missing values
print("Missing Values in SARS Data:\n", sars_df.isnull().sum(), "\n")
print("Missing Values in B-Cell Data:\n", bcell_df.isnull().sum(), "\n")
print("Missing Values in COVID Data:\n", covid_df.isnull().sum(), "\n")

# Check for duplicates
print("Duplicate Rows in COVID Data:\n", covid_df.duplicated().sum(), "\n")

# Summary statistics
print("Summary Statistics for SARS Data:\n", sars_df.describe(), "\n")
print("Summary Statistics for B-Cell Data:\n", bcell_df.describe(), "\n")
print("Summary Statistics for COVID Data:\n", covid_df.describe(), "\n")

# Remove constant or near-constant columns in COVID dataset
covid_df_refined = covid_df.drop(columns=['parent_protein_id', 'protein_seq', 'isoelectric_point', 'aromaticity', 'hydrophobicity', 'stability'])

# Multicollinearity check using VIF for COVID Data
X_covid = covid_df_refined[['chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker']]
X_covid_with_const = sm.add_constant(X_covid)

# Calculate VIF
vif = pd.DataFrame()
vif["Variable"] = X_covid_with_const.columns
vif["VIF"] = [variance_inflation_factor(X_covid_with_const.values, i) for i in range(X_covid_with_const.shape[1])]
print("Variance Inflation Factor (VIF):\n", vif, "\n")

# Drop variables with high VIF (if any)
X_covid_refined = X_covid.drop(columns=['emini'])  # Remove 'emini' as predictor since it's the target

# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_covid_refined, covid_df_refined['emini'], test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_c, y_train_c)
y_ridge_pred = ridge.predict(X_test_c)
ridge_mse = mean_squared_error(y_test_c, y_ridge_pred)
ridge_r2 = r2_score(y_test_c, y_ridge_pred)
print(f"Ridge Regression Results: MSE = {ridge_mse:.4f}, R2 = {ridge_r2:.4f}")

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_c, y_train_c)
y_lasso_pred = lasso.predict(X_test_c)
lasso_mse = mean_squared_error(y_test_c, y_lasso_pred)
lasso_r2 = r2_score(y_test_c, y_lasso_pred)
print(f"Lasso Regression Results: MSE = {lasso_mse:.4f}, R2 = {lasso_r2:.4f}")

# Gaussian GLM after handling duplicates and refining predictors
covid_df_no_duplicates = covid_df_refined.drop_duplicates()
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    covid_df_no_duplicates[['chou_fasman', 'kolaskar_tongaonkar', 'parker']],
    covid_df_no_duplicates['emini'],
    test_size=0.2, random_state=42
)

# Fit GLM with Gaussian family
glm_gaussian = sm.GLM(y_train_c, sm.add_constant(X_train_c), family=sm.families.Gaussian())
result_gaussian = glm_gaussian.fit()

# Predictions
y_pred_glm = result_gaussian.predict(sm.add_constant(X_test_c))
glm_mse = mean_squared_error(y_test_c, y_pred_glm)
glm_r2 = r2_score(y_test_c, y_pred_glm)

# Output GLM results
print("Gaussian GLM Results:")
print(result_gaussian.summary())
print(f"Gaussian GLM MSE: {glm_mse:.4f}, R2: {glm_r2:.4f}")

# Plot distribution of residuals
residuals = y_test_c - y_pred_glm
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
