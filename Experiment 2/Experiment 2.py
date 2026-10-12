# Name :- Arpit Giri
# CU24250181
# BTech CSE 3rd Year SECTION - "B"
# Roll No. :- 11

# ============================================================
# Experiment No. 2
# Experiment Name
# Exploratory Data Analysis (EDA) on a Real-World Business
# Dataset using Python
# ============================================================

# Aim:
# To perform Exploratory Data Analysis (EDA) on a real-world
# dataset using Python in order to understand the dataset's
# structure, identify trends, detect anomalies, analyze feature
# relationships, and generate meaningful business insights
# through descriptive statistics and visualizations.


# ============================================================
# Step 1: Import Required Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tkinter import Tk, filedialog


# ============================================================
# Step 2: Select / Upload Dataset from Computer
# ============================================================

root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Cleaned Business Dataset",
    filetypes=[
        ("CSV Files", "*.csv"),
        ("Excel Files", "*.xlsx *.xls"),
        ("All Files", "*.*")
    ]
)

if not file_path:
    print("No dataset selected.")
    exit()


# ============================================================
# Step 3: Load Dataset
# ============================================================

if file_path.lower().endswith(".csv"):
    df = pd.read_csv(file_path)

elif file_path.lower().endswith((".xlsx", ".xls")):
    df = pd.read_excel(file_path)

else:
    print("Unsupported file format.")
    exit()


print("Dataset loaded successfully.")
print("Selected Dataset:", file_path)


# ============================================================
# Step 4: Display First Five Records
# ============================================================

print("\n========== FIRST FIVE RECORDS ==========")

print(df.head())


# ============================================================
# Step 5: Display Dataset Dimensions
# ============================================================

print("\n========== DATASET DIMENSIONS ==========")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ============================================================
# Step 6: Display Column Names
# ============================================================

print("\n========== COLUMN NAMES ==========")

print(df.columns.tolist())


# ============================================================
# Step 7: Display Data Types
# ============================================================

print("\n========== DATA TYPES ==========")

print(df.dtypes)


# ============================================================
# Step 8: Display Dataset Information
# ============================================================

print("\n========== DATASET INFORMATION ==========")

df.info()


# ============================================================
# Step 9: Statistical Summary
# ============================================================

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe(include="all"))


# ============================================================
# Step 10: Missing Value Analysis
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:",
      df.isnull().sum().sum())


# ============================================================
# Step 11: Duplicate Analysis
# ============================================================

print("\n========== DUPLICATE RECORDS ==========")

duplicate_count = df.duplicated().sum()

print("Number of Duplicate Records:",
      duplicate_count)


# ============================================================
# Step 12: Identify Numerical and Categorical Columns
# ============================================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\n========== NUMERICAL FEATURES ==========")

print(numerical_columns)

print("\n========== CATEGORICAL FEATURES ==========")

print(categorical_columns)


# ============================================================
# Step 13: Value Counts for Categorical Variables
# ============================================================

print("\n========== CATEGORICAL VALUE COUNTS ==========")

for column in categorical_columns:

    print("\nColumn:", column)

    print(
        df[column]
        .value_counts()
        .head(10)
    )


# ============================================================
# Step 14: Univariate Analysis - Histograms
# ============================================================

print("\n========== UNIVARIATE ANALYSIS ==========")

if len(numerical_columns) > 0:

    for column in numerical_columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True
        )

        plt.title(
            "Distribution of " + column
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

print("Numerical feature distributions analyzed.")


# ============================================================
# Step 15: Categorical Variable Analysis
# ============================================================

if len(categorical_columns) > 0:

    # Select first few categorical columns
    selected_categories = categorical_columns[:4]

    for column in selected_categories:

        # Limit categories for better visualization
        top_categories = (
            df[column]
            .value_counts()
            .head(10)
        )

        plt.figure(figsize=(10, 5))

        sns.barplot(
            x=top_categories.values,
            y=top_categories.index
        )

        plt.title(
            "Top Categories in " + column
        )

        plt.xlabel("Count")
        plt.ylabel(column)

        plt.tight_layout()
        plt.show()

print("Categorical variables analyzed using bar charts.")


# ============================================================
# Step 16: Box Plot for Numerical Features
# ============================================================

if len(numerical_columns) > 0:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df[numerical_columns]
    )

    plt.title(
        "Box Plot of Numerical Features"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

print("Box plot generated successfully.")


# ============================================================
# Step 17: Correlation Matrix
# ============================================================

if len(numerical_columns) >= 2:

    correlation_matrix = df[
        numerical_columns
    ].corr()

    print("\n========== CORRELATION MATRIX ==========")

    print(correlation_matrix)


    # ========================================================
    # Step 18: Correlation Heatmap
    # ========================================================

    plt.figure(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()
    plt.show()

    print(
        "Correlation heatmap generated successfully."
    )

else:

    print(
        "\nNot enough numerical features "
        "for correlation analysis."
    )


# ============================================================
# Step 19: Bivariate Analysis
# ============================================================

print("\n========== BIVARIATE ANALYSIS ==========")

if len(numerical_columns) >= 2:

    x_column = numerical_columns[0]
    y_column = numerical_columns[1]

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column
    )

    plt.title(
        x_column + " vs " + y_column
    )

    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plt.tight_layout()
    plt.show()

    print(
        "Scatter plot generated for:",
        x_column,
        "and",
        y_column
    )

else:

    print(
        "Not enough numerical columns "
        "for scatter plot."
    )


# ============================================================
# Step 20: Multivariate Analysis
# ============================================================

print("\n========== MULTIVARIATE ANALYSIS ==========")

if len(numerical_columns) >= 3:

    sns.pairplot(
        df[numerical_columns[:5]].dropna()
    )

    plt.show()

    print(
        "Pair plot generated successfully."
    )

else:

    print(
        "Not enough numerical features "
        "for multivariate analysis."
    )


# ============================================================
# Step 21: Mean, Median and Standard Deviation
# ============================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")

if len(numerical_columns) > 0:

    for column in numerical_columns:

        print("\nFeature:", column)

        print(
            "Mean:",
            round(df[column].mean(), 2)
        )

        print(
            "Median:",
            round(df[column].median(), 2)
        )

        print(
            "Standard Deviation:",
            round(df[column].std(), 2)
        )

        print(
            "Minimum:",
            df[column].min()
        )

        print(
            "Maximum:",
            df[column].max()
        )


# ============================================================
# Step 22: Detect Highly Correlated Features
# ============================================================

if len(numerical_columns) >= 2:

    correlation_matrix = df[
        numerical_columns
    ].corr()

    print(
        "\n========== HIGH CORRELATIONS =========="
    )

    found_correlation = False

    for i in range(
        len(correlation_matrix.columns)
    ):

        for j in range(i + 1,
                       len(correlation_matrix.columns)):

            correlation_value = (
                correlation_matrix.iloc[i, j]
            )

            if abs(correlation_value) >= 0.70:

                print(
                    correlation_matrix.columns[i],
                    "<->",
                    correlation_matrix.columns[j],
                    ":",
                    round(correlation_value, 2)
                )

                found_correlation = True

    if not found_correlation:

        print(
            "No strong correlation "
            "(absolute correlation >= 0.70) found."
        )


# ============================================================
# Step 23: Business Insights
# ============================================================

print("\n========== BUSINESS INSIGHTS ==========")

print(
    "1. Numerical distributions were analyzed "
    "using histograms and density plots."
)

print(
    "2. Frequently occurring categories were "
    "identified using value counts and bar charts."
)

print(
    "3. Box plots were used to identify "
    "potential outliers."
)

print(
    "4. Correlation analysis was used to "
    "identify relationships between numerical features."
)

print(
    "5. Scatter plots were used to analyze "
    "relationships between pairs of variables."
)

print(
    "6. Multivariate analysis was used to "
    "observe relationships among multiple features."
)


# ============================================================
# Step 24: Final Output
# ============================================================

print(
    "\n========== EXPLORATORY DATA ANALYSIS =========="
)

print("Dataset Loading: Completed")
print("Dataset Inspection: Completed")
print("Statistical Analysis: Completed")
print("Missing Value Analysis: Completed")
print("Duplicate Analysis: Completed")
print("Univariate Analysis: Completed")
print("Categorical Analysis: Completed")
print("Correlation Analysis: Completed")
print("Heatmap Generation: Completed")
print("Box Plot Analysis: Completed")
print("Scatter Plot Analysis: Completed")
print("Multivariate Analysis: Completed")
print("Business Insight Generation: Completed")

print("\nExperiment Completed Successfully.")


# ============================================================
# EXPERIMENT 2 - ANSWERS TO QUESTIONS
# ============================================================


# ------------------------------------------------------------
# QUESTION 1:
# What is Exploratory Data Analysis (EDA), and why is it
# performed before machine learning?
# ------------------------------------------------------------

# Exploratory Data Analysis (EDA) is the process of examining,
# summarizing and visualizing a dataset to understand its
# structure, patterns, relationships and anomalies.
#
# EDA is performed before machine learning because it helps
# identify:
#
# 1. Missing values.
# 2. Duplicate records.
# 3. Outliers.
# 4. Data distributions.
# 5. Relationships between features.
# 6. Skewness.
# 7. Class imbalance.
#
# EDA helps analysts understand the data before selecting
# features, preprocessing techniques and machine learning
# models.


# ------------------------------------------------------------
# QUESTION 2:
# Differentiate between univariate, bivariate, and
# multivariate analysis with suitable examples.
# ------------------------------------------------------------

# Univariate Analysis:
#
# Analysis of a single variable.
#
# Example:
# Studying the distribution of employee age using a histogram.
#
#
# Bivariate Analysis:
#
# Analysis of two variables to understand their relationship.
#
# Example:
# Studying the relationship between sales and profit using
# a scatter plot.
#
#
# Multivariate Analysis:
#
# Analysis involving three or more variables simultaneously.
#
# Example:
# Studying relationships between sales, profit, quantity and
# discount using a pair plot or multivariate visualization.
#
#
# Therefore:
#
# Univariate  -> One variable.
# Bivariate   -> Two variables.
# Multivariate -> Three or more variables.


# ------------------------------------------------------------
# QUESTION 3:
# What insights can be obtained from a correlation heatmap?
# ------------------------------------------------------------

# A correlation heatmap represents the correlation between
# numerical variables using colors and correlation values.
#
# It can help identify:
#
# 1. Strong positive relationships.
# 2. Strong negative relationships.
# 3. Weak relationships.
# 4. Highly correlated features.
# 5. Possible multicollinearity.
#
# Correlation values generally range from -1 to +1.
#
# +1 -> Strong positive correlation.
# -1 -> Strong negative correlation.
#  0 -> Little or no linear correlation.
#
# A heatmap therefore provides a quick visual summary of
# relationships between numerical features.


# ------------------------------------------------------------
# QUESTION 4:
# Explain the purpose of histograms, box plots, and scatter
# plots in EDA.
# ------------------------------------------------------------

# Histogram:
#
# A histogram shows the distribution of a numerical variable.
# It helps identify:
#
# 1. Central tendency.
# 2. Spread.
# 3. Skewness.
# 4. Possible multiple peaks.
#
#
# Box Plot:
#
# A box plot summarizes the distribution of a numerical
# variable and helps identify possible outliers.
#
# It displays:
#
# 1. Median.
# 2. Quartiles.
# 3. Interquartile range.
# 4. Potential outliers.
#
#
# Scatter Plot:
#
# A scatter plot shows the relationship between two numerical
# variables.
#
# It helps identify:
#
# 1. Positive relationships.
# 2. Negative relationships.
# 3. Clusters.
# 4. Outliers.
# 5. Patterns.


# ------------------------------------------------------------
# QUESTION 5:
# How can EDA help identify data quality issues before analysis?
# ------------------------------------------------------------

# EDA helps identify data quality problems by examining the
# structure, values and distributions of the dataset.
#
# It can detect:
#
# 1. Missing values using isnull().
# 2. Duplicate records using duplicated().
# 3. Incorrect data types using info() and dtypes.
# 4. Outliers using box plots.
# 5. Inconsistent categories using value_counts().
# 6. Unusual distributions using histograms.
# 7. Extreme values using descriptive statistics.
#
# Identifying these issues before analysis improves the
# reliability of the results.


# ------------------------------------------------------------
# QUESTION 6:
# Why is correlation important in predictive analytics?
# Can correlation imply causation?
# ------------------------------------------------------------

# Correlation is important because it measures the strength
# and direction of a linear relationship between variables.
#
# In predictive analytics, correlation can help identify
# potentially useful features and detect highly related
# variables.
#
# However, correlation does NOT necessarily imply causation.
#
# Two variables can be correlated because:
#
# 1. One variable influences another.
# 2. Both are influenced by another variable.
# 3. The relationship occurs by coincidence.
#
# Therefore, correlation indicates association, not necessarily
# a cause-and-effect relationship.


# ------------------------------------------------------------
# QUESTION 7:
# Which visualization would you use to analyze categorical
# and numerical variables? Justify your choice.
# ------------------------------------------------------------

# For categorical variables:
#
# Bar charts and count plots are suitable because they show
# the frequency of different categories clearly.
#
#
# For numerical variables:
#
# Histograms are useful for understanding the distribution.
#
# Box plots are useful for comparing distributions and
# identifying outliers.
#
#
# For relationships between numerical variables:
#
# Scatter plots are useful because they display the relationship
# between two numerical variables.
#
#
# Therefore:
#
# Categorical -> Bar Chart / Count Plot.
# Numerical -> Histogram / Box Plot.
# Two Numerical Variables -> Scatter Plot.


# ------------------------------------------------------------
# QUESTION 8:
# What business insights can be derived from the Netflix
# (or Superstore/HR Analytics) dataset through EDA?
# ------------------------------------------------------------

# EDA can provide different business insights depending on
# the selected dataset.
#
#
# Netflix Dataset:
#
# 1. Distribution of movies and TV shows.
# 2. Most common genres.
# 3. Content distribution by country.
# 4. Content added over time.
# 5. Distribution of movie durations.
#
#
# Superstore Dataset:
#
# 1. Sales performance.
# 2. Profitability.
# 3. Product category performance.
# 4. Regional performance.
# 5. Relationship between discount and profit.
#
#
# IBM HR Dataset:
#
# 1. Employee attrition patterns.
# 2. Job-role distribution.
# 3. Salary-related patterns.
# 4. Department-wise employee characteristics.
# 5. Factors associated with employee attrition.
#
# Therefore, EDA can help organizations identify trends,
# anomalies and factors that may influence business decisions.


# ------------------------------------------------------------
# QUESTION 9:
# How does EDA contribute to feature selection and model
# building?
# ------------------------------------------------------------

# EDA helps understand which features may contain useful
# information for predicting the target variable.
#
# Correlation analysis can identify relationships between
# numerical features.
#
# Distribution analysis can identify highly skewed variables.
#
# Box plots can identify extreme values.
#
# Categorical analysis can reveal important groups or classes.
#
# EDA can also identify redundant or highly correlated
# features that may create multicollinearity.
#
# Therefore, EDA helps in:
#
# 1. Selecting useful features.
# 2. Removing irrelevant features.
# 3. Detecting redundant features.
# 4. Choosing suitable preprocessing techniques.
# 5. Selecting appropriate machine learning models.


# ------------------------------------------------------------
# QUESTION 10:
# What challenges might arise while performing EDA on
# large-scale real-world datasets?
# ------------------------------------------------------------

# EDA on large datasets can present several challenges.
#
# 1. High computational requirements:
#    Large datasets require more CPU and memory.
#
# 2. Large storage requirements:
#    Very large datasets may require substantial storage.
#
# 3. Slow visualization:
#    Plotting millions of records can be time-consuming.
#
# 4. Missing and inconsistent data:
#    Large datasets may contain many data quality problems.
#
# 5. High dimensionality:
#    Datasets with many features can be difficult to visualize
#    and analyze.
#
# 6. Processing time:
#    Statistical calculations can take longer on very large
#    datasets.
#
# Sampling, efficient data structures, aggregation and
# distributed processing can be used to make EDA more
# manageable on large-scale datasets.
