# Predicting-Bike-Rentals-with-Regression-Neural-Networks

This project uses the Seoul Bike Sharing Dataset to predict the demand for bike rentals based on various weather conditions. It implements both linear regression models and neural network models to perform simple and multiple regression tasks.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Feature Selection](#feature-selection)
- [Models](#models)
  - [Simple Linear Regression](#simple-linear-regression)
  - [Multiple Linear Regression](#multiple-linear-regression)
  - [Neural Network Models](#neural-network-models)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Dataset
The dataset used in this project is the **Seoul Bike Sharing Dataset**. The data consists of various features such as:
- Bike rental count (`bike_count`)
- Temperature (`temp`)
- Humidity (`humidity`)
- Wind speed (`wind`)
- Visibility (`visibility`)
- Dew point temperature (`dew_pt_temp`)
- Solar radiation (`radiation`)
- Rainfall (`rain`)
- Snowfall (`snow`)
- Whether the day is a holiday or not (`Holiday`)
- The season (`Seasons`)

For simplicity, some features like `Date`, `Holiday`, and `Seasons` are dropped from the final analysis.

## Project Structure
- **Data Preprocessing**: 
  - Initial data exploration (`describe()`, `info()`)
  - Dropping irrelevant columns
  - Converting categorical features like `functional` to numeric values
  - Filtering the dataset for records only at noon (`hour == 12`)
  
- **Feature Selection**: 
  - Initial feature selection based on scatter plots to assess linear relationships with the target variable (`bike_count`)
  - Dropping features such as `wind`, `visibility`, `functional`, `rain`, and `snow` due to lack of linearity

- **Train-Test Split**: 
  - The data is split into training, validation, and test sets using `np.split()` with a 60-20-20 ratio

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn tensorflow imbalanced-learn
```

## Feature Selection
After initial visualizations, we filtered the dataset to only include relevant features:
- Dropped columns: `Date`, `Holiday`, `Seasons`, `wind`, `visibility`, `functional`, `rain`, and `snow`
- Final features used: `temp`, `humidity`, `dew_pt_temp`, `radiation`

## Models

### Simple Linear Regression
We use `temperature` as the single feature to predict the `bike_count`. The model is trained using Scikit-learn's `LinearRegression`.

```python
simple_reg = LinearRegression()
simple_reg.fit(X_train_temp, y_train_temp)
```

### Multiple Linear Regression
For multiple regression, all remaining features (`temp`, `humidity`, `dew_pt_temp`, `radiation`) are utilized.

## Data Source
- **UCI Machine Learning Repository**: [http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)
- **South Korea Public Holidays**: [http://data.seoul.go.kr/](http://data.seoul.go.kr/) and [publicholidays.go.kr](http://publicholidays.go.kr)
