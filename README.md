# UltraTech Cement Ltd Stock Price Prediction

This project focuses on predicting the stock prices of UltraTech Cement Ltd using various machine learning models, including **LSTM**, **SVR**, **KNN**, and **K-means Clustering**. The dataset used is historical stock data containing Open, High, Low, Close, and Volume prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Models Used](#models-used) 
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [LSTM](#lstm)
  - [SVR](#svr)
  - [KNN](#knn)
  - [K-means Clustering](#k-means-clustering)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Project Overview

This project aims to predict stock prices by training and evaluating different models:
- **Long Short-Term Memory (LSTM)** for time-series analysis.
- **Support Vector Regression (SVR)** for predicting future stock prices.
- **K-Nearest Neighbors (KNN)** for regression analysis.
- **K-means Clustering** for identifying patterns in stock price movements.

## Models Used

1. **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) well-suited for time-series data.
2. **SVR (Support Vector Regression)**: Used to predict continuous values based on the historical price data.
3. **KNN (K-Nearest Neighbors)**: A simple and intuitive model used for regression and classification tasks.
4. **K-means Clustering**: Unsupervised learning method to cluster stock prices based on patterns in the data.

## Dependencies

Ensure that the following Python libraries are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow` or `keras`

You can install these dependencies using `pip`:

# UltraTech Cement Ltd Stock Price Prediction

This project focuses on predicting the stock prices of UltraTech Cement Ltd using various machine learning models, including **LSTM**, **SVR**, **KNN**, and **K-means Clustering**. The dataset used is historical stock data containing Open, High, Low, Close, and Volume prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Models Used](#models-used)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [LSTM](#lstm)
  - [SVR](#svr)
  - [KNN](#knn)
  - [K-means Clustering](#k-means-clustering)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Project Overview

This project aims to predict stock prices by training and evaluating different models:
- **Long Short-Term Memory (LSTM)** for time-series analysis.
- **Support Vector Regression (SVR)** for predicting future stock prices.
- **K-Nearest Neighbors (KNN)** for regression analysis.
- **K-means Clustering** for identifying patterns in stock price movements.

## Models Used

1. **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) well-suited for time-series data.
2. **SVR (Support Vector Regression)**: Used to predict continuous values based on the historical price data.
3. **KNN (K-Nearest Neighbors)**: A simple and intuitive model used for regression and classification tasks.
4. **K-means Clustering**: Unsupervised learning method to cluster stock prices based on patterns in the data.

## Dependencies

Ensure that the following Python libraries are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow` or `keras`

You can install these dependencies using `pip`:

```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Data Preprocessing

- **Date Conversion**: The ```Date``` column is converted to ```datetime``` format for proper time-series handling.

- **Normalization**: Features (Open, High, Low, Close, Volume) are normalized using ```MinMaxScaler```.

- **Train-Test Split**: 80% of the data is used for training, and 20% for testing.

## Model Training and Evaluation

## LSTM

- Used for time-series forecasting based on the past 100 days of stock data.

- The model is built using two LSTM layers followed by Dense layers.

- **Evaluation**: RMSE (Root Mean Square Error) is used to evaluate the model performance.

## SVR

- The ```rbf``` kernel is used to train the SVR model.

- The input features include Open, High, Low, and Volume, with the target being the Close price.

- **Evaluation**: RMSE is calculated for training and testing data.

## KNN

- KNN is used to predict stock prices based on the historical data.

- It takes into account the 5 nearest neighbors.

- **Evaluation**: RMSE is used to measure the accuracy.

## K-means Clustering

- Applied to group similar stock price movements together.

- Helps in identifying patterns and anomalies in the stock data.

- Visualized using scatter plots with distinct clusters.

## Results

- The performance of each model is evaluated using RMSE (Root Mean Square Error).

- Graphs are generated to visualize the predictions against the actual stock price values.

## Usage

1. Clone this repository:

```
git clone <repository_url>
```

2. Run the Python script that includes the preprocessing, training, and evaluation:

```
python stock_price_prediction.py
```

3. Ensure the dataset (```ULTRACEMCO.csv```) is placed in the working directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
