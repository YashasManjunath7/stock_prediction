# Stock Prediction using Machine Learning
## Abstract
This dissertation investigates machine learning models for stock price prediction, comparing traditional approaches (ARIMA, ARIMA+GARCH) with advanced deep learning models like Long Short-Term Memory (LSTM) networks, Convolutional Neural Networks (CNNs), and hybrid models (CNN-LSTM, CNN-GRU, LSTM-CNN-GRU). The study evaluates model performance using metrics like Root Mean Squared Error (RMSE) and R² scores. Findings suggest that deep learning models, especially hybrid architectures, significantly outperform traditional models by capturing complex patterns and temporal dependencies. However, deep learning models face challenges such as computational demands, hyperparameter sensitivity, and interpretability.

**Keywords**: Stock price prediction, machine learning, ARIMA, LSTM, CNN, hybrid models, deep learning, financial forecasting.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Challenges in Stock Prediction](#challenges-in-stock-prediction)  
3. [Objectives of the Study](#objectives-of-the-study)  
4. [Literature Review](#literature-review)  
    - [Early Research](#early-research)  
    - [ARIMA and GARCH Models](#arima-and-garch-models)  
    - [The Rise of Deep Learning](#the-rise-of-deep-learning)  
    - [Ensemble Learning & Hybrid Models](#ensemble-learning--hybrid-models)  
    - [Sentiment Analysis](#sentiment-analysis)  
    - [Reinforcement Learning](#reinforcement-learning)  
5. [Methodology](#methodology)  
    - [Data Collection](#data-collection)  
    - [Model Selection](#model-selection)  
    - [Model Implementation](#model-implementation)  
    - [Evaluation Metrics](#evaluation-metrics)  
6. [Experimental Setup](#experimental-setup)  
    - [Hardware/Software](#hardwaresoftware)  
    - [Tools & Libraries](#tools--libraries)  
    - [Model Training & Testing](#model-training--testing)  
7. [Results and Analysis](#results-and-analysis)  
    - [ARIMA](#arima)  
    - [ARIMA + GARCH](#arima--garch)  
    - [LSTM](#lstm)  
    - [LSTM-CNN](#lstm-cnn)  
    - [CNN-GRU](#cnn-gru)  
    - [Bidirectional LSTM-GRU](#bidirectional-lstm-gru)  
    - [LSTM-CNN-GRU](#lstm-cnn-gru)  
8. [Performance Evaluation](#performance-evaluation)  
9. [Discussion](#discussion)  
10. [Conclusion](#conclusion)  
11. [References](#references)

---

## Introduction
Stock prediction, the art of forecasting future stock prices, plays a critical role in financial decision-making. This study investigates machine learning models for predicting stock prices, focusing on traditional statistical approaches and advanced deep learning techniques.

## Challenges in Stock Prediction
Stock prediction faces challenges due to market efficiency, data complexity, overfitting, and noise in the data. These challenges stem from the dynamic and unpredictable nature of the financial markets.

## Objectives of the Study
The main objectives of this study are:
- Analyze traditional statistical models for stock prediction.
- Investigate the potential of machine learning models (ARIMA, LSTM, CNN).
- Identify key challenges in stock prediction and propose solutions.
- Develop and implement a hybrid model combining various approaches.

## Literature Review

### Early Research
Early studies in stock prediction used traditional models like Artificial Neural Networks (ANNs), which could capture non-linear relationships in stock market data.

### ARIMA and GARCH Models
ARIMA models capture linear relationships, while GARCH models address volatility clustering. Combining ARIMA with GARCH improves performance in time series forecasting.

### The Rise of Deep Learning
LSTM networks and CNNs have revolutionized stock prediction by effectively handling time series data and learning long-term dependencies. Hybrid CNN-LSTM models outperform standalone models in stock prediction.

### Ensemble Learning & Hybrid Models
Ensemble methods like CNN-LSTM and CNN-GRU improve prediction accuracy by combining multiple models. These approaches mitigate overfitting and increase robustness in volatile markets.

### Sentiment Analysis
Sentiment analysis leverages alternative data, such as social media and news articles, to predict stock price movements. This is becoming an essential component of modern stock prediction models.

### Reinforcement Learning
Reinforcement learning adapts to market changes by interacting with the environment, making decisions based on real-time data. This emerging approach is promising for automated trading strategies.

## Methodology

### Data Collection
The study uses stock data from sources like **Yahoo Finance** and **Alpha Vantage**. Key features include stock prices, technical indicators, and alternative data like sentiment analysis.

### Model Selection
The models selected include:
- **ARIMA**
- **ARIMA+GARCH**
- **LSTM Networks**
- **CNNs**
- **Hybrid Models (CNN-GRU, LSTM-GRU)**

### Model Implementation
Each model is implemented using **Python** and libraries such as **TensorFlow**, **Keras**, and **Scikit-learn**. Techniques like hyperparameter tuning and regularization are used to improve model performance.

### Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## Experimental Setup

### Hardware/Software
The models were trained using:
- **Hardware**: Intel Core i7, NVIDIA RTX 3080
- **Software**: Python 3.8, TensorFlow, Keras

### Tools & Libraries
- **TensorFlow/Keras**: For building deep learning models.
- **Scikit-learn**: For data preprocessing and evaluation.
- **Pandas/NumPy**: For data manipulation.
- **Matplotlib/Seaborn**: For visualizing predictions and evaluation metrics.

### Model Training & Testing
Models are trained using a split between training (70%), validation (15%), and test sets (15%). **Cross-validation** techniques ensure the model generalizes well to unseen data.

## Results and Analysis

### ARIMA
ARIMA captures trends and seasonality but struggles with non-linear patterns.

### ARIMA + GARCH
Combining ARIMA with GARCH improves volatility forecasting, but both models underperform on non-linear data.

### LSTM
LSTMs effectively capture long-term dependencies, outperforming traditional models.

### LSTM-CNN
This hybrid model captures both spatial and temporal features, resulting in improved accuracy.

### CNN-GRU
CNN-GRU models offer better generalization and faster training times.

### Bidirectional LSTM-GRU
This combination further enhances performance by processing data in both forward and backward directions.

### LSTM-CNN-GRU
The most sophisticated model in the study, offering the best performance in terms of capturing complex patterns and reducing RMSE.

## Performance Evaluation
| **Model**           | **RMSE**  | **R² Score** |  
|---------------------|-----------|--------------|  
| ARIMA               | 18.82     | -0.93        |  
| ARIMA + GARCH       | 53.27     | -13.93       |  
| LSTM                | 7.38      | 0.8          |  
| LSTM-CNN            | 14.07     | 0.66         |  
| CNN-GRU             | 8.97      | 0.77         |  
| Bidirectional LSTM  | 6.54      | 0.86         |  
| LSTM-CNN-GRU        | 6.54      | 0.86         |

## Discussion
Deep learning models, particularly LSTMs and hybrid architectures, demonstrate superior performance in capturing complex patterns and temporal dependencies. However, computational demands and hyperparameter sensitivity remain challenges.

## Conclusion
The study concludes that hybrid deep learning models, such as LSTM-CNN-GRU, offer the best performance for stock prediction. Future research should focus on developing more interpretable models and incorporating alternative data sources.

## References
1. Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series.
2. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks.
3. Karunasingha, D. (2021). Root Mean Square Error vs Mean Absolute Error.
4. Zhang, G. P. (2003). Time series forecasting using ARIMA and neural network models.

