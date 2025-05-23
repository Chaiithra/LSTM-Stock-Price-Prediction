# 📈 Stock Price Prediction Using LSTM (Deep Learning)

Predict future stock prices with a multi-layered LSTM neural network trained on historical stock data. Built with TensorFlow/Keras, this project applies sequence modeling to forecast stock trends based on 100-day historical windows.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🌟 Overview

This project demonstrates a deep learning-based approach to predict the stock prices of **POWERGRID.NS** using historical closing prices. It uses:
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) layers
- Dropout regularization
- Mean Squared Error loss for training

---

## 📊 Model Architecture

```python
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(100,1)),
    Dropout(0.2),
    LSTM(60, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(80, activation='relu', return_sequences=True),
    Dropout(0.4),
    LSTM(120, activation='relu'),
    Dropout(0.5),
    Dense(1)
])
````

🔧 Optimizer: `Adam`
📉 Loss: `Mean Squared Error`
🧠 Total Parameters: `178,761`

---

<details>
<summary>📁 <strong>Project Structure</strong></summary>

```
├── stock_prediction_lstm/
│   ├── stock_dl_model.h5        # Trained model
│   ├── notebook.ipynb           # Main code notebook
│   └── README.md                # Project readme
```

</details>

---

## 🧪 Data & Preprocessing

* **Dataset**: Stock price data for POWERGRID.NS
* **Windowing**: 100-day time steps for LSTM
* **Normalization**: MinMaxScaler
* **Train-Test Split**: Sequential (no shuffle)

---

## 📈 Predictions

The model predicts the stock price using a 100-day sliding window. Predictions are compared against true prices using plots.

📌 **Visualization Sample**:

![Predicted vs Actual](https://user-images.githubusercontent.com/placeholder/image.png) <!-- Replace with your actual plot image -->

---

## 🚀 How to Run

1. Clone the repo

```bash
git clone https://github.com/yourusername/LSTM-Stock-Price-Prediction.git
cd stock-lstm
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook or script

```bash
jupyter notebook
```

---

## 🧠 What You'll Learn

* LSTM architecture for time-series prediction
* Feature scaling & sequence shaping
* Model evaluation with plots
* How to train, save, and load Keras models

---

## ✅ Results Summary

| Metric          | Value                      |
| --------------- | -------------------------- |
| Final Loss      | \~0.0015 (MSE)             |
| Scaler Factor   | \~284.3                    |
| Prediction RMSE | \[calculate test set RMSE] |

---

## 💾 Save & Reload

```python
# Save model
model.save("stock_dl_model.h5")

# Load later
from keras.models import load_model
model = load_model("stock_dl_model.h5")
```

---

## 📚 Future Improvements

* Add live stock feed via APIs (e.g., Yahoo Finance)
* Use Bidirectional LSTM or GRU layers
* Hyperparameter optimization (learning rate, units, dropout)
* Integrate attention mechanisms for better sequence learning

---

## 🙌 Acknowledgments

* [TensorFlow Documentation](https://www.tensorflow.org/)
* [Keras API](https://keras.io/)
* [Yahoo Finance for Data](https://finance.yahoo.com/)

---

## 📧 **Contact**

Made with ❤️ by **[Chaiithra Thota]**  

Connect on LinkedIn: (https://www.linkedin.com/in/chaiithrathota/)

Connect on Twitter: (https://x.com/DebugDiary_)

---


