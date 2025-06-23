📈 Stock Price Prediction using TensorFlow
This project focuses on predicting stock closing prices using deep learning techniques with TensorFlow and LSTM (Long Short-Term Memory) models. It utilizes historical stock price data for multiple companies over a 5-year period to build and train a predictive model.

📁 Files in this Project
all_stocks_5yr.csv – Historical stock prices for various companies over 5 years.

Stock Price Prediction Project using TensorFlow.ipynb – The Jupyter Notebook containing the full code for data preprocessing, modeling, training, and evaluation.

🎯 Objective
Perform time-series forecasting to predict future stock closing prices.

Use LSTM (a type of RNN) model to capture long-term dependencies in stock price data.

Evaluate the model’s performance using appropriate metrics and visualizations.

⚙️ Technologies & Libraries Used
Python

Pandas, NumPy – Data handling

Matplotlib, Seaborn – Visualization

TensorFlow, Keras – Deep learning and LSTM modeling

scikit-learn – Data preprocessing and evaluation

🔍 Project Workflow
Data Loading & Exploration

Loaded all_stocks_5yr.csv containing date, open, close, volume, etc.

Checked missing values, stock symbols, and distributions.

Data Preprocessing

Filtered stock symbol(s) (e.g., AAPL, GOOG) for training.

Scaled the 'Close' price using MinMaxScaler.

Created time-series sequences for LSTM input.

Model Building

Built an LSTM-based neural network using TensorFlow/Keras.

Included layers: LSTM, Dense, Dropout.

Model Training

Split data into training and testing sets.

Trained the model using a time-based sequence.

Model Evaluation

Plotted actual vs. predicted closing prices.

Calculated metrics like Mean Squared Error (MSE) and RMSE.

📊 Visualizations
Line plots of stock prices over time.

Predicted vs. actual price graphs.

Loss function curve during training.

🧪 Results
The LSTM model was able to capture the trend of stock prices fairly well.

Predictions closely followed real prices in the testing set, although short-term fluctuations remain challenging.

Model performance can improve with more features (like technical indicators, macroeconomic data, etc.).

🚀 How to Run
Clone the repository or download the files.

Open the notebook Stock Price Prediction Project using TensorFlow.ipynb.

Install required libraries:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
Run the notebook step-by-step.
