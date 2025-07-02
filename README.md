# 🧠 RAM Price Prediction Web App

This project is a machine learning-powered web application built with **Flask** that predicts **RAM prices** based on hardware features. It includes data preprocessing, model training using **Gradient Boosting**, and interactive visualizations of both the data and the model results.

## 📊 Features

- IQR-based **outlier removal** from dataset
- Automatic **data preprocessing pipeline** with scaling and encoding
- **Gradient Boosting Regressor** model for RAM price prediction
- Web interface built with **Flask**
- Visualizations: boxplot, histograms, bar chart, and prediction scatter plot

## 📁 Project Structure

```
├── app.py                  # Flask application
├── train_model.py          # Data cleaning and model training script
├── model.pkl               # Saved trained model
├── data.csv                # Dataset (RAM and hardware features)
├── static/images/          # Folder where plots are saved
└── templates/
    ├── data-overview.html  # Webpage for data visualization
    └── model-results.html  # Webpage for model results
```

## ⚙️ How to Run

1. **Install dependencies**  
   *(Create a virtual environment if needed)*
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python train_model.py
   ```

3. **Run the Flask app**
   ```bash
   python app.py
   ```

4. **Visit in browser**
   ```
   http://127.0.0.1:5000/
   ```

## 🌐 Routes

- `/` → Displays summary tables and data visualizations (boxplots, histograms, brand chart)
- `/modelresults` → Shows model evaluation metrics and prediction scatter plot

## 📦 Requirements

You can generate a `requirements.txt` like this:
```bash
pip freeze > requirements.txt
```

Or manually include:

```txt
Flask
pandas
matplotlib
seaborn
scikit-learn
joblib
```

## 📷 Example Visualizations

- 📦 Boxplot for `Price`, `RAM Size`, `Speed`
- 📈 Histograms for numerical columns
- 🧠 Brand frequency bar chart
- 🎯 Actual vs Predicted Price plot

## 🧑‍💻 Author

**Yunus Bozan**  
Computer Engineering Student  
Manisa Celal Bayar University

## 📌 License

This project is open source and free to use under the MIT License.
