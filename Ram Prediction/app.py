from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
from sklearn.model_selection import train_test_split

app = Flask(__name__)

os.makedirs('static/images', exist_ok=True)


def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    return df[~outliers]


df = pd.read_csv('data.csv')

num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df = remove_outliers_iqr(df, col)

df = df.dropna()

x = df.drop(columns=["Price"])
y = df["Price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model = joblib.load('model.pkl')


@app.route('/')
def dataoverview():
    describe_table = df.describe().round(2).to_html(classes='table table-bordered')
    head_table = df.head(10).to_html(classes='table table-striped')

    # Boxplot
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df[['Price', 'Speed', 'RAM Size']])
    plt.title("Boxplot of Numerical Features")
    plt.tight_layout()
    plt.savefig('static/images/boxplot.png')
    plt.close()

    # Distribution plots
    for col in ['Price', 'Speed', 'RAM Size']:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'static/images/dist_{col.lower()}.png')
        plt.close()

    # Brand bar chart
    plt.figure(figsize=(10, 6))
    brand_counts = df['Brand'].value_counts()
    sns.barplot(x=brand_counts.index, y=brand_counts.values, palette='Set2')
    plt.title('Brand Frequency')
    plt.xlabel('Brand')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/images/brand_freq.png')
    plt.close()

    return render_template('data-overview.html', table=describe_table, head_table=head_table)


@app.route('/modelresults')
def modelresults():
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')
    plt.tight_layout()
    plt.savefig('static/images/prediction_plot.png')
    plt.close()

    return render_template('model-results.html', mse=mse, rmse=rmse, r2=r2)


if __name__ == '__main__':
    app.run(debug=True)
