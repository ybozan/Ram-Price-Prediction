import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor


def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"{col} -> lower_bound: {lower_bound}, upper_bound: {upper_bound}")
    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
    print(f"{col} sütunundaki aykırı değer sayısı: {outliers.sum()}")
    return df[~outliers]


df = pd.read_csv('data.csv')

num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df = remove_outliers_iqr(df, col)
    print(f"{col} için aykırı değerler çıkarıldı. Kalan satır sayısı: {len(df)}")
    print('---')

df = df.dropna()

categorical_features = ['Brand', 'DDR Type']
numerical_features = ['RAM Size', 'Speed', 'Score']




full_pipeline = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
x = df.drop(columns=["Price"])
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model_pipeline = Pipeline([
    ('preprocessing', full_pipeline),
    ('regressor', GradientBoostingRegressor())
])
model_pipeline.fit(x_train, y_train)

print(model_pipeline.score(x_test, y_test))

if __name__ == "__main__":
    joblib.dump(model_pipeline, 'model.pkl')
    print("✅ Model başarıyla model.pkl olarak kaydedildi.")

