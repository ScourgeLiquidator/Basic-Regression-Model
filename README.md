# 🏠 California Housing Price Prediction using Random Forest Regressor

This project demonstrates how to build and use a machine learning model to predict **California housing prices** based on various features such as location, income, number of rooms, etc. The model uses a `RandomForestRegressor` trained on preprocessed data via a Scikit-learn pipeline.

---

## 📦 What's Inside?

- **Preprocessing with Pipelines**: Handles numerical and categorical features using `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
- **Model**: Trained using `RandomForestRegressor`.
- **Persistence**: Both the pipeline and model are saved using `joblib`.
- **Inference-ready**: Automatically loads the saved model to predict on new data.

---

## 📁 Project Structure
- **housing.csv**: Contains the data of California Housing price
- **master.csv**:  A smaller dataset for testing and comparing how closely my model is predicting the median house values.
- **input.csv**: Contains the features of houses which is to be predicted
- **output.csv**: Contains the predicted housing values which can be compared from master.csv
- **main.py**: Python program for applying all pre-processing, building the model and providing the final inference.
