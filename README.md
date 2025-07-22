# 🏠 California Housing Price Prediction using Random Forest Regressor

This project demonstrates how to build and use a machine learning model to predict **California housing prices** based on various features such as location, income, number of rooms, etc. The model uses a `RandomForestRegressor` trained on preprocessed data via a Scikit-learn pipeline.

---

## 📦 What's Inside?

- **Preprocessing with Pipelines**: Handles numerical and categorical features using `ColumnTransformer`, `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
- **Model**: Trained using `RandomForestRegressor`.
- **Stratified Split**: Uses `StratifiedShuffleSplit` to maintain the distribution of key categorical features across training and test sets, reducing sampling bias.
- **Persistence**: Both the pipeline and model are saved using `joblib`.
- **Inference-ready**: Automatically loads the saved model to predict on new data.
---

## 📁 Project Structure

- **housing.csv**:  
  The primary dataset containing California housing data, including various features and the target variable `median_house_value`. This dataset is used for training and validating the machine learning model.

- **master.csv**:  
  A smaller, curated subset of data intended for evaluating model performance. It serves as a reference for comparing predicted values to actual median house values, allowing manual inspection and error analysis.

- **input.csv**:  
  A dataset consisting of feature values for previously unseen housing instances. These are the input records on which the trained model will perform inference (i.e., predict the median house values).

- **output.csv**:  
  The file where the predicted `median_house_value` results are stored. This output can be compared directly with `master.csv` for validating the model's real-world predictive accuracy.

- **main.py**:  
  The central Python script responsible for:
  - Loading and preprocessing the data (handling missing values, encoding, scaling, etc.)
  - Training the regression model on `housing.csv`
  - Performing predictions on data from `input.csv`
  - Exporting the results to `output.csv`
  - (Optionally) Comparing predictions with `master.csv` to assess model accuracy

