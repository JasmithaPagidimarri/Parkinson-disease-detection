# Parkinson's Disease Detection

# ✅ Importing the necessary Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,roc_auc_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score

import warnings
warnings.filterwarnings('ignore')
sns.set() # Seaborn's style for plots

# ✅ Install opendatasets and download the dataset
{"username":"dhanyasriakella","key":"23883ce0831e3d3d889cce1445ded473"}
# dpip install opendatasets --quiet 
import opendatasets

dataset_url = 'https://www.kaggle.com/datasets/thecansin/parkinsons-data-set?datasetId=409297&sortBy=voteCount'
opendatasets.download(dataset_url)

# ✅ Read data
data_path = './parkinsons-data-set/parkinsons.data'
df = pd.read_csv(data_path) # Loads CSV files to panda DataFrame

# ✅ Drop name column
df = df.drop('name', axis=1)

# ✅ Visualize target distribution
df['status'].value_counts().plot(kind='bar')
plt.title('Distribution of Target variable')
plt.show() # No. of Healthy vs Parkinson's samples: Bar plot

df['status'].value_counts().plot(kind='pie', autopct='%.f%%')
plt.title('Target variable distribution')
plt.show() # Pie chart

# ✅ Plot function for EDA
def plots(plot_kind, dataframe):
    plot_kind = plot_kind.lower()
    plot_func = {
        'violin': sns.violinplot,
        'box': sns.boxplot,
        'histogram': sns.histplot,
        'kde': sns.kdeplot
    }

    figure = plt.figure(figsize=(25, 16)) # Makes it tall and wide for all subplots to fit
    for index, column in enumerate(dataframe.columns):
        axis = figure.add_subplot(5, 5, index + 1) # Creats 5x5 grid subplots
        if plot_kind in ['violin', 'box']:
            plot_func[plot_kind](y=dataframe[column], ax=axis) # Data passing in Violin &Box
        else:
            plot_func[plot_kind](dataframe[column], ax=axis) # Data passing in Histogram and KDE
        axis.set_title(f'{plot_kind.capitalize()} plot for {column}') # Naming of each subplot
    plt.tight_layout() ##Makes sure plots don't Overlap
    plt.show()

# ✅ EDA Plots
plots('violin', df) # Shows distribution, density, and summary statistics all in one.
plots('box', df) # Summarizes the distribution of data
plots('histogram', df) # Shows frequency distribution
plots('kde', df) # Shows probability distribution

# ✅ Features and target separation
features = df.drop('status', axis=1)
target = df['status']

# ✅ Train-Test Split
train_data, test_data, train_labels, test_labels = train_test_split(
    features, target, stratify=target, test_size=0.2, random_state=2)

# ✅ Data Normalization with MinMaxScaler, i.e; scales features to the range [0,1] for better model performance
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data_scaled = scaler.transform(train_data)
test_data_scaled = scaler.transform(test_data)

# ✅ Optimized Feature Selection for Random Forest
from sklearn.feature_selection import SelectKBest, f_classif

# Test different numbers of features to find optimal for Random Forest
feature_range = range(5, len(features.columns) + 1, 2)  # Test for k=5, 7, 9, ..., all features
rf_scores = [] #List to store cross Validation scoresfor each value of k
best_k = 5 #Initialization of Best k value as '5'
best_score = 0 #Best score used to find best_k value in the end 

print("🔍 Testing different numbers of features for Random Forest optimization:")

for k in feature_range:
    # Select k best features
    selector = SelectKBest(score_func=f_classif, k=k) #initializes SelectKBest 
    train_selected = selector.fit_transform(train_data_scaled, train_labels) #fits SelectKBest to  training data & helps to select top  'k' features
    
    # Test Random Forest with current feature set using cross-validation
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  #Initializes temporary Random Forest
    cv_scores = cross_val_score(rf_temp, train_selected, train_labels, cv=5, scoring='accuracy')
    avg_score = cv_scores.mean()
    
    rf_scores.append(avg_score)
    print(f"Features: {k:2d} | RF CV Accuracy: {avg_score*100:.2f}") #Prints no. of features and its accuracy
    
    if avg_score > best_score: #checks if current score is better than best_score
        best_score = avg_score #Updates best score
        best_k = k #Updates best_k value

print(f"\n🎯 Optimal number of features for Random Forest: {best_k}")

# Plot feature selection results for Random Forest
plt.figure(figsize=(12, 6))
plt.plot(feature_range, rf_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.xlabel('Number of Features')
plt.ylabel('Random Forest CV Accuracy')
plt.title('Feature Selection Optimization for Random Forest')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Apply optimal feature selection
selector_optimal = SelectKBest(score_func=f_classif, k=best_k) #Initializes SelectKBest for optimal k
train_data_optimal = selector_optimal.fit_transform(train_data_scaled, train_labels) 
test_data_optimal = selector_optimal.transform(test_data_scaled)

# Get selected 'k' features names for reference
feature_names = features.columns.tolist()
selected_features = selector_optimal.get_support()
selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
print(f"\n📋 Selected features ({len(selected_feature_names)}): {selected_feature_names}")

# ✅ Tune RandomForestClassifier with GridSearchCV using optimal features
param_grid = {
    'n_estimators': [100, 150, 200], #No.of trees in forest
    'max_depth': [None, 10, 15, 20], #Max. depth of tree
    'min_samples_split': [2, 5, 10], #Min. no. of samples to split internal node
    'min_samples_leaf': [1, 2, 4] #Min no.  of samples to be at leaf node
}

print("\n🔧 Performing hyperparameter tuning for Random Forest...")
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1) #Initializes GridSearchCV
grid.fit(train_data_optimal, train_labels)

best_rf_model = grid.best_estimator_ #Gets best Random Forest model found by GridSearchCV
print(f"🎯 Best Random Forest parameters: {grid.best_params_}") #Prints Best parameters

# Predict and evaluate
log_prediction = best_rf_model.predict(test_data_optimal)
log_accuracy = accuracy_score(log_prediction, test_labels)
print(f'🔍 Accuracy of Optimized RandomForest: {round(log_accuracy*100, 2)}%')

# ✅ Confusion Matrix with ConfusionMatrixDisplay
log_confusion = confusion_matrix(test_labels, log_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=log_confusion)
disp.plot()
plt.title("Confusion Matrix - Optimized RandomForest")
plt.show()

# ✅ Testing multiple models with optimal feature selection
models = [
    SVC(kernel='linear'), # Doesn't work well for non-linear data
    KNeighborsClassifier(), # Slow for large data
    RandomForestClassifier(), # Handles non-linear data
    XGBClassifier() # It might overfit and can be slower
]

model_results = []

def best_model(model_list):
    for model in model_list:
        model.fit(train_data_optimal, train_labels) # Model learns from data
        prediction = model.predict(test_data_optimal) # Uses trained model to predict as 0 or 1 on test set
        accuracy = accuracy_score(prediction, test_labels)
        #Save model result
        model_results.append({
            'Model Name': type(model).__name__,
            'Model Accuracy Score': round(accuracy * 100, 2)
        })
    return pd.DataFrame(model_results).sort_values(
        by='Model Accuracy Score', ascending=False) # Coverts results of model to a panda DataFrame

# ✅ Show results
results_df = best_model(models)
print("\nModel Accuracy Comparison (with optimized features):")
print(results_df)

# ✅ Feature importance analysis

feature_importance = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': best_rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
"""
print(f"\n📊 Top 13 Most Important Features:")
print(feature_importance.head(13))
"""
# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(13), x='Importance', y='Feature')
plt.title('Top 13 Feature Importance - Optimized Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ✅ Prediction function for new data
def predict_parkinsons(input_features):
    """
    Predict Parkinson's disease from voice features
    input_features: list or array of 22 voice features
    """
    # Convert to numpy array and reshape
    new_data = np.array([input_features])
    
    # Apply same preprocessing
    new_data_scaled = scaler.transform(new_data)
    new_data_selected = selector_optimal.transform(new_data_scaled)
    
    # Make prediction
    prediction = best_rf_model.predict(new_data_selected)[0]
    probability = best_rf_model.predict_proba(new_data_selected)[0]
    result = "Has Parkinson's" if prediction == 1 else "Healthy"
    return result

# If the person has parkinson's or not
user_input = input("Enter 22 comma-separated voice features: \n")
input_features = list(map(float, user_input.split(',')))
result = predict_parkinsons(input_features)
print(f"Result: {result}")