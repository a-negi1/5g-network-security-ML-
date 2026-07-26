import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate():
    print("Starting model training pipeline...")
    
    os.makedirs('models', exist_ok=True)
    
    data_path = os.path.join('data', '5g_traffic_dataset.csv')
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found. Run generate_dataset.py first.")
        return
        
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    feature_cols = X.columns.tolist()
    with open(os.path.join('models', 'feature_columns.json'), 'w') as f:
        json.dump(feature_cols, f)
        
    print("Preprocessing data...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(le, os.path.join('models', 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    metrics_dict = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics_dict[name] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        }
        
        model_filename = name.replace(' ', '_').lower() + '.pkl'
        joblib.dump(model, os.path.join('models', model_filename))
        
    with open(os.path.join('models', 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print("\nModel Comparison Summary:")
    print("-" * 65)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 65)
    for name, metrics in metrics_dict.items():
        print(f"{name:<15} | {metrics['accuracy']:.4f}     | {metrics['precision']:.4f}      | {metrics['recall']:.4f}     | {metrics['f1_score']:.4f}")
    print("-" * 65)
    
    print("\nGenerating comparison plot...")
    plot_data = []
    for model_name, metrics in metrics_dict.items():
        for metric_name, value in metrics.items():
            plot_data.append({'Model': model_name, 'Metric': metric_name.capitalize(), 'Score': value})
            
    plot_df = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'model_comparison.png'))
    print(f"Saved plot to models/model_comparison.png")
    
    print("Pipeline completed successfully.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    train_and_evaluate()
