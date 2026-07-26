import sys
import json
import joblib
import pandas as pd
import argparse
import os

def load_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    
    metrics_path = os.path.join(models_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}. Run train_model.py first.")
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    best_model_name = max(metrics, key=lambda k: metrics[k]['f1_score'])
    best_model_filename = best_model_name.replace(' ', '_').lower() + '.pkl'
    
    model_path = os.path.join(models_dir, best_model_filename)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    le_path = os.path.join(models_dir, 'label_encoder.pkl')
    features_path = os.path.join(models_dir, 'feature_columns.json')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    
    with open(features_path, 'r') as f:
        feature_cols = json.load(f)
        
    return model, scaler, le, feature_cols

def predict_single(data, model, scaler, le, feature_cols):
    try:
        df = pd.DataFrame([data])
        
        for col in feature_cols:
            if col not in df.columns:
                return {"error": f"Missing feature: {col}"}
                
        X = df[feature_cols]
        
        X_scaled = scaler.transform(X)
        
        pred_idx = model.predict(X_scaled)[0]
        prediction = le.inverse_transform([pred_idx])[0]
        
        probs = model.predict_proba(X_scaled)[0]
        classes = le.classes_
        
        prob_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
        confidence = prob_dict[prediction]
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": prob_dict
        }
    except Exception as e:
        return {"error": str(e)}

def predict_batch(data_list, model, scaler, le, feature_cols):
    results = []
    for data in data_list:
        results.append(predict_single(data, model, scaler, le, feature_cols))
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="5G Network Traffic ML Predictor")
    parser.add_argument('--batch', action='store_true', help="Accept an array of JSON objects")
    args = parser.parse_args()
    
    input_str = sys.stdin.read().strip()
    
    try:
        if not input_str:
            raise ValueError("No input provided. Pipe JSON data to stdin.")
            
        input_data = json.loads(input_str)
        
        model, scaler, le, feature_cols = load_pipeline()
        
        if args.batch:
            if not isinstance(input_data, list):
                raise ValueError("--batch flag used but input is not a JSON array.")
            result = predict_batch(input_data, model, scaler, le, feature_cols)
        else:
            if isinstance(input_data, list):
                raise ValueError("Input is a JSON array but --batch flag is not used.")
            result = predict_single(input_data, model, scaler, le, feature_cols)
            
        print(json.dumps(result, indent=2))
        
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON format: {str(e)}"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
