import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--model-out", type=str, required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_path)
    if len(df) < 10:
        print(f"Error: Insufficient data in {args.csv_path}")
        return

    X = df[['num_inliers']].values
    y = df['label'].values
    
    # Training
    model = LogisticRegression()
    model.fit(X, y)
    
    joblib.dump(model, args.model_out)
    print(f"Model saved as {args.model_out}")

if __name__ == "__main__":
    main()