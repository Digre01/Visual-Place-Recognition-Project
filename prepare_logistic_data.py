import torch
import numpy as np
import pandas as pd
from pathlib import Path
from util import get_list_distances_from_preds
import argparse
from glob import glob
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, required=True)
    parser.add_argument("--inliers-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    args = parser.parse_args()

    data = []
    txt_files = sorted(glob(os.path.join(args.preds_dir, "*.txt")))
    
    for txt_file in txt_files:
        query_id = Path(txt_file).stem
        
        #ground truth
        dists = get_list_distances_from_preds(txt_file)
        
        #matching result
        torch_file = Path(args.inliers_dir) / f"{query_id}.torch"
        if not torch_file.exists(): continue
        
        results = torch.load(torch_file, weights_only=False)
        if len(results) > 0:
            num_inliers = results[0]['num_inliers'] #save only the Top-1
            
            label = 1 if dists[0] <= 25 else 0 #assign 1 if the Top-1 is label correctly
            data.append([num_inliers, label])
            
    df = pd.DataFrame(data, columns=['num_inliers', 'label'])
    df.to_csv(args.output_csv, index=False)
    print(f"Dataset created with {len(df)} values -> {args.output_csv}")

if __name__ == "__main__":
    main()