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
    # Ordiniamo i file per matchare i risultati
    txt_files = sorted(glob(os.path.join(args.preds_dir, "*.txt")))
    
    for txt_file in txt_files:
        query_id = Path(txt_file).stem
        # Distanze reali (Ground Truth)
        dists = get_list_distances_from_preds(txt_file)
        
        # Risultati del matching
        torch_file = Path(args.inliers_dir) / f"{query_id}.torch"
        if not torch_file.exists(): continue
        
        # Carichiamo i dati (usiamo il primo match per il training del Top-1 exit)
        results = torch.load(torch_file, weights_only=False)
        if len(results) > 0:
            num_inliers = results[0]['num_inliers']
            # Etichetta 1 se la distanza geografica è <= 25 metri
            label = 1 if dists[0] <= 25 else 0
            data.append([num_inliers, label])
            
    df = pd.DataFrame(data, columns=['num_inliers', 'label'])
    df.to_csv(args.output_csv, index=False)
    print(f"Dataset creato con {len(df)} campioni -> {args.output_csv}")

if __name__ == "__main__":
    main()