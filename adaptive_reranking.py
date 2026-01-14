import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch
from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, required=True, help="directory with VPR predictions")
    parser.add_argument("--inliers-dir", type=str, required=True, help="directory with matching results")
    parser.add_argument("--num-preds", type=int, default=20, help="max predictions to consider")
    parser.add_argument("--positive-dist-threshold", type=int, default=25, help="meters for positive")
    parser.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    txt_files = sorted(glob(os.path.join(preds_folder, "*.txt")), key=lambda x: int(Path(x).stem))

    recalls = np.zeros(len(args.recall_values))
    total_matchings_possible = len(txt_files) * args.num_preds
    matchings_actually_done = 0

    for txt_file_query in tqdm(txt_files):
        # 1. Carica distanze geografiche originali
        geo_dists_orig = torch.tensor(get_list_distances_from_preds(txt_file_query))[:args.num_preds]
        
        # 2. Carica risultati matching
        torch_file_query = inliers_folder / f"{Path(txt_file_query).stem}.torch"
        query_results = torch.load(torch_file_query, weights_only=False)
        
        num_matchings_this_query = len(query_results)
        matchings_actually_done += num_matchings_this_query
        
        # 3. Logica di Re-ranking
        if num_matchings_this_query > 1:
            inliers = torch.tensor([r['num_inliers'] for r in query_results], dtype=torch.float32)
            _, sort_idx = torch.sort(inliers, descending=True)
            
            geo_dists_top = geo_dists_orig[:num_matchings_this_query][sort_idx]
            geo_dists_final = torch.cat([geo_dists_top, geo_dists_orig[num_matchings_this_query:]])
        else:
            # "Easy"
            geo_dists_final = geo_dists_orig

        # 4. Update Recalls
        for i, n in enumerate(args.recall_values):
            if torch.any(geo_dists_final[:n] <= args.positive_dist_threshold):
                recalls[i:] += 1
                break

    # Calcolo Cost Savings
    cost_saving = (1 - (matchings_actually_done / total_matchings_possible)) * 100
    recalls = recalls / len(txt_files) * 100
    
    print("-" * 30)
    print(f"ADAPTIVE RE-RANKING RESULTS")
    print(f"Cost Savings: {cost_saving:.2f}%")
    print(", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)]))
    print("-" * 30)

if __name__ == "__main__":
    main(parse_arguments())