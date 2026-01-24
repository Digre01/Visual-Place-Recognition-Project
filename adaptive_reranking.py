import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch
from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, required=True, help="directory con le predizioni VPR (.txt)")
    parser.add_argument("--inliers-dir", type=str, required=True, help="directory con i risultati del matching (.torch)")
    parser.add_argument("--num-preds", type=int, default=20, help="numero massimo di candidati considerati")
    parser.add_argument("--positive-dist-threshold", type=int, default=25, help="soglia in metri per un match positivo")
    parser.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    
    txt_files = sorted(glob(os.path.join(preds_folder, "*.txt")), 
                       key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem)

    total_queries = len(txt_files)
    recalls = np.zeros(len(args.recall_values))
    retrieval_correct_r1 = 0
    
    total_matchings_possible = total_queries * args.num_preds
    matchings_actually_done = 0

    print(f"Eval over {total_queries} query...")

    for txt_file_query in tqdm(txt_files):
        geo_dists_orig = torch.tensor(get_list_distances_from_preds(txt_file_query))[:args.num_preds]
        
        # Baseline: 1 if Top-1 < 25m
        if geo_dists_orig[0] <= args.positive_dist_threshold:
            retrieval_correct_r1 += 1
        
        torch_file_query = inliers_folder / f"{Path(txt_file_query).stem}.torch"
        
        if not torch_file_query.exists():
            geo_dists_final = geo_dists_orig
        else:
            query_results = torch.load(torch_file_query, weights_only=False)
            num_matchings_this_query = len(query_results)
            matchings_actually_done += num_matchings_this_query
            
            if num_matchings_this_query > 1:
                inliers = torch.tensor([r['num_inliers'] for r in query_results], dtype=torch.float32)
                _, sort_idx = torch.sort(inliers, descending=True)
                
                
                geo_dists_top = geo_dists_orig[:num_matchings_this_query][sort_idx]
                geo_dists_final = torch.cat([geo_dists_top, geo_dists_orig[num_matchings_this_query:]])
            else:
                #Easy case: don't change the order
                geo_dists_final = geo_dists_orig

        for i, n in enumerate(args.recall_values):
            if torch.any(geo_dists_final[:n] <= args.positive_dist_threshold):
                recalls[i:] += 1
                break

    cost_saving = (1 - (matchings_actually_done / total_matchings_possible)) * 100
    retrieval_r1_pct = (retrieval_correct_r1 / total_queries) * 100
    recalls_pct = (recalls / total_queries) * 100
    
    print("\n" + "="*45)
    print(f"RESULTS RE-RANKING ADAPTIVE")
    print("="*45)
    print(f"Baseline (Retrieval-only) R@1: {retrieval_r1_pct:.2f}%")
    print(f"Adaptive Matching R@1:       {recalls_pct[0]:.2f}%")
    print(f"Computational cost saving:    {cost_saving:.2f}%")
    print("-" * 45)
    #print("OTHER DATA:")
    #print("Recall Adattive:")
    #for val, rec in zip(args.recall_values, recalls_pct):
    #    print(f" R@{val}: {rec:.2f}%")
    #print("="*45)

if __name__ == "__main__":
    main(parse_arguments())