import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch

from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to re-rank")
    parser.add_argument(
        "--save-reranked-dir",
        type=str,
        default=None,
        help=(
            "if set, saves re-ranked prediction files to this directory (same filenames as input, top-N reordered)"
        ),
    )
    parser.add_argument(
        "--save-reranked-inliers-dir",
        type=str,
        default=None,
        help=(
            "if set, saves re-ranked .torch image-matching results to this directory (top-N reordered to match preds)"
        ),
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 100],
        help="values for recall (e.g. recall@1, recall@5)",
    )

    return parser.parse_args()

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values
    save_dir = Path(args.save_reranked_dir) if args.save_reranked_dir else None
    
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    inliers_save_dir = Path(args.save_reranked_inliers_dir) if getattr(args, "save_reranked_inliers_dir", None) else None
    if inliers_save_dir is not None:
        inliers_save_dir.mkdir(parents=True, exist_ok=True)

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)
    recalls = np.zeros(len(recall_values))

    for txt_file_query in tqdm(txt_files):
        geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace('txt', 'torch'))
        query_results = torch.load(torch_file_query, weights_only=False)
        query_db_inliers = torch.zeros(num_preds, dtype=torch.float32)
        
        for i in range(num_preds):
            query_db_inliers[i] = query_results[i]['num_inliers']
        query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
        geo_dists = geo_dists[indices]

        # Print the number of inliers for the top-1 match
        top1_inliers = int(query_db_inliers[0].item()) if query_db_inliers.numel() > 0 else 0
        print(f"Query {Path(txt_file_query).stem}: top-1 inliers = {top1_inliers}")

        # Optionally save the re-ranked predictions to disk
        if save_dir is not None:
            # Read original file structure to preserve headers and formatting
            with open(txt_file_query, 'r') as f:
                lines = f.read().splitlines()

            # Identify predictions block [start:end)
            start_idx = 4  # by convention in util.read_file_preds
            try:
                end_idx = lines.index('', start_idx)
            except ValueError:
                end_idx = len(lines)

            # Original predictions as read in util.get_list_distances_from_preds
            # Reorder only the top-N predictions, keep the tail as-is
            orig_pred_paths = lines[start_idx:end_idx]
            top = orig_pred_paths[:num_preds]
            tail = orig_pred_paths[num_preds:]
            # indices is a tensor of sorted positions for the top-N
            reordered_top = [top[i] for i in indices.cpu().numpy().tolist()]
            new_pred_paths = reordered_top + tail

            # Reconstruct the file keeping preamble and trailing content
            new_lines = lines[:start_idx] + new_pred_paths + lines[end_idx:]
            # Ensure there's an empty line after predictions block
            if len(new_lines) == 0 or new_lines[-1] != '':
                new_lines.append('')

            out_path = save_dir / Path(txt_file_query).name
            with open(out_path, 'w') as f:
                f.write("\n".join(new_lines))

        # Optionally save the re-ranked inliers (.torch) to disk
        if inliers_save_dir is not None:
            # Reorder only top-N entries of the inliers results according to the same indices
            if isinstance(query_results, (list, tuple)):
                top_inliers = list(query_results[:num_preds])
                tail_inliers = list(query_results[num_preds:])
                reordered_top_inliers = [top_inliers[i] for i in indices.cpu().tolist()]
                new_inliers_results = reordered_top_inliers + tail_inliers
            else:
                # If unexpected type, fall back to original object without modification
                new_inliers_results = query_results

            out_inliers_path = inliers_save_dir / Path(torch_file_query).name
            torch.save(new_inliers_results, out_inliers_path)
        
        for i, n in enumerate(recall_values):
            if torch.any(geo_dists[:n] <= threshold):
                recalls[i:] += 1
                break

    recalls = recalls / total_queries * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])

    print(recalls_str)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)