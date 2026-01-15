import os
import sys
import argparse
import torch
import time
import joblib  # Per caricare il modello di regressione logistica
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))

from matching import get_matcher, available_models
from matching.utils import get_default_device

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, required=True, help="directory with predictions of a VPR model")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory of image matching results")
    parser.add_argument("--matcher", type=str, default="sift-lg", choices=available_models, help="choose your matcher")
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--num-preds", type=int, default=20, help="number of predictions to match")
    parser.add_argument("--start-query", type=int, default=-1, help="query to start from")
    parser.add_argument("--num-queries", type=int, default=-1, help="number of queries")

    
    parser.add_argument("--adaptive-threshold", type=int, default=None, 
                        help="Se top-1 >= soglia, interrompe il matching")
    parser.add_argument("--logistic-model-path", type=str, default=None, 
                        help="Percorso al modello .joblib del regressore logistico")
    parser.add_argument("--logistic-prob-threshold", type=float, default=0.5, 
                        help="Soglia di probabilità per il regressore logistico")

    return parser.parse_args()

def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries

    output_folder = Path(args.preds_dir + f"_{args.matcher}") if args.out_dir is None else Path(args.out_dir)
    output_folder.mkdir(exist_ok=True, parents=True)

    #load model
    clf = joblib.load(args.logistic_model_path) if args.logistic_model_path else None

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    start_query = start_query if start_query >= 0 else 0
    num_queries = num_queries if num_queries >= 0 else len(txt_files)


    #time saving track
    start_time = time.time()
    total_matches_possible = len(txt_files) * args.num_preds
    matches_executed = 0

    for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
        out_file = output_folder.joinpath(f"{Path(txt_file).stem}.torch")
        if out_file.exists(): 
            continue
        
        results = []
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=args.im_size)
        
        for i, pred_path in enumerate(pred_paths[:args.num_preds]):
            img1 = matcher.load_image(pred_path, resize=args.im_size)
            result = matcher(deepcopy(img0), img1)
            result["all_desc0"] = result["all_desc1"] = None 
            results.append(result)
            matches_executed += 1

            if i == 0: #Top-1
                num_inliers = result['num_inliers']
                is_easy = False
                

                #without logistic model
                if args.adaptive_threshold is not None:
                    if num_inliers >= args.adaptive_threshold:
                        is_easy = True
                
                #with logistic model
                elif clf is not None:
                    prob = clf.predict_proba([[num_inliers]])[0][1]
                    if prob >= args.logistic_prob_threshold:
                        is_easy = True
                
                if is_easy:
                    break

        torch.save(results, out_file)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Match: {matches_executed} over {total_matches_possible}")

if __name__ == "__main__":
    main(parse_arguments())