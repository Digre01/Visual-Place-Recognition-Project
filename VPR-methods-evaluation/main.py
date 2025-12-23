import parser
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset


def main(args):
    start_time = datetime.now()

    # ------------------------------------------------------------
    # Logger setup
    # ------------------------------------------------------------
    logger.remove()
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}",
        level="INFO",
    )
    logger.add(log_dir / "info.log", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")

    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone "
        f"and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model = vpr_models.get_model(
        args.method, args.backbone, args.descriptors_dimension
    )
    model = model.eval().to(args.device)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    logger.info(f"Testing on {test_ds}")

    # ------------------------------------------------------------
    # Descriptor extraction
    # ------------------------------------------------------------
    with torch.inference_mode():

        logger.debug("Extracting database descriptors")
        database_subset = Subset(test_ds, list(range(test_ds.num_database)))
        database_loader = DataLoader(
            database_subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        all_descriptors = np.empty(
            (len(test_ds), args.descriptors_dimension), dtype="float32"
        )

        for images, indices in tqdm(database_loader):
            descriptors = model(images.to(args.device))
            all_descriptors[indices.numpy()] = descriptors.cpu().numpy()

        logger.debug("Extracting query descriptors")
        queries_subset = Subset(
            test_ds,
            list(
                range(
                    test_ds.num_database,
                    test_ds.num_database + test_ds.num_queries,
                )
            ),
        )
        queries_loader = DataLoader(
            queries_subset,
            batch_size=1,
            num_workers=args.num_workers,
        )

        for images, indices in tqdm(queries_loader):
            descriptors = model(images.to(args.device))
            all_descriptors[indices.numpy()] = descriptors.cpu().numpy()

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    # ------------------------------------------------------------
    # Save descriptors (optional)
    # ------------------------------------------------------------
    if args.save_descriptors:
        logger.info("Saving descriptors")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # ------------------------------------------------------------
    # kNN SEARCH
    # ------------------------------------------------------------

    # ORIGINAL VERSION (L2 distance – used in the NetVLAD paper)
    # faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)

    # NEW VERSION (Dot Product)
    # Used for CosPlace and experimentally tested for NetVLAD
    faiss_index = faiss.IndexFlatIP(args.descriptors_dimension)

    # NOTE:
    # Dot Product assumes L2-normalized descriptors.
    # NetVLAD descriptors are already normalized in the forward pass.
    faiss_index.add(database_descriptors)

    del database_descriptors, all_descriptors

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    logger.debug("Calculating recalls")
    distances, predictions = faiss_index.search(
        queries_descriptors, max(args.recall_values)
    )

    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))

        for query_idx, preds in enumerate(predictions):
            for i, k in enumerate(args.recall_values):
                if np.any(np.isin(preds[:k], positives_per_query[query_idx])):
                    recalls[i:] += 1
                    break

        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join(
            f"R@{k}: {r:.1f}" for k, r in zip(args.recall_values, recalls)
        )
        logger.info(recalls_str)

    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------
    if args.num_preds_to_save > 0:
        logger.info("Saving final predictions")
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save],
            test_ds,
            log_dir,
            args.save_only_wrong_preds,
            args.use_labels,
        )

    # ------------------------------------------------------------
    # Save uncertainty data
    # ------------------------------------------------------------
    if args.save_for_uncertainty:
        z_data = {
            "database_utms": test_ds.database_utms,
            "positives_per_query": positives_per_query,
            "predictions": predictions,
            "distances": distances,
        }
        torch.save(z_data, log_dir / "z_data.torch")


if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
