# Part of this code are taken from https://github.com/cvg/Hierarchical-Localization
# and https://github.com/naver/deep-image-retrieval

import os
import sys
from pathlib import Path
from zipfile import ZipFile

import gdown
import sklearn
import torch
import argparse
from sklearn.decomposition import PCA
import torch.serialization

# ---------------------------------------------------------
# FIX 1: Allowlist needed globals for PyTorch 2.6+
# ---------------------------------------------------------
torch.serialization.add_safe_globals([argparse.Namespace])
torch.serialization.add_safe_globals([PCA])

# DIR checkpoints reference sklearn.decomposition.pca.PCA
sys.modules["sklearn.decomposition.pca"] = sklearn.decomposition._pca

# ---------------------------------------------------------
# Add deep-image-retrieval into Python path
# ---------------------------------------------------------
sys.path.append(str(
    Path(__file__).parent.parent
        .joinpath("third_party")
        .joinpath("deep-image-retrieval")
))

# Required environment variable for DIR
os.environ["DB_ROOT"] = ""

# Imports from DIR
from dirtorch.extract_features import load_model  # noqa: E402
from dirtorch.utils import common                # noqa: E402

class GeM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conf = {
            "model_name": "Resnet-101-AP-GeM",
            "whiten_name": "Landmarks_clean",
            "whiten_params": {
                "whitenp": 0.25,
                "whitenv": None,
                "whitenm": 1.0,
            },
            "pooling": "gem",
            "gemp": 3,
        }

        dir_models = {
            "Resnet-101-AP-GeM":
                "https://docs.google.com/uc?export=download&id=1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy"
        }

        checkpoint = Path(
            torch.hub.get_dir(),
            "dirtorch",
            self.conf["model_name"] + ".pt"
        )

        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True, parents=True)
            link = dir_models[self.conf["model_name"]]
            zip_path = str(checkpoint) + ".zip"

            gdown.download(link, zip_path, quiet=False)

            with ZipFile(zip_path, "r") as zf:
                zf.extractall(checkpoint.parent)

            os.remove(zip_path)

        # IMPORTANT: load_model() must remain unchanged
        self.net = load_model(str(checkpoint), iscuda=False)

        if self.conf["whiten_name"]:
            assert self.conf["whiten_name"] in self.net.pca

    def forward(self, image):
        descs = self.net(image)

        if len(descs.shape) == 1:
            descs = descs.unsqueeze(0)

        if self.conf["whiten_name"]:
            whitened = []
            pca = self.net.pca[self.conf["whiten_name"]]

            for desc in descs:
                desc = desc.unsqueeze(0)
                desc = common.whiten_features(
                    desc.cpu().numpy(), pca, **self.conf["whiten_params"]
                )
                desc = torch.from_numpy(desc)[0]
                whitened.append(desc)

            descs = torch.stack(whitened)

        return descs

