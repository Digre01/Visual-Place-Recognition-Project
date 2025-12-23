import torch
import torch.nn as nn
import torchvision
import gdown
import os

MODELS_INFO = {
    128: (
        "https://drive.google.com/file/d/1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz/view",
        "resnet50_MixVPR_128_channels(64)_rows(2)",
        64,
        2,
    ),
    512: (
        "https://drive.google.com/file/d/1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj/view",
        "resnet50_MixVPR_512_channels(256)_rows(2)",
        256,
        2,
    ),
    4096: (
        "https://drive.google.com/file/d/1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L/view",
        "resnet50_MixVPR_4096_channels(1024)_rows(4)",
        1024,
        4,
    ),
}

# -----------------------------
# Backbone = RESNET50 (Truncated at Layer 3)
# -----------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        
        # Zgjidhja e problemit të emrave dhe dimensioneve:
        # Ne krijojmë një objekt 'model' që ruan emrat (conv1, layer1, etj.)
        # dhe heqim layer4 për të pasur output 1024 kanale.
        self.model = nn.Module()
        self.model.conv1 = resnet.conv1
        self.model.bn1 = resnet.bn1
        self.model.relu = resnet.relu
        self.model.maxpool = resnet.maxpool
        self.model.layer1 = resnet.layer1
        self.model.layer2 = resnet.layer2
        self.model.layer3 = resnet.layer3
        # layer4 nuk përfshihet sepse MixVPR origjinal e pret këtu

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x

# -----------------------------
# Aggregator
# -----------------------------
class FeatureMixerLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        return x + self.mix(x)


class MixVPR(nn.Module):
    def __init__(self, channels=1024, h=20, w=20,
                 out_channels=1024, out_rows=4,
                 depth=4, mlp_ratio=1):
        super().__init__()

        dim = h * w
        # FIX: Emri duhet të jetë 'mix' (jo mix_layers) për t'u përputhur me checkpoint
        self.mix = nn.Sequential(
            *[FeatureMixerLayer(dim, mlp_ratio) for _ in range(depth)]
        )

        self.channel_proj = nn.Linear(channels, out_channels)
        self.row_proj = nn.Linear(dim, out_rows)

    def forward(self, x):
        x = x.flatten(2)  # (B, C, HW)
        x = self.mix(x)   # Përdorim self.mix

        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)

        x = x.permute(0, 2, 1)
        x = self.row_proj(x)

        return nn.functional.normalize(x.flatten(1), p=2)

# -----------------------------
# Full model
# -----------------------------
class MixVPRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = Backbone()
        self.aggregator = MixVPR(**config)

    def forward(self, x):
        # Sigurohu që inputi të jetë i duhuri për dimensionet h=20, w=20 pas shtresës 3
        # ResNet layer3 bën downsampling 16x. 320 / 16 = 20. E saktë.
        x = nn.functional.interpolate(x, size=(320, 320))
        feats = self.backbone(x)
        return self.aggregator(feats)

# -----------------------------
# Loader
# -----------------------------
def get_mixvpr(descriptors_dimension):
    url, filename, out_channels, out_rows = MODELS_INFO[descriptors_dimension]

    config = {
        "channels": 1024,   # FIX: Ndryshuar nga 2048 në 1024 (Output i Layer 3)
        "h": 20,
        "w": 20,
        "out_channels": out_channels,
        "out_rows": out_rows,
        "depth": 4,
        "mlp_ratio": 1,
    }

    model = MixVPRModel(config)

    folder = "trained_models/mixvpr"
    os.makedirs(folder, exist_ok=True)
    file_path = f"{folder}/{filename}"

    if not os.path.exists(file_path):
        gdown.download(url=url, output=file_path, fuzzy=True)

    checkpoint = torch.load(file_path, map_location="cpu")
    model.load_state_dict(checkpoint)

    return model.eval()