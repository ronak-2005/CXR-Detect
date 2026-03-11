import torch
import torch.nn as nn
from torchvision import models


class PneumoniaNet(nn.Module):
    def __init__(self, num_classes, dropout, pretrained):
        super().__init__()

        weights   = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone  = models.resnet50(weights=weights)

        for param in backbone.parameters():
            param.requires_grad = False

        backbone.fc  = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def unfreeze(self, layers):
        for name, child in self.backbone.named_children():
            if name in layers:
                for param in child.parameters():
                    param.requires_grad = True

    def get_gradcam_layer(self):
        return self.backbone.layer4[-1]

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(cfg, device):
    model = PneumoniaNet(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)
    return model


def save_checkpoint(model, optimizer, epoch, val_auc, path, is_best=False):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "val_auc":     val_auc,
    }
    torch.save(state, path)
    if is_best:
        best_path = Path(path).parent / "best.pt"
        torch.save(state, best_path)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def export_onnx(model, save_path, img_size, device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    torch.onnx.export(
        model,
        dummy,
        save_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"ONNX exported to {save_path}")