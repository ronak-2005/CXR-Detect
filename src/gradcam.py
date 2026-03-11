import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._handles    = []
        self._register(target_layer)

    def _register(self, layer):
        self._handles.append(
            layer.register_forward_hook(self._save_activation)
        )
        self._handles.append(
            layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
       self.model.eval()
       img_tensor = img_tensor.unsqueeze(0).to(next(self.model.parameters()).device)

       logits = self.model(img_tensor)
       probs  = torch.softmax(logits, dim=1)

       if class_idx is None:
          class_idx = logits.argmax(dim=1).item()

       self.model.zero_grad()
       logits[0, class_idx].backward()

       if self.gradients is None:
        return np.zeros((img_tensor.shape[2], img_tensor.shape[3])), class_idx, probs[0, class_idx].item()

       weights = self.gradients.mean(dim=(2, 3), keepdim=True)
       cam     = (weights * self.activations).sum(dim=1, keepdim=True)
       cam     = F.relu(cam)
       cam     = F.interpolate(cam, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)
       cam     = cam.squeeze().cpu().numpy()
       cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

       return cam, class_idx, probs[0, class_idx].item()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()


def render_overlay(img_pil, cam, img_size, alpha=0.4):
    img_resized = img_pil.resize((img_size, img_size)).convert("RGB")
    img_np      = np.array(img_resized) / 255.0
    heatmap     = cm.jet(cam)[:, :, :3]
    overlay     = np.clip(alpha * heatmap + (1 - alpha) * img_np, 0, 1)
    return img_np, heatmap, overlay


def plot_gradcam(img_np, cam, overlay, title="", save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(img_np);          axes[0].set_title("Input");    axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    axes[2].imshow(overlay);         axes[2].set_title(title);      axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig