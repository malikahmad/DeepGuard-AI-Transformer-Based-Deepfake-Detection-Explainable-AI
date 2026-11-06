import os
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_SIZE = 224
CLASS_NAMES = ["Fake", "Real"]  # Must match notebook mapping: Fake=0, Real=1

def build_model(pretrained=False):
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=2
    )

def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device

def inference_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

def reshape_transform(tokens):
    # ViT output: [B, 1 + number_of_patches, channels]
    tokens = tokens[:, 1:, :]
    batch, patches, channels = tokens.shape
    side = int(patches ** 0.5)
    if side * side != patches:
        raise ValueError(f"Expected square ViT patch grid, got {patches} patches")
    return tokens.reshape(batch, side, side, channels).permute(0, 3, 1, 2)

def predict_with_explanation(model, device, pil_image):
    image = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    transform = inference_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index]) * 100.0

    # Grad-CAM++ on the final ViT block. For ViT, reshape patch tokens into a 2D grid.
    target_layer = model.blocks[-1].norm1
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None
    )[0]

    rgb = np.asarray(image).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 2),
        "probabilities": {
            "Fake": round(float(probabilities[0]) * 100, 2),
            "Real": round(float(probabilities[1]) * 100, 2)
        },
        "heatmap": (np.clip(grayscale_cam * 255, 0, 255)).astype(np.uint8),
        "overlay": overlay
    }
