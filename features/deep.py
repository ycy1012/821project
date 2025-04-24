import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Load pretrained ResNet18 and remove classification head
weights = ResNet18_Weights.DEFAULT
_resnet = resnet18(weights=weights)
_model = torch.nn.Sequential(*list(_resnet.children())[:-1])  # remove FC layer
_model.eval()

# Define input transform (for single-channel gray images → 3-channel normalized tensor)
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts [H, W] or PIL Image to [C, H, W]
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_deep_features(img: np.ndarray) -> np.ndarray:
    """
    Extract deep CNN embedding (e.g., 512-d) from a preprocessed grayscale image.
    
    Parameters:
        img (np.ndarray): 2D float32 grayscale image, shape (H, W), values in [0, 1]

    Returns:
        np.ndarray: 1D feature vector (e.g., shape (512,))
    """
    if img.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    # Convert numpy to PIL Image
    pil_img = Image.fromarray((img * 255).astype(np.uint8))

    # Transform and add batch dimension
    input_tensor = _transform(pil_img).unsqueeze(0)  # shape: [1, 3, 224, 224]

    # Run through model
    with torch.no_grad():
        output = _model(input_tensor)  # shape: [1, 512, 1, 1]
        embedding = output.squeeze().cpu().numpy()   # shape: (512,)

    return embedding
