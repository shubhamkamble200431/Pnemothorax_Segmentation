import torch
import cv2
import numpy as np
from models.unet_effnet_b0 import UNetEffNetB0


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(image_path):

    model = UNetEffNetB0(pretrained=False)
    model.load_state_dict(torch.load("checkpoints/unet_effnet_b0.pth"))
    model.to(DEVICE).eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)/255.0

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(img))[0,0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8)

    cv2.imwrite("prediction.png", mask*255)


if __name__ == "__main__":
    predict("sample.png")