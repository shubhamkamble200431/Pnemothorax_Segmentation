import cv2
import torch
import numpy as np

from models import PTXSegNet


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):

    model = PTXSegNet().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    return model


def predict(model, img_path, size=512):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))

    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img)[0, 0].cpu().numpy()

    return (pred > 0.5).astype(np.uint8)


if __name__ == "__main__":

    model = load_model("checkpoints/ptxsegnet_best.pth")

    mask = predict(model, "sample.png")

    cv2.imwrite("prediction.png", mask * 255)