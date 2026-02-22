import torch
import matplotlib.pyplot as plt


def visualize_positive_samples(dataset, model_builders, device, num_samples=10):
    positive_indices = []

    for i, (_, mask) in enumerate(dataset):
        if mask.sum() > 0:
            positive_indices.append(i)
        if len(positive_indices) >= num_samples:
            break

    for idx in positive_indices:
        img, mask = dataset[idx]
        img_disp = img.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(12, 3))
        plt.subplot(1, len(model_builders) + 2, 1)
        plt.imshow(img_disp)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, len(model_builders) + 2, 2)
        plt.imshow(mask[0].cpu(), cmap="gray")
        plt.title("GT")
        plt.axis("off")

        col = 3

        for name, builder in model_builders.items():
            model = builder().to(device)
            model.load_state_dict(torch.load(f"checkpoints/{name}_best.pth"))
            model.eval()

            with torch.no_grad():
                pred = (torch.sigmoid(model(img.unsqueeze(0).to(device)))[0] > 0.5).float()

            plt.subplot(1, len(model_builders) + 2, col)
            plt.imshow(pred[0].cpu(), cmap="gray")
            plt.title(name)
            plt.axis("off")
            col += 1

        plt.show()