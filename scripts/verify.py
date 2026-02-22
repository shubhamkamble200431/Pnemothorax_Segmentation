def verify_model(model, dataset, device, thr=0.5, n=5):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        c = 0
        for img, mask, _, path in loader:
            if mask.sum() == 0:
                continue
            img = img.to(device)
            logit = model(img)
            if logit.shape[2:] != mask.shape[2:]:
                logit = F.interpolate(logit, size=mask.shape[2:], mode="bilinear", align_corners=False)

            prob = torch.sigmoid(logit)
            pred = (prob > thr).float()

            print(os.path.basename(path[0]),
                  float(logit.min()), float(logit.max()),
                  int(mask.sum()), int(pred.sum()))

            c += 1
            if c >= n:
                break


def main():
    results = []

    for mcfg in cfg.models:
        if not os.path.isfile(mcfg["ckpt_path"]):
            continue

        transforms = get_test_transforms(mcfg["img_size"])
        dataset = PneumothoraxDataset(
            cfg.dataset_root,
            split="test",
            transforms=transforms,
            img_size=mcfg["img_size"]
        )

        try:
            model = create_model(mcfg["model_name"], pretrained=False).to(cfg.device)
            ckpt = torch.load(mcfg["ckpt_path"], map_location=cfg.device)

            if isinstance(ckpt, dict):
                sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            else:
                sd = ckpt

            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)

            verify_model(model, dataset, cfg.device, cfg.threshold)

        except Exception:
            continue

        files, dice, preds, img_map = compute_dice_per_image(
            model, dataset, batch_size=cfg.batch_size
        )

        if sum(p.sum() for p in preds.values()) == 0:
            continue

        masks = {}
        for p in files:
            mp = dataset._get_mask_path(p)
            m = cv2.imread(mp, 0)
            m = cv2.resize(m, (mcfg["img_size"], mcfg["img_size"]))
            masks[p] = (m > 0).astype(np.uint8)

        print(mcfg["name"],
              np.mean(dice),
              np.std(dice),
              np.min(dice),
              np.max(dice),
              np.median(dice))

        results.append({
            "name": mcfg["name"],
            "file_list": files,
            "dice_list": dice,
            "preds_map": preds,
            "img_plot_map": img_map,
            "masks_map": masks,
            "dataset": dataset
        })

    if not results:
        raise RuntimeError("No valid models loaded")

    visualize_multi_model_comparison(results, results[0]["dataset"])


if __name__ == "__main__":
    main()