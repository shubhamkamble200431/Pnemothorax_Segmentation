import segmentation_models_pytorch as smp


def build_unet():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )


def build_unetpp():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )


def build_manet():
    return smp.MAnet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )


MODEL_REGISTRY = {
    "unet": build_unet,
    "unetpp": build_unetpp,
    "manet": build_manet
}