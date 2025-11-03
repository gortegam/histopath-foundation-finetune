import timm
def make_model(arch: str, num_classes: int, pretrained: bool = True):
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
