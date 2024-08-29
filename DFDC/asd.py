import timm

pretrained_model = timm.create_model("xception", pretrained=True)
a = pretrained_model.default_cfg
print(a)
