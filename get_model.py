from lib import *
from lora import apply_lora

def get_model_mobilenetv3_large(num_classes, model_dropout=0.2, quantize=False, lora=False):
  model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

  # Modify the first convolutional layer to accept 1 channel input (grayscale images)
  model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

  # Modify dropout layer
  model.classifier[2] = nn.Dropout(p=model_dropout)

  # Modify the classifier for custom classification
  model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

  if quantize:
    model = torch.quantization.quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8
    )

  if lora:
    apply_lora(model)

  return model

def get_model_mobilenetv3_small(num_classes, model_dropout=0.2, quantize=False):
  model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

  # Modify the first convolutional layer to accept 1 channel input (grayscale images)
  model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

  # Modify dropout layer
  model.classifier[2] = nn.Dropout(p=model_dropout)

  # Modify the classifier for custom classification
  model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

  if quantize:
    model = torch.quantization.quantize_dynamic(
      model, {torch.nn.Linear}, dtype=torch.qint8
    )

  return model

def get_model_mobilenetv4(
    num_classes,
    # model_name='mobilenetv4_conv_large.e600_r384_in1k',
    model_name='mobilenetv4_conv_small.e1200_r224_in1k',
    model_dropout=0.2,
    grayscale=True,
    quantize=False,
    lora=False
  ):

  # Load pretrained MobileNetV4 from Hugging Face via timm
  model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

  model.classifier = torch.nn.Linear(
    in_features=model.classifier.in_features,
    out_features=model.classifier.out_features,
    bias=model.classifier.bias is not None,
  )

  # Modify input layer for grayscale input
  if grayscale:
    conv1 = model.conv_stem
    new_conv = nn.Conv2d(
      in_channels=1,
      out_channels=conv1.out_channels,
      kernel_size=conv1.kernel_size,
      stride=conv1.stride,
      padding=conv1.padding,
      bias=conv1.bias is not None
    )
    # Copy original weights by averaging across RGB channels
    with torch.no_grad():
      new_conv.weight[:] = conv1.weight.mean(dim=1, keepdim=True)
    model.conv_stem = new_conv

  if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
    for i, layer in enumerate(model.classifier):
      if isinstance(layer, nn.Dropout):
        model.classifier[i] = nn.Dropout(p=model_dropout)

  if lora:
    apply_lora(model)

  if quantize:
    model = torch.quantization.quantize_dynamic(
      model, {nn.Linear}, dtype=torch.qint8
    )

  return model