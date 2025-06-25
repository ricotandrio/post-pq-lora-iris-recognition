from lib import *

class LoRAConv2d(nn.Module):
  def __init__(self, original_conv, r=4, lora_alpha=2):
    super().__init__()
    self.original_conv = original_conv
    self.r = r
    self.lora_alpha = lora_alpha
    self.scaling = self.lora_alpha / self.r

    # Disable gradient for original weights
    self.original_conv.weight.requires_grad = False
    if original_conv.bias is not None:
      self.original_conv.bias.requires_grad = False

    # LoRA low-rank layers (1x1 convs)
    self.lora_A = nn.Conv2d(
      in_channels=original_conv.in_channels,
      out_channels=r,
      kernel_size=original_conv.kernel_size,
      stride=original_conv.stride,
      padding=original_conv.padding,
      dilation=original_conv.dilation,
      groups=1,
      bias=False
    )

    self.lora_B = nn.Conv2d(
      in_channels=r,
      out_channels=original_conv.out_channels,
      kernel_size=1,
      stride=1,
      padding=0,
      groups=1,
      bias=False
    )

    # Initialize LoRA layers to zero so original conv is unchanged initially
    nn.init.kaiming_uniform_(self.lora_A.weight, a=5)
    nn.init.zeros_(self.lora_B.weight)

    self.original_conv.weight.requires_grad = False
    if original_conv.bias is not None:
      self.original_conv.bias.requires_grad = False

  def forward(self, x):
    result = self.original_conv(x)

    lora_update = self.lora_B(self.lora_A(x)) * self.scaling

    return result + lora_update
  
def apply_lora(module, r=4, alpha=2):
  for name, child in module.named_children():
    if isinstance(child, nn.Conv2d):
      lora_layer = LoRAConv2d(child, r=r, lora_alpha=alpha)

      setattr(module, name, lora_layer)
      pass
    else:
      apply_lora(child)