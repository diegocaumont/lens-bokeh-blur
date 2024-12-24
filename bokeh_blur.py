import torch
import torch.nn.functional as F

class BokehBlur:
    """Bokeh blur effect filter for images.
    
    This filter creates a bokeh-style blur effect by applying a circular kernel
    with customizable strength and gamma parameters. The kernel is designed to
    simulate the out-of-focus blur characteristic of camera lenses.
    
    Copyright (c) CAMCONNECTING SARL
    All rights reserved.
    
    Author: CAMCONNECTING SARL
    Version: 1.0
    """
    
    def __init__(self):
        size = 31
        center = size // 2
        kernel = torch.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                if dist <= center:
                    kernel[i, j] = 1.0 - (dist / center) ** 2
        self.default_kernel = kernel

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "bokeh_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "bokeh_blur"
    CATEGORY = "image/filters"

    def bokeh_blur(self, image, strength, gamma, bokeh_intensity):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        print(f"Input image shape: {image.shape}")

        max_size = max(image.shape[-2:])
        radius = int((max_size / 100.0) * strength)
        if radius == 0:
            return (image,)

        kernel_size = 2 * radius + 1
        kernel = F.interpolate(
            self.default_kernel.unsqueeze(0).unsqueeze(0),
            size=(kernel_size, kernel_size),
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        # Normalize kernel to [0,1] range first
        kernel = kernel / kernel.max()
        
        # Apply gamma with normalization
        kernel = torch.pow(kernel, gamma)
        kernel = kernel / kernel.max()  # Renormalize after gamma
        
        # Apply bokeh intensity with normalization
        kernel = torch.where(kernel > 0.5, 
                           torch.pow(kernel, 1/bokeh_intensity),  # Enhance bright areas
                           torch.pow(kernel, bokeh_intensity))    # Suppress dark areas
        
        # Final normalization to ensure proper convolution
        kernel = kernel / kernel.sum()
        
        if len(image.shape) == 4:
            x = image.squeeze(0).permute(2, 0, 1).unsqueeze(0)
        else:
            x = image.permute(2, 0, 1).unsqueeze(0)
        
        padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
        output = F.conv2d(
            padded,
            kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1),
            groups=3
        )
        
        if len(image.shape) == 4:
            return (output.squeeze(0).permute(1, 2, 0).unsqueeze(0),)
        else:
            return (output.squeeze(0).permute(1, 2, 0),)
