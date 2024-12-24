import torch
import torch.nn.functional as F

class BokehBlurMask:
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
                "mask": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "bokeh_intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "transition_width": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "easing_type": (["linear", "smooth", "smoother", "ease-in", "ease-out", "ease-in-out"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "bokeh_blur"
    CATEGORY = "image/filters"

    def apply_easing(self, x, easing_type):
        if easing_type == "linear":
            return x
        elif easing_type == "smooth":
            return x * x * (3 - 2 * x)
        elif easing_type == "smoother":
            return x * x * x * (x * (x * 6 - 15) + 10)
        elif easing_type == "ease-in":
            return x * x
        elif easing_type == "ease-out":
            return 1 - (1 - x) * (1 - x)
        elif easing_type == "ease-in-out":
            return 0.5 * (torch.sin((x - 0.5) * torch.pi) + 1)
        return x

    def bokeh_blur(self, image, mask, strength, gamma, bokeh_intensity, threshold, transition_width, easing_type):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        # Store original image dimensions
        batch_mode = len(image.shape) == 4
        height = image.shape[1] if batch_mode else image.shape[0]
        width = image.shape[2] if batch_mode else image.shape[1]

        # Calculate radius based on image size and strength
        max_size = max(height, width)
        radius = int((max_size / 100.0) * strength)
        
        # Return original image if radius is 0
        if radius == 0:
            return (image,)

        # Store original for composition
        original = image.clone()

        # Normalize mask to [0, 1] and convert to grayscale if needed
        if len(mask.shape) > 2:
            if mask.shape[-1] > 1:  # If RGB/RGBA
                mask = 0.299 * mask[..., 0] + 0.587 * mask[..., 1] + 0.114 * mask[..., 2]
            else:
                mask = mask.squeeze(-1)

        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Resize mask to match image dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims for interpolate
        mask = F.interpolate(
            mask,
            size=(height, width),
            mode='bicubic',
            align_corners=False
        )
        mask = mask.squeeze()  # Back to 2D

        # Store original mask before gamma correction
        blend_mask = mask.clone()

        # Apply gamma correction only to the blur kernel, not the blending mask
        kernel_size = 2 * radius + 1
        kernel = F.interpolate(
            self.default_kernel.unsqueeze(0).unsqueeze(0),
            size=(kernel_size, kernel_size),
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        # Normalize kernel to [0,1] range first
        kernel = kernel / kernel.max()
        
        # Apply gamma and bokeh intensity only to the blur kernel
        kernel = torch.pow(kernel, gamma)
        kernel = kernel / kernel.max()  # Renormalize after gamma
        
        # Apply bokeh intensity with normalization
        kernel = torch.where(kernel > 0.5, 
                           torch.pow(kernel, 1/bokeh_intensity),  # Enhance bright areas
                           torch.pow(kernel, bokeh_intensity))    # Suppress dark areas
        
        # Final normalization to ensure proper convolution
        kernel = kernel / kernel.sum()

        # Process image for blur
        if len(image.shape) == 4:
            x = image.squeeze(0).permute(2, 0, 1).unsqueeze(0)
        else:
            x = image.permute(2, 0, 1).unsqueeze(0)
        
        # Apply blur
        padded = F.pad(x, (radius, radius, radius, radius), mode='reflect')
        blurred = F.conv2d(
            padded,
            kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1),
            groups=3
        )

        # Format blurred result for blending
        if len(image.shape) == 4:
            blurred = blurred.squeeze(0).permute(1, 2, 0).unsqueeze(0)
        else:
            blurred = blurred.squeeze(0).permute(1, 2, 0)

        # Use the unmodified blend_mask for composition
        blend_mask = blend_mask.unsqueeze(-1)
        if len(image.shape) == 4:
            blend_mask = blend_mask.unsqueeze(0)
        blend_mask = blend_mask.expand(*image.shape)

        # Debug prints
        print(f"Image shape: {image.shape}")
        print(f"Blend mask shape: {blend_mask.shape}")
        print(f"Blurred shape: {blurred.shape}")

        # Apply threshold with smooth transition
        blend_mask = mask.clone()
        
        # Calculate transition boundaries
        lower_bound = threshold - (transition_width / 2)
        upper_bound = threshold + (transition_width / 2)
        
        # Create transition mask
        transition_mask = torch.zeros_like(blend_mask)
        
        # Apply transition
        transition_region = (blend_mask >= lower_bound) & (blend_mask <= upper_bound)
        normalized_values = (blend_mask[transition_region] - lower_bound) / (upper_bound - lower_bound)
        transition_mask[transition_region] = self.apply_easing(normalized_values, easing_type)
        
        # Apply hard threshold outside transition region
        blend_mask = torch.where(blend_mask > upper_bound, 
                               torch.ones_like(blend_mask),
                               torch.where(blend_mask < lower_bound,
                                         torch.zeros_like(blend_mask),
                                         transition_mask))

        # Ensure proper dimensions for blending
        if len(image.shape) == 4:  # Batch mode
            blend_mask = blend_mask.view(1, height, width, 1).expand(1, height, width, 3)
        else:
            blend_mask = blend_mask.view(height, width, 1).expand(height, width, 3)

        # Debug prints
        print(f"Final shapes:")
        print(f"Original: {original.shape}")
        print(f"Blurred: {blurred.shape}")
        print(f"Blend mask: {blend_mask.shape}")

        # Compose final image
        result = original * blend_mask + blurred * (1.0 - blend_mask)
        result = torch.clamp(result, 0.0, 1.0)

        return (result,)
