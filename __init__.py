from .bokeh_blur import BokehBlur
from .bokeh_blur_mask import BokehBlurMask

NODE_CLASS_MAPPINGS = {
    "CAM_BokehBlur": BokehBlur,
    "CAM_BokehBlurMask": BokehBlurMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CAM_BokehBlur": "CAM Bokeh Blur",
    "CAM_BokehBlurMask": "CAM Bokeh Blur Mask"
}
