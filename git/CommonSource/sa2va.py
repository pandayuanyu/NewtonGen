"""
Sa2VA: State-of-the-art vision-language model for image/video understanding (early 2025)

This module provides simple functions to work with ByteDance's Sa2VA-4B model, 
which handles both images and videos for:
- Image and video captioning
- Visual question answering 
- Referring segmentation (generates pixel masks for objects)

Functions automatically download the model on first use. No class initialization needed.

Input formats:
- Images: np.ndarray (HW3 uint8 or float 0-1), PIL Image, or path/URL
- Videos: List of frames, path, or URL
- Text: String prompts for questions or segmentation references

Example:
    # Caption an image
    caption = describe_image("photo.jpg")
    
    # Ask questions about an image
    answer = chat_image("photo.jpg", "What color is the car?")
    
    # Generate segmentation mask for a referred object
    text, masks = segment_image("photo.jpg", "Please segment the cat")

See: https://huggingface.co/ByteDance/Sa2VA-4B
"""
import rp
import torch
import numpy as np


__all__ = [
    "describe_image", 
    "describe_video",
    "chat_image", 
    "chat_video",
    "segment_image",
    "segment_video"
]


#If you want to set the device before initializing, you can set this variable
_sa2va_device = None


def _default_sa2va_device():
    """
    Get or initialize the default device for Sa2VA model.
    Uses rp.select_torch_device() to pick the best available device.
    This global device is updated whenever a model is loaded with 
    a specific device.
    
    Returns:
        The default torch device for Sa2VA
    """
    global _sa2va_device
    if _sa2va_device is None:
        _sa2va_device = rp.select_torch_device(reserve=True)
    return _sa2va_device


@rp.memoized
def _get_sa2va_model_helper(path, device):
    """
    Helper function to download and set up the Sa2VA model.
    Results are memoized for efficiency. Sets the global _sa2va_device.
    
    Args:
        path: HuggingFace model path
        device: Device to load the model on
        
    Returns:
        model
    """
    # These packages are needed
    rp.pip_import("timm")
    rp.pip_import("flash_attn")
    rp.pip_import("transformers")

    from transformers import AutoTokenizer, AutoModel

    # load the model
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()

    model._tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    model = model.to(device)

    return model

#This path will cause it to download from Huggingface
#But you can also specify a local path to the model
default_model_path = "ByteDance/Sa2VA-4B"

def _get_sa2va_model(path=None, device=None):
    """
    Get the Sa2VA model. Downloads from HuggingFace if not cached.
    The model will be loaded onto the specified device, or a default device 
    if none is specified. The device used becomes the new default device.

    If you need to download this manually, clone from here:
        https://huggingface.co/ByteDance/Sa2VA-4B
    
    Args:
        path: HuggingFace model path
        device: Device to load the model on
        
    Returns:
        model
    """

    path = path or default_model_path

    #The helper is cached with respect to device, so normalize that here
    if device is None:
        device = _default_sa2va_device()
    else:
        global _sa2va_device
        _sa2va_device = device
    device = torch.device(device)

    return _get_sa2va_model_helper(path, device)


def _load_video(video, num_frames=None):
    """
    Load and preprocess a video for Sa2VA model input.
    
    Args:
        video: A video as path, URL, or list of frames
        num_frames: Number of frames to sample from the video
        
    Returns:
        List of PIL.Image objects ready for the model
    """
    # Load the video if its a path or URL
    if isinstance(video, str):
        video = rp.load_video(video)

    # Evenly space the frames
    if num_frames is not None:
        video = rp.resize_list(video, num_frames)

    # Convert to PIL images
    video = rp.as_numpy_images(video)
    video = rp.as_rgb_images(video)
    video = rp.as_byte_images(video)
    video = rp.as_pil_images(video)

    return video


def _load_image(image):
    """
    Load and preprocess an image for Sa2VA model input.
    
    Args:
        image: An image as path, URL, np.ndarray, or PIL.Image
        
    Returns:
        PIL.Image object ready for the model
    """
    # Load the image if its a path or URL
    if isinstance(image, str):
        image = rp.load_image(image)

    # Convert to PIL image
    image = rp.as_numpy_image(image)
    image = rp.as_rgb_image(image)
    image = rp.as_byte_image(image)
    image = rp.as_pil_image(image)

    return image


def _run_sa2va(content, prompt, *, is_video=False, device=None, return_masks=False, num_frames=None, debug=False, model_path=None) -> str | tuple[str, list]:
    """
    Internal helper function that handles both image and video inputs.
    
    Args:
        content: Image or video input
        prompt: Text prompt for querying the model
        is_video: Whether the content is a video
        device: Device to run the model on
        return_masks: Whether to return segmentation masks
        num_frames: If is_video, only processes N frames of the video, evenly spaced from start to end
        debug: If True, returns the raw output of the model. Use for developing this module only.
        
    Returns:
        String response or tuple of (response, masks) if return_masks=True
    """
    model = _get_sa2va_model(path=model_path, device=device)
    
    # Load and process the content
    if is_video:
        content = _load_video(content, num_frames)
        content_key = "video"
    else:
        content = _load_image(content)
        content_key = "image"
    
    # Prepare the prompt
    text_prompts = "<image>" + prompt
    
    # Build input dictionary
    input_dict = {
        content_key: content,
        "text": text_prompts,
        "past_text": "",
        "mask_prompts": None,
        "tokenizer": model._tokenizer,
    }
    
    # Run the model
    with model.device, torch.no_grad():
        return_dict = model.predict_forward(**input_dict)

    if debug:
        return return_dict
    
    # Process the output
    predicted_text = return_dict["prediction"]
    predicted_text = predicted_text.rstrip("<|end|>")
    
    if return_masks and "prediction_masks" in return_dict:
        masks = return_dict["prediction_masks"][0]
        return masks
    
    return predicted_text


def chat_image(image, prompt, device=None, model_path=None) -> str:
    """
    Given an image and a text prompt, return text.
    Can answer questions about the image, etc.
    
    Args:
        image: np.ndarray, PIL Image, or path/URL
        prompt: Text prompt for querying the model
        device: Optional device to run inference on. If any Sa2VA model has been 
                initialized previously, the most recent device becomes the default.
        
    Returns:
        Text response from the model
    """
    return _run_sa2va(image, prompt, is_video=False, device=device, model_path=model_path)


def chat_video(video, prompt, device=None, *, num_frames=5, model_path=None) -> str:
    """
    Given a video and a text prompt, return text.
    Can caption videos or answer questions about it, etc.
    
    Args:
        video: List of frames, path, or URL
        prompt: Text prompt for querying the model
        device: Optional device to run inference on. If any Sa2VA model has been 
                initialized previously, the most recent device becomes the default.
        num_frames: Number of frames to process, evenly spaced from start to end of the video
        
    Returns:
        Text response from the model
    """
    return _run_sa2va(video, prompt, is_video=True, device=device, num_frames=num_frames, model_path=model_path)


def describe_image(image, device=None, model_path=None) -> str:
    """
    Image captioning: generates a description of the given image.
    
    Args:
        image: np.ndarray, PIL Image, or path/URL
        device: Optional device to run inference on. If any Sa2VA model has been 
                initialized previously, the most recent device becomes the default.
        model_path: Optional model path to use a specific Sa2VA model
        
    Returns:
        Text description of the image
    """
    return _run_sa2va(image, "Generate a detailed description of the image.", is_video=False, device=device, model_path=model_path)


def describe_video(video, device=None, *, num_frames=5, model_path=None) -> str:
    """
    Video captioning: generates a description of the given video.
    
    Args:
        video: List of frames, path, or URL 
        device: Optional device to run inference on. If any Sa2VA model has been 
                initialized previously, the most recent device becomes the default.
        num_frames: Number of frames to process, evenly spaced from start to end of the video
        model_path: Optional model path to use a specific Sa2VA model
        
    Returns:
        Text description of the video
    """
    return _run_sa2va(video, "Generate a detailed description of the video.", is_video=True, device=device, num_frames=num_frames, model_path=model_path)


def segment_image(image, prompt, device=None, model_path=None) -> np.ndarray:
    """
    Performs referring segmentation on an image based on the text prompt.
    
    Args:
        image: np.ndarray, PIL Image, or path/URL
        prompt: Text prompt describing what to segment (e.g., "Please segment the person")
        device: Optional device to run inference on. If any Sa2VA model has been 
                initialized previously, the most recent device becomes the default.
        model_path: Optional model path to use a specific Sa2VA model
        
    Returns:
        segmentation_mask: HW bool np.ndarray: A binary mask matrix
    """
    return _run_sa2va(image, prompt, is_video=False, device=device, return_masks=True, model_path=model_path)


def segment_video(video, prompt, device=None, *, num_frames=None, model_path=None) -> np.ndarray:
    """
    Performs referring segmentation on a video based on the text prompt.

    Args:
        video: List of frames, path, or URL
        prompt: Text prompt describing what to segment (e.g., "Please segment the person")
        device: Optional device to run inference on. If any Sa2VA model has been 
                initialized previously, the most recent device becomes the default.
        num_frames: Number of frames to process, evenly spaced from start to end of the video.
                   If None, uses all frames in the video.
        model_path: Optional model path to use a specific Sa2VA model
        
    Returns:
        segmentation_masks: THW bool np.ndarray: Binary masks for each frame
    """
    return _run_sa2va(video, prompt, is_video=True, device=device, return_masks=True, num_frames=num_frames, model_path=model_path)

def demo():
    """ 
    Run this demo inside a Jupyter notebook!
    It shows this model's capabilities
    """
    import rp

    #Download and update this RP extension
    rp.git_import('CommonSource',pull=True)
    import rp.git.CommonSource.sa2va as sa2va

    #Get our input video
    video_url='https://videos.pexels.com/video-files/6507082/6507082-hd_1920_1080_25fps.mp4'
    video=rp.load_video(video_url,use_cache=True)
    video=rp.resize_video_to_fit(video, height=512, width=512, allow_growth=False) #Resize in both diections until it fits in a 512x512 box
    video=rp.resize_list_to_fit(video, 120) #Keep up to this many frames, spaced evenly throughout the video

    #Caption the video
    description = sa2va.describe_video(video)
    print("VIDEO DESCRIPTION:", description)

    #Ask a question about the video
    prompt = 'How many people are in this video?'
    chat_output = sa2va.chat_video(video, prompt)
    print("CHAT OUTPUT:",chat_output)

    #Segment the video
    man_masks   = sa2va.segment_video(video, 'Please segment the man'  )
    woman_masks = sa2va.segment_video(video, 'Please segment the woman')
    preview_video = rp.labeled_images(
        rp.tiled_videos(
            rp.labeled_videos(
                [video, man_masks[..., None] * video, woman_masks[..., None] * video],
                ["Input Video", "Man Masks", "Woman Masks"],
                font="R:Futura",
                show_progress=True,
            ),
            length=3,
            show_progress=True,
        ),
        "SA2VA Demo",
        size=30,
        font="R:Futura",
        text_color="yellow",
        show_progress=True,
    )
    rp.display_video(preview_video, framerate=30)

