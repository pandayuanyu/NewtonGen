"""
Motion tracking visualization module for creating control videos.

This module provides functionality for tracking differences in motion between videos
by using CoTracker to follow randomly sampled points across video frames and generating
visual representations of their paths.

The main function is `generate_dotted_latents` which creates a control video with
tracked dots that visualize motion patterns.
"""

__all__ = ['generate_dotted_latents', 'demo_dotted_latents', 'default_visibility_mode']

import rp
import torch
import numpy as np

# Global default for visibility mode
# Options:
# - 'hide-invisible': Only show dots that are visible
# - 'show-invisible': Show all dots regardless of visibility
default_visibility_mode = 'hide-invisible'


def _fast_scatter_add(
    output_tensor,
    latent_tracks,
    track_colors,
    visibility,
    num_timesteps,
    num_points,
    width,
    height,
    visibility_mode=None,
):
    """
    Efficiently adds tracking point colors to a latent tensor using scatter_add.
    
    Analogy: 
    The original implementation is like manually placing colored dots on a canvas one at a time:
    ```
    for lt in range(LT):
        for n in range(N):
            color = track_colors[n]
            x, y = latent_tracks[lt, n].long()
            if 0<=x<LW and 0<=y<LH:
                # Check visibility if needed
                if visibility_mode == 'show-invisible' or visibility[lt, n] > 0:
                    dotted_latent[lt, :, y, x] += color
    ```
    
    This optimized version is like having a machine that can place all dots of the same color
    simultaneously at their respective coordinates, with the ability to filter out invisible dots.
    
    Args:
        output_tensor: Zero-initialized tensor to populate with values (LT, LC, LH, LW)
        latent_tracks: Tensor containing track coordinates (LT, N, 2)
        track_colors: Tensor of colors for each track point (N, LC)
        visibility: Tensor indicating if each point is visible (LT, N)
        num_timesteps: Number of timesteps (LT)
        num_points: Number of track points (N)
        width: Width of latent tensor (LW)
        height: Height of latent tensor (LH)
        visibility_mode: How to handle point visibility:
                        'hide-invisible': Only show dots that are visible
                        'show-invisible': Show all dots regardless of visibility
                        If None, uses the module's default_visibility_mode (default: 'hide-invisible')
        
    Returns:
        torch.Tensor: Populated tensor with added track colors
    """
    C = track_colors.shape[1]  # Number of channels
    
    # Extract x and y coordinates
    xs, ys = latent_tracks[..., 0], latent_tracks[..., 1]
    
    # Create base mask for valid coordinates (within bounds)
    valid_mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    
    # Use global default if visibility_mode is None
    if visibility_mode is None:
        visibility_mode = default_visibility_mode
    
    # Apply visibility check if mode is 'hide-invisible'
    if visibility_mode == 'hide-invisible':
        valid_mask = valid_mask & (visibility > 0)
    
    # Loop over timesteps (still needed, but with vectorized inner operations)
    for t in range(num_timesteps):
        # Get valid points for this timestep
        t_valid = valid_mask[t]
        if not t_valid.any():
            continue
            
        # Get coordinates of valid points
        valid_xs = xs[t, t_valid]
        valid_ys = ys[t, t_valid]
        valid_indices = t_valid.nonzero().squeeze(1)
        
        # Get colors for valid points
        valid_colors = track_colors[valid_indices]
        
        # Compute indices for scatter operation
        indices = valid_ys * width + valid_xs
        
        # Perform scatter_add for each channel
        for c in range(C):
            # Create flat output tensor for this timestep and channel
            flat_output = output_tensor[t, c].view(-1)
            
            # Scatter add the values
            flat_output.scatter_add_(0, indices, valid_colors[:, c])
    
    return output_tensor

def _get_all_mask_positions(mask):
    """
    Get (t, x, y) coordinates of all True pixels in the mask.
    
    Args:
        mask: A 3D boolean mask of shape (T, H, W) where True indicates positions to extract.
              Can be either torch.Tensor or numpy.ndarray.
    
    Returns:
        torch.Tensor: Tensor of shape (M, 3) containing (t, x, y) coordinates of all True pixels,
                      where M is the number of True pixels in the mask
    """
    assert mask.ndim == 3
    assert mask.dtype == bool or mask.dtype == torch.bool
    
    if isinstance(mask, torch.Tensor):
        # Already a torch tensor, use torch.where
        indices = torch.where(mask)
        ts, ys, xs = indices
        # Stack to get M×3 tensor
        return torch.stack([ts, xs, ys], dim=1)
    else:
        # Convert numpy array to torch tensor
        mask_tensor = torch.from_numpy(mask)
        indices = torch.where(mask_tensor)
        ts, ys, xs = indices
        #NOTE: torch.where (like np.where) returns ys then xs, we return xs then ys!
        return torch.stack([ts, xs, ys], dim=1)

def _get_random_mask_positions(mask, N):
    """
    Randomly samples N positions from the True pixels in a 3D mask.
    
    Args:
        mask: A 3D boolean mask of shape (T, H, W) where True indicates valid positions.
              Can be either torch.Tensor or numpy.ndarray.
        N: Number of random positions to sample with replacement
    
    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing randomly sampled (t, x, y) coordinates
    """
    assert mask.ndim == 3
    assert mask.dtype == bool or mask.dtype == torch.bool
    T, H, W = mask.shape
    
    # Get all positions as a torch tensor
    all_positions = _get_all_mask_positions(mask)  # M×3 torch tensor
    
    if len(all_positions) == 0:
        raise ValueError("Mask contains no True values to sample from")
    
    # Determine device based on the tensor
    device = all_positions.device
    
    # Randomly sample N indices with replacement
    idx = torch.randint(0, len(all_positions), (N,), device=device)
    chosen_positions = all_positions[idx]
    
    assert chosen_positions.shape == (N, 3)
    return chosen_positions  # N×3 torch tensor

def generate_dotted_latents(
    videos,
    latent_mask,
    *,
    out_channels = 16, 
    num_points=1024,
    device = None,
    silent = True,
    visibility_mode = None,
):
    """
    Generates a latent-tracking-point control video from a source video
    Uses CoTracker to move random dots around a latent video

    Args:
        - videos: Give a list of videos, either in torch BTCHW form or a list of rp videos
        - latent_mask: Determines the duration and dimensions of the output, 
                       and where tracking points are initialized.
                       Either numpy arrays or torch tensors are ok.
                       In THW form. 
        - out_channels: Determines the number of channels in the output
        - num_points: The number of tracking points we use
        - device: Optional, if specified we use that torch device.
        - silent: If False, will print debug info.
        - visibility_mode: How to handle point visibility, options are:
                          'hide-invisible': Only show dots that are visible
                          'show-invisible': Show all dots regardless of visibility
                          If None, uses the module's default_visibility_mode (default: 'hide-invisible')

    Returns:
        - torch.Tensor: BTCHW form (where B comes from videos, THW come from latent_mask, and C comes from out_channels)
    """
    dtype = torch.bfloat16
    device = device or rp.select_torch_device(prefer_used=True, silent=silent, reserve=True)

    LT, LH, LW = latent_mask.shape
    LC = out_channels
    N = num_points
        
    videos = rp.as_torch_videos(videos, device=device, dtype=dtype)
    B, VT, _, VH, VW = videos.shape

    spatial_scale = rp.assert_equality(VH/LH, VW/LW)

    #latent_mask -> mask has same dimensions and duration as video, and is bool
    mask = latent_mask
    mask = rp.resize_list(mask, VT)
    mask = rp.as_numpy_array(mask)
    mask = rp.as_grayscale_images(mask)
    mask = rp.as_binary_images(mask)
    mask = rp.resize_images(mask, size=(VH, VW), interp='nearest')
    mask = rp.as_binary_images(mask)
    assert rp.is_numpy_array(mask)

    # Sample N random spatio-temporal points from the mask
    track_points = _get_random_mask_positions(mask, N)
    # Convert to numpy array since run_cotracker expects numpy
    track_points = track_points.cpu().numpy() if isinstance(track_points, torch.Tensor) else track_points
    track_colors = torch.randn(N, LC, device=device, dtype=dtype)

    dotted_latents = []
    for video in videos:
        #TODO: Can we get away with simply tracking less frames instead of tracking all frames and squashing time afterwards?
        tracks, visibility = rp.run_cotracker(
            video, 
            device=device,
            queries=track_points,
        )

        latent_tracks = rp.resize_list(tracks, LT) // spatial_scale #Scale it down spatiotemporally
        latent_tracks = latent_tracks.long()
        
        dotted_latent = torch.zeros(LT, LC, LH, LW, device=device, dtype=dtype)
        
        # Resize visibility to match latent_tracks timesteps
        latent_visibility = rp.resize_list(visibility, LT)
        
        # Use optimized scatter_add helper function with visibility mode
        dotted_latent = _fast_scatter_add(
            dotted_latent,
            latent_tracks,
            track_colors,
            latent_visibility,
            LT,
            N,
            LW,
            LH,
            visibility_mode,
        )
        
        dotted_latents.append(dotted_latent)
    dotted_latents = torch.stack(dotted_latents)
    
    rp.validate_tensor_shapes(
        videos            = 'torch: B VT   RGB VH VW',
        dotted_latents    = 'torch: B LT   LC  LH LW',
        dotted_latent     = 'torch:   LT   LC  LH LW',
        video             = 'torch:   VT   RGB VH VW',
        mask              = 'numpy:   VT       VH VW',
        tracks            = 'torch:   VT N XY       ',
        visibility        = 'torch:   VT N          ',
        latent_visibility = 'torch:   LT N          ',
        track_points      = 'numpy:      N TXY      ',
        latent_tracks     = 'torch:   LT N XY       ',
        latent_mask       = '         LT       LH LW',
        track_colors      = 'torch:      N LC       ',
        XY  = 2,
        TXY = 3,
        RGB = 3,
        **rp.gather_vars('N LC LT LH LW B VT VH VW'), #Already decided
        verbose = not silent and 'bold white random green',
    )

    #CogVideoX Assumptions
    assert spatial_scale==8          , "Assuming we're using CogVideoX"
    assert (LT, LH, LW)==(13, 60, 90), "Assuming we're using CogVideoX"
    assert (VT, VH, VW)==(49,480,720), "Assuming we're using CogVideoX"

    return rp.gather_vars('dotted_latents')

def demo_dotted_latents(*video_urls, visibility_mode=None):
    """
    Demonstrates the dotted latents functionality.
    
    Args:
        video_urls: URLs of videos to use for demonstration
        visibility_mode: How to handle point visibility:
                        'hide-invisible': Only show dots that are visible
                        'show-invisible': Show all dots regardless of visibility
                        If None, uses the module's default_visibility_mode (default: 'hide-invisible')
    """
    video_urls = rp.detuple(video_urls) or ["https://video-previews.elements.envatousercontent.com/23ce1f71-c55d-4bc3-bfad-bc7bf8d8168a/watermarked_preview/watermarked_preview.mp4"]
    videos = rp.load_videos(video_urls, use_cache=True)
    videos = rp.resize_videos(videos,size=(480,720))
    videos = rp.resize_lists(videos, length=49)

    rp.tic()
    result = generate_dotted_latents(
        videos,
        np.ones((13, 60, 90)),
        out_channels=3,
        silent=False,
        visibility_mode=visibility_mode,
    )
    [dotted_latent_video] = result.dotted_latents
    rp.ptoc("generate_dotted_latents")
    
    rp.display_video(
        rp.horizontally_concatenated_videos(
            *rp.resize_lists(videos, 13),
            rp.as_numpy_images(
                rp.torch_resize_images(
                    dotted_latent_video, size=8, interp="nearest", copy=True
                )
                / 5
                + 0.5
            ),
        ),
        framerate=10,
    )
