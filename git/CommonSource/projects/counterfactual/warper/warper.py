import sys, os
import torch
from einops import rearrange
from einops import reduce
import itertools

os.chdir("/Users/burgert/CleanCode/Sandbox/youtube-ds/---qt1n3W14_301373607_319178010")
os.chdir("/Users/burgert/CleanCode/Sandbox/youtube-ds/-7Cxuw5aZAY_405555963_425701967")

video_tracks  = as_easydict(torch.load("video.mp4__DiffusionAsShaderCondition/video_tracks_spatracker.pt"                                      , map_location="cpu")).tracks
counter_tracks = as_easydict(torch.load("firstLastInterp_Jack2000.mp4__DiffusionAsShaderCondition/firstLastInterp_Jack2000_tracks_spatracker.pt", map_location="cpu")).tracks

video         = load_video("video_480p49.mp4"            , use_cache=True)
counter_video = load_video("firstLastInterp_Jack2000.mp4", use_cache=True)

tracks = video_tracks #Choose one for now...

#After counting the dots, I found the default spatialtracker results in a 70x70 grid.
TH = 70 #Tracks height
TW = 70 #Tracks width

T, N, VH, VW = validate_tensor_shapes(
    interp_tracks = "torch: T N XYZ",
    video_tracks  = "torch: T N XYZ",
    video         = "numpy: T VH VW RGB",
    N   = TH * TW,
    XYZ = 3,
    RGB = 3,
    return_dims = 'T N VH VW',
    verbose     = 'white white altbw green',
)

track_grids = rearrange(tracks, 'T (TH TW) XYZ -> T TH TW XYZ', TH=TH, TW=TW)

def subdivide_tracks(track_grids, new_TH, new_TW):
    #Takes a T TH TW XYZ grid, does subdivision along the X and Y axes.
    #Output shape is T new_TH new_TW XYZ

    upsampled = rearrange(track_grids, 'T TH TW XYZ -> T XYZ TH TW')
    upsampled = torch.nn.functional.interpolate(
        upsampled, 
        size=(new_TH, new_TW), 
        mode='bilinear', 
        align_corners=True
    )
    upsampled = rearrange(upsampled, 'T XYZ TH TW -> T TH TW XYZ')
    
    return upsampled

factor = 3
TH *= 3
TW *= 3
TH=VH
TW=VW
track_grids = subdivide_tracks(track_grids, TH, TW)

def preview_video():
    uv_image = get_identity_uv_map(height=TH, width=TW, uv_form="xy")
    uv_image = video[0]/255
    
    # Pre-compute coordinate indices
    y_indices, x_indices = torch.meshgrid(torch.arange(TH), torch.arange(TW), indexing='ij')
    coordinates = torch.stack([y_indices.flatten(), x_indices.flatten()], dim=1)
    
    for t in eta(range(T), "Drawing Video"):
        frame = as_rgb_image(uniform_float_color_image(VH, VW))
        
        # Extract z values for all coordinates at once
        z_values = track_grids[t, coordinates[:, 0], coordinates[:, 1], 2]
        
        # Sort coordinates by z-value
        sorted_indices = torch.argsort(z_values)
        sorted_coords = coordinates[sorted_indices]
        
        # Extract destination coordinates and colors in a vectorized way
        src_y, src_x = sorted_coords[:, 0], sorted_coords[:, 1]
        destinations = track_grids[t, src_y, src_x]
        dst_x = destinations[:, 0].int()
        dst_y = destinations[:, 1].int()
        
        # Get colors for all coordinates
        colors = uv_image[src_y, src_x]
        
        # Filter valid destination coordinates
        valid_mask = (dst_y >= 0) & (dst_y < VH) & (dst_x >= 0) & (dst_x < VW)
        valid_dst_y = dst_y[valid_mask]
        valid_dst_x = dst_x[valid_mask]
        valid_colors = colors[valid_mask]
        
        # Assign colors to frame
        frame[valid_dst_y, valid_dst_x] = valid_colors
        
        yield frame




display_video(
    vertically_concatenated_videos(
        list(preview_video()),
        video,
    ),
    loop=True,
)
