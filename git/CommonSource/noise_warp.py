import glob
import sys

import fire
import numpy as np
import rp
import torch
from einops import rearrange
from tqdm import tqdm

sys.path.append(rp.get_path_parent(__file__))
import raft
from background_remover import BackgroundRemover

def unique_pixels(image):
    """
    Find unique pixel values in an image tensor and return their RGB values, counts, and inverse indices.

    Args:
        image (torch.Tensor): Image tensor of shape [c, h, w], where c is the number of channels (e.g., 3 for RGB),
                              h is the height, and w is the width of the image.

    Returns:
        tuple: A tuple containing three tensors:
            - unique_colors (torch.Tensor): Tensor of shape [u, c] representing the unique RGB values found in the image,
                                            where u is the number of unique colors.
            - counts (torch.Tensor): Tensor of shape [u] representing the counts of each unique color.
            - index_matrix (torch.Tensor): Tensor of shape [h, w] representing the inverse indices of each pixel,
                                           mapping each pixel to its corresponding unique color index.
    """
    c, h, w = image.shape

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Find unique RGB values, counts, and inverse indices
    unique_colors, inverse_indices, counts = torch.unique(flattened_pixels, dim=0, return_inverse=True, return_counts=True, sorted=False)
    # unique_colors, inverse_indices, counts = torch.unique_consecutive(flattened_pixels, dim=0, return_inverse=True, return_counts=True)

    # Get the number of unique indices
    u = unique_colors.shape[0]

    # Reshape the inverse indices back to the original image dimensions [h, w] using einops
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)

    # Assert the shapes of the output tensors
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    assert index_matrix.min() == 0
    assert index_matrix.max() == u - 1

    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    """
    Sum the values in the CHW image tensor based on the indices specified in the HW index matrix.

    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W], where C is the number of channels,
                              H is the height, and W is the width of the image.
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique value.
                                     Indices range [0, U), where U is the number of unique indices

    Returns:
        torch.Tensor: Tensor of shape [U, C] representing the sum of values in the image tensor
                      based on the indices in the index matrix, where U is the number of unique
                      indices in the index matrix.
    """
    c, h, w = image.shape
    u = index_matrix.max() + 1

    # Rearrange the image tensor from [c, h, w] to [h, w, c] using einops
    pixels = rearrange(image, "c h w -> h w c")

    # Flatten the image tensor to [h*w, c]
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")

    # Create an output tensor of shape [u, c] initialized with zeros
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)

    # Scatter sum the flattened pixel values using the index matrix
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)

    # Assert the shapes of the input and output tensors
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"
    assert index_matrix.shape == (h, w), f"Expected index_matrix shape: ({h}, {w}), but got: {index_matrix.shape}"
    assert output.shape == (u, c), f"Expected output shape: ({u}, {c}), but got: {output.shape}"

    return output

def indexed_to_image(index_matrix, unique_colors):
    """
    Create a CHW image tensor from an HW index matrix and a UC unique_colors matrix.

    Args:
        index_matrix (torch.Tensor): Index matrix tensor of shape [H, W] containing indices
                                     specifying the mapping of each pixel to its corresponding
                                     unique color.
        unique_colors (torch.Tensor): Unique colors matrix tensor of shape [U, C] containing
                                      the unique color values, where U is the number of unique
                                      colors and C is the number of channels.

    Returns:
        torch.Tensor: Image tensor of shape [C, H, W] representing the reconstructed image
                      based on the index matrix and unique colors matrix.
    """
    h, w = index_matrix.shape
    u, c = unique_colors.shape

    # Assert the shapes of the input tensors
    assert index_matrix.max() < u, f"Index matrix contains indices ({index_matrix.max()}) greater than the number of unique colors ({u})"

    # Gather the colors based on the index matrix
    flattened_image = unique_colors[index_matrix.view(-1)]

    # Reshape the flattened image to [h, w, c]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)

    # Rearrange the image tensor from [h, w, c] to [c, h, w] using einops
    image = rearrange(image, "h w c -> c h w")

    # Assert the shape of the output tensor
    assert image.shape == (c, h, w), f"Expected image shape: ({c}, {h}, {w}), but got: {image.shape}"

    return image


def demo_pixellation_via_proxy():
    real_image = rp.as_torch_image(
        rp.cv_resize_image(
            rp.load_image("https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg"),
            (512, 512),
        )
    )

    c, h, w = real_image.shape

    noise_image = torch.randn(c, h // 4, w // 4)

    # Resize noise_image using nearest-neighbor interpolation to match the dimensions of real_image
    pixelated_noise_image = rp.torch_resize_image(noise_image, 4, "nearest")
    assert pixelated_noise_image.shape==(c,h,w)

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(pixelated_noise_image)

    # Sum the color values from real_image based on the indices of the unique noise pixels
    summed_colors = sum_indexed_values(real_image, index_matrix)

    # Divide the summed color values by the counts to get the average color for each unique pixel
    average_colors = summed_colors / rearrange(counts, "u -> u 1")

    # Create a new pixelated image using the average colors and the index matrix
    pixelated_dog_image = indexed_to_image(index_matrix, average_colors)

    rp.display_image(pixelated_dog_image)
    
def calculate_wave_pattern(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the wave pattern based on the distance and angle
    wave_freq = 0.05  # Frequency of the waves
    wave_amp = 10.0   # Amplitude of the waves
    wave_offset = frame * 0.05  # Offset for animation
    
    dx = wave_amp * torch.cos(dist_from_center * wave_freq + angle_from_center + wave_offset)
    dy = wave_amp * torch.sin(dist_from_center * wave_freq + angle_from_center + wave_offset)
    
    return dx, dy

def starfield_zoom(h, w, frame):
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    
    # Calculate the distance from the center of the image
    center_x, center_y = w // 2, h // 2
    dist_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Calculate the angle from the center of the image
    angle_from_center = torch.atan2(y - center_y, x - center_x)
    
    # Calculate the starfield zoom effect
    zoom_speed = 0.01  # Speed of the zoom effect
    zoom_scale = 1.0 + frame * zoom_speed  # Scale factor for the zoom effect
    
    # Calculate the displacement based on the distance and angle
    dx = dist_from_center * torch.cos(angle_from_center) / zoom_scale
    dy = dist_from_center * torch.sin(angle_from_center) / zoom_scale
    
    return dx, dy

_arange_cache={}
def _cached_arange(length, device, dtype):
    code=hash((length,device,dtype))
    if code in _arange_cache:
        return _arange_cache[code]

    
    _arange_cache[code]= torch.arange(length , device=device, dtype=dtype)
    return _arange_cache[code]

def fast_nearest_torch_remap_image(image, x, y, *, relative=False, add_alpha_mask=False, use_cached_meshgrid=False):
    # assert rp.r.is_torch_image(image), "image must be a torch tensor with shape [C, H, W]"
    # assert is_torch_tensor(x) and is_a_matrix(x), "x must be a torch tensor with shape [H_out, W_out]"
    # assert is_torch_tensor(y) and is_a_matrix(y), "y must be a torch tensor with shape [H_out, W_out]"
    # assert x.shape == y.shape, "x and y must have the same shape, but got x.shape={} and y.shape={}".format(x.shape, y.shape)
    # assert image.device==x.device==y.device, "all inputs must be on the same device"

    # pip_import('torch')

    import torch

    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    if add_alpha_mask:
        alpha_mask = torch.ones_like(image[:1])
        image = torch.cat([image, alpha_mask], dim=0)

    if torch.is_floating_point(x): x = x.round_().long()
    if torch.is_floating_point(y): y = y.round_().long()

    if relative:
        # assert in_height == out_height, "For relative warping, input and output heights must match, but got in_height={} and out_height={}".format(in_height, out_height)
        # assert in_width  == out_width , "For relative warping, input and output widths must match, but got in_width={} and out_width={}".format(in_width, out_width)
        x += _cached_arange(in_width , device=x.device, dtype=x.dtype)
        y += _cached_arange(in_height, device=y.device, dtype=y.dtype)[:,None]

    x.clamp_(0, in_width - 1)
    y.clamp_(0,in_height-1)
    out = image[:, y, x]

    expected_c = in_c+1 if add_alpha_mask else in_c
    assert out.shape == (expected_c, out_height, out_width), "Expected output shape: ({}, {}, {}), but got: {}".format(expected_c, out_height, out_width, out.shape)

    return out


def warp_noise(noise, dx, dy, s=1):
    #This is *certainly* imperfect. We need to have particle swarm in addition to this.

    dx=dx.round_().int()
    dy=dy.round_().int()

    c, h, w = noise.shape
    assert dx.shape==(h,w)
    assert dy.shape==(h,w)

    #s is scaling factor
    hs = h * s
    ws = w * s
    
    #Upscale the warping with linear interpolation. Also scale it appropriately.
    if s!=1:
        up_dx = rp.torch_resize_image(dx[None], (hs, ws), interp="bilinear")[0]
        up_dy = rp.torch_resize_image(dy[None], (hs, ws), interp="bilinear")[0]
        up_dx *= s
        up_dy *= s

        up_noise = rp.torch_resize_image(noise, (hs, ws), interp="nearest")
    else:
        up_dx = dx
        up_dy = dy
        up_noise = noise
    assert up_noise.shape == (c, hs, ws)

    # Warp the noise - and put 0 where it lands out-of-bounds
    # up_noise = rp.torch_remap_image(up_noise, up_dx, up_dy, relative=True, interp="nearest")
    up_noise = fast_nearest_torch_remap_image(up_noise, up_dx, up_dy, relative=True)
    assert up_noise.shape == (c, hs, ws)
    
    # Regaussianize the noise
    output, _ = regaussianize(up_noise)

    #Now we resample the noise back down again
    if s!=1:
        output = rp.torch_resize_image(output, (h, w), interp='area')
        output = output * s #Adjust variance by multiplying by sqrt of area, aka sqrt(s*s)=s

    return output


def regaussianize(noise):
    c, hs, ws = noise.shape

    # Find unique pixel values, their indices, and counts in the pixelated noise image
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)
    assert unique_colors.shape == (u, 1)
    assert counts.shape == (u,)
    assert index_matrix.max() == u - 1
    assert index_matrix.min() == 0
    assert index_matrix.shape == (hs, ws)

    foreign_noise = torch.randn_like(noise)
    assert foreign_noise.shape == noise.shape == (c, hs, ws)

    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    assert summed_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    assert meaned_foreign_noise_colors.shape == (u, c)

    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    assert meaned_foreign_noise.shape == (c, hs, ws)

    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    assert zeroed_foreign_noise.shape == (c, hs, ws)

    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    assert counts_image.shape == (1, hs, ws)

    #To upsample noise, we must first divide by the area then add zero-sum-noise
    output = noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise

    assert output.shape == noise.shape == (c, hs, ws)

    return output, counts_image
    
def demo_noise_warp(Q=-6,scale_factor=1,num_frames=300):
    #Run this in a Jupyter notebook and watch the noise go brrrrrrr

    
    d=rp.JupyterDisplayChannel()
    d.display()
    device='cuda'
    h=w=128

    # rp.cv_imshow(rp.apply_colormap_to_image(output[ω]/output[ω].mean()/4),label='weight')

    warper = NoiseWarper(3,h,w,device=device,scale_factor=scale_factor,)

    #Add some ink
    warper._state[4:, int(h/256*10 * scale_factor) : int(h/256*80 * scale_factor), int(h/256*10 * scale_factor) : int(h/256*80 * scale_factor)] = 1  # Make a little black square
    warper._state[
        4:,
        h * scale_factor // 2 - 5 * scale_factor : h * scale_factor // 2 + 5 * scale_factor,
        w * scale_factor // 2 - 5 * scale_factor : w * scale_factor // 2 + 5 * scale_factor,
    ] = 10  # Make a little black square
    
    noise=torch.randn(3,h,w).to(device)
    wdx,wdy=calculate_wave_pattern(h,w,frame=0)
    sdx,sdy=starfield_zoom(h,w,frame=1)

    dx=sdx+2*wdx
    dy=sdy+2*wdy
    
    dx/=dx.max()
    dy/=dy.max()
    # Q=-6
    dy*=Q
    dx*=Q
    dx=dx.to(device)
    dy=dy.to(device)
    new_noise=noise

    frames=[]
    try:
        for _ in range(num_frames):
            new_noise=warper(dx,dy).noise
            weights = warper._state[2]
            weights = rp.torch_resize_image(weights[None],(h,w))[0]
            
            # rp.display_image(new_noise)
            frame=rp.tiled_images(
                    [
                        rp.as_numpy_image(new_noise/4+.5),
                        rp.apply_colormap_to_image(weights/weights.mean()/4),
                    ],
                    border_thickness=0,
                )
            frames.append(rp.labeled_image(frame,'Frame %i'%_))
            d.update(
                frame
            )
    except KeyboardInterrupt:
        print("Interrupted demo at frame",_)
    return frames

def demo_webcam_noise_warp():
    import cv2
    from rp import (
        as_numpy_image,
        cv_bgr_rgb_swap,
        display_image,
        get_image_dimensions,
        rp,
        tiled_images,
    )

    def resize_frame(frame, target_height=64):
        aspect_ratio = frame.shape[1] / frame.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized_frame = cv2.resize(frame, (target_width, target_height))
        # print(resized_frame.shape)
        return resized_frame

    def main():
        cap = cv2.VideoCapture(0)
        ret, prev_frame = cap.read()
        prev_frame = resize_frame(prev_frame)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Initialize DeepFlow Optical Flow
        optical_flow = cv2.optflow.createOptFlow_DeepFlow()

        d = rp.JupyterDisplayChannel()
        d.display()
        device = "cpu"
        h, w = get_image_dimensions(prev_frame)
        wdx, wdy = calculate_wave_pattern(h, w, frame=0)
        sdx, sdy = starfield_zoom(h, w, frame=1)

        dx = sdx + 2 * wdx
        dy = sdy + 2 * wdy

        dx /= dx.max()
        dy /= dy.max()
        Q = -6
        dy *= Q
        dx *= Q
        dx = dx.to(device)
        dy = dy.to(device)
        # new_xyωc = noise_to_xyωc(noise)

        warper=NoiseWarper(3,h,w,device=device,scale_factor=2)


        while True:
            ret, frame = cap.read()
            frame = resize_frame(frame)
            frame=rp.horizontally_flipped_image(frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            # Compute the optical flow
            flow = optical_flow.calc(prev_gray, frame_gray, None)

            x = flow[:, :, 0]
            y = flow[:, :, 1]

            dx = torch.Tensor(x)
            dy = torch.Tensor(y)

            prev_gray = frame_gray.copy()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            new_noise = warper(dx, dy).noise
            weights = warper._state[2]
            weights = rp.torch_resize_image(weights[None],(h,w))[0]

            display_image(
                tiled_images(
                    [
                        as_numpy_image(new_noise / 5 + 0.5),
                        cv_bgr_rgb_swap(frame),
                        rp.apply_colormap_to_image(weights / weights.mean() / 4),
                    ]
                )
            )

            # d.update(rp.as_numpy_image(new_noise/4+.5))

        cap.release()
        cv2.destroyAllWindows()

    main()


@rp.memoized
def _xy_meshgrid(h,w,device,dtype):
    y, x = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
    )

    output = torch.stack(
        [x, y],
    ).to(device, dtype)

    assert output.shape == (2, h, w)
    return output

def xy_meshgrid_like_image(image):
    """
    Example:
        >>> image=load_image('https://picsum.photos/id/28/367/267')
        ... image=as_torch_image(image)
        ... xy=xy_meshgrid_like_image(image)
        ... display_image(full_range(as_numpy_array(xy[0])))
        ... display_image(full_range(as_numpy_array(xy[1])))
    """
    assert image.ndim == 3, "image is in CHW form"
    c, h, w = image.shape
    return _xy_meshgrid(h,w,image.device,image.dtype)

def noise_to_xyωc(noise):
    assert noise.ndim == 3, "noise is in CHW form"
    zeros=torch.zeros_like(noise[0][None])
    ones =torch.ones_like (noise[0][None])

    #Prepend [dx=0, dy=0, weights=1] channels
    output=torch.concat([zeros, zeros, ones, noise])
    return output

def xyωc_to_noise(xyωc):
    assert xyωc.ndim == 3, "xyωc is in [ω x y c]·h·w form"
    assert xyωc.shape[0]>3, 'xyωc should have at least one noise channel'
    noise=xyωc[3:]
    return noise

def warp_xyωc(
    I,
    F,
    xy_mode="none",
    # USED FOR ABLATIONS:
    expand_only=False,
):
    """
    For ablations, set:
        - expand_only=True #No contraction
        - expand_only='bilinear' #Bilinear Interpolation
        - expand_only='nearest' #Nearest Neighbors Warping
    """
    #Input assertions
    assert F.device==I.device
    assert F.ndim==3, str(F.shape)+' F stands for flow, and its in [x y]·h·w form'
    assert I.ndim==3, str(I.shape)+' I stands for input, in [ω x y c]·h·w form where ω=weights, x and y are offsets, and c is num noise channels'
    xyωc, h, w = I.shape
    assert F.shape==(2,h,w) # Should be [x y]·h·w
    device=I.device
    
    #How I'm going to address the different channels:
    x   = 0        #          // index of Δx channel
    y   = 1        #          // index of Δy channel
    xy  = 2        # I[:xy]
    xyω = 3        # I[:xyω]
    ω   = 2        # I[ω]     // index of weight channel
    c   = xyωc-xyω # I[-c:]   // num noise channels
    ωc  = xyωc-xy  # I[-ωc:]
    # h_dim = 1
    w_dim = 2
    assert c, 'I has no noise channels. There is nothing to warp.'
    assert (I[ω]>0).all(), 'All weights should be greater than 0'

    #Compute the grid of xy indices
    grid = xy_meshgrid_like_image(I)
    assert grid.shape==(2,h,w) # Shape is [x y]·h·w

    #The default values we initialize to. Todo: cache this.
    init = torch.empty_like(I)
    init[:xy]=0
    init[ω]=1
    init[-c:]=0

    #Caluclate initial pre-expand
    pre_expand = torch.empty_like(I)

    #The original plan was to use init xy during expand, because the query position is arbitrary....
    #It doesn't actually make deep sense to copy the offsets during this step, but it doesn't seem to hurt either...
    #BUT I think I got slightly better results...?...so I'm going to do it anyway.
    # pre_expand[:xy] = init[:xy] # <---- Original algorithm I wrote on paper

    #ABLATION STUFF IN THIS PARAGRAPH
    #Using F_index instead of F so we can use ablations like bilinear, bicubic etc
    interp = 'nearest' if not isinstance(expand_only, str) else expand_only
    regauss = not isinstance(expand_only, str)
    F_index = F
    if interp=='nearest':
        #Default behaviour, ablations or not
        F_index=F_index.round()

    pre_expand[:xy] = rp.torch_remap_image(I[:xy], * -F, relative=True, interp=interp)# <---- Last minute change
    pre_expand[-ωc:] = rp.torch_remap_image(I[-ωc:], * -F, relative=True, interp=interp)
    pre_expand[ω][pre_expand[ω]==0]=1 #Give new noise regions a weight of 1 - effectively setting it to init there

    if expand_only:
        if regauss:
            #This is an ablation option - simple warp + regaussianize
            #Enable to preview expansion-only noise warping
            #The default behaviour! My algo!
            pre_expand[-c:]=regaussianize(pre_expand[-c:])[0]
        else:
            #Turn zeroes to noise
            pre_expand[-c:]=torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:]==0) + pre_expand[-c:]
        return pre_expand

    #Calculate initial pre-shrink
    pre_shrink = I.clone()
    pre_shrink[:xy] += F

    #Pre-Shrink mask - discard out-of-bounds pixels
    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (0<= pos[x]) & (pos[x] < w) & (0<= pos[y]) & (pos[y] < h)
    in_bounds = in_bounds[None] #Match the shape of the input
    out_of_bounds = ~in_bounds
    assert out_of_bounds.dtype==torch.bool
    assert out_of_bounds.shape==(1,h,w)
    assert pre_shrink.shape == init.shape
    pre_shrink = torch.where(out_of_bounds, init, pre_shrink)

    #Deal with shrink positions offsets
    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy

    #FLOATING POINT POSITIONS: I will disable this for now. It does in fact increase sensitivity! But it also makes it less long-term coherent
    assert xy_mode in ['float', 'none'] or isinstance(xy_mode, int)
    if xy_mode=='none':
        pre_shrink[:xy] = 0 #DEBUG: Uncomment to ablate floating-point swarm positions

    if isinstance(xy_mode, int):
        # XY quantization: best to use odd numbers!
        quant = xy_mode
        pre_shrink[:xy] = (
            pre_shrink[:xy] * quant
        ).round() / quant  

    #OTHER ways I tried reducing sensitivity to motion. They work - but 0 is best. Let's just use high resolution.
    # pre_shrink[:xy][pre_shrink[:xy].abs()<.1] = 0  #DEBUG: Uncomment to ablate floating-point swarm positions
    # pre_shrink[:xy] *= -1 #I can't even tell that this is wrong.....
    # pre_shrink[:xy] *= .9 
    # sensitivity_factor = 4

    scat = lambda tensor: rp.torch_scatter_add_image(tensor, *scat_xy, relative=True)

    #Where mask==True, we output shrink. Where mask==0, we output expand.
    shrink_mask = torch.ones(1,h,w,dtype=bool,device=device) #The purpose is to get zeroes where no element is used
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype==torch.bool, 'If this fails we gotta convert it with mask.=astype(bool)'

    # rp.cv_imshow(rp.tiled_images([out_of_bounds[0],shrink_mask[0]]),label='OOB') ; return I #DEBUG - uncomment to see the masks

    #Remove the expansion points where we'll use shrink
    pre_expand = torch.where(shrink_mask, init, pre_expand)
    # rp.cv_imshow(pre_expand[-c:]/5+.5,'preex')

    #Horizontally Concat
    concat_dim = w_dim
    concat     = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    #Regaussianize
    concat[-c:], counts_image = regaussianize(concat[-c:])
    assert  counts_image.shape == (1, h, 2*w)
    # rp.cv_imshow(concat[-c:]/5+.5,label='regauss') ; return pre_expand #DEBUG - Uncomment to preview regaussianization

    #Distribute Weights
    concat[ω] /= counts_image[0]
    concat[ω] = concat[ω].nan_to_num() #We shouldn't need this, this is a crutch. Final mask should take care of this.

    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)
    assert pre_shrink.shape == expand.shape == (3+c, h, w)
 
    shrink = torch.empty_like(pre_shrink)
    shrink[ω]   = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy]*pre_shrink[ω][None]) / shrink[ω][None]
    shrink[-c:] = scat(pre_shrink[-c:]*pre_shrink[ω][None]) / scat(pre_shrink[ω][None]**2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[ω] = output[ω] / output[ω].mean() #Don't let them get too big or too small
    ε = .00001
    output[ω] += ε #Don't let it go too low
    
    # rp.debug_comment([output[ω].min(),output[ω].max()])# --> [tensor(0.0010), tensor(2.7004)]
    # rp.debug_comment([shrink[ω].min(),shrink[ω].max()])# --> [tensor(0.), tensor(2.7004)]
    # rp.debug_comment([expand[ω].min(),expand[ω].max()])# --> [tensor(0.0001), tensor(0.3892)]
    # rp.cv_imshow(rp.apply_colormap_to_image(output[ω]/output[ω].mean()/4),label='weight')
    # rp.cv_imshow(rp.apply_colormap_to_image(output[ω]/10),label='weight')
    assert (output[ω]>0).all()
    # print(end='\r%.08f %.08f'%(float(output[ω].min()), float(output[ω].max())))

    output[ω] **= .9999 #Make it tend towards 1


    return output


class NoiseWarper:
    def __init__(
        self,
        c, h, w,
        device,
        dtype=torch.float32,
        scale_factor=1,
        post_noise_alpha = 0,
        progressive_noise_alpha = 0,
        warp_kwargs=dict(),
    ):

        #Some non-exhaustive input assertions
        assert isinstance(c,int) and c>0
        assert isinstance(h,int) and h>0
        assert isinstance(w,int) and w>0
        assert isinstance(scale_factor,int) and w>=1

        #Record arguments
        self.c=c
        self.h=h
        self.w=w
        self.device=device
        self.dtype=dtype
        self.scale_factor=scale_factor
        self.progressive_noise_alpha=progressive_noise_alpha
        self.post_noise_alpha=post_noise_alpha
        self.warp_kwargs=warp_kwargs

        #Initialize the state
        self._state = self._noise_to_state(
            noise=torch.randn(
                c,
                h * scale_factor,
                w * scale_factor,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def noise(self):
        #TODO: The noise should be downsampled to respect the weights!! 
        noise = self._state_to_noise(self._state)
        weights = self._state[2][None] #xyωc
        noise = (
              rp.torch_resize_image(noise * weights, (self.h, self.w), interp="area")
            / rp.torch_resize_image(weights**2     , (self.h, self.w), interp="area").sqrt()
        )
        noise = noise * self.scale_factor

        if self.post_noise_alpha:
            noise = mix_new_noise(noise, self.post_noise_alpha)

        return noise

    def __call__(self, dx, dy):

        if rp.is_numpy_array(dx): dx = torch.tensor(dx).to(self.device, self.dtype)
        if rp.is_numpy_array(dy): dy = torch.tensor(dy).to(self.device, self.dtype)

        flow = torch.stack([dx, dy]).to(self.device, self.dtype)
        _, oflowh, ofloww = flow.shape #Original height and width of the flow
        
        assert flow.ndim == 3 and flow.shape[0] == 2, "Flow is in [x y]·h·w form"
        flow = rp.torch_resize_image(
            flow,
            (
                self.h * self.scale_factor,
                self.w * self.scale_factor,
            ),
        ) 

        _, flowh, floww = flow.shape

        #Multiply the flow values by the size change
        flow[0] *= flowh / oflowh * self.scale_factor
        flow[1] *= floww / ofloww * self.scale_factor

        self._state = self._warp_state(self._state, flow)
        return self

    #The following three methods can be overridden in subclasses:

    @staticmethod
    def _noise_to_state(noise):
        return noise_to_xyωc(noise)

    @staticmethod
    def _state_to_noise(state):
        return xyωc_to_noise(state)

    def _warp_state(self, state, flow):

        if self.progressive_noise_alpha:
            state[3:] = mix_new_noise(state[3:], self.progressive_noise_alpha)

        return warp_xyωc(state, flow, **self.warp_kwargs)
    
def blend_noise(noise_background, noise_foreground, alpha):
    """ Variance-preserving blend """
    return (noise_foreground * alpha + noise_background * (1-alpha))/(alpha ** 2 + (1-alpha) ** 2)**.5

def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed"""
    if isinstance(noise, torch.Tensor): return blend_noise(noise, torch.randn_like(noise)      , alpha)
    elif isinstance(noise, np.ndarray): return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else: raise TypeError(f"Unsupported input type: {type(noise)}. Expected PyTorch Tensor or NumPy array.")

def resize_noise(noise, size, alpha=None):
    """
    Can resize gaussian noise, adjusting for variance and preventing cross-correlation
    """

    if rp.is_numpy_array(noise):
        noise=rp.as_torch_image(noise)
        output = resize_noise(noise, size, alpha)
        output = rp.as_numpy_array(output)
        output = einops.rearrange(output, "C H W -> H W C")
        return output

    assert noise.ndim == 3, "resize_noise: noise should be a CHW tensor"
    num_channels, old_height, old_width = noise.shape

    if noise.ndim==4:
        #If given a batch of noises, do it for each one
        return torch.stack([resize_noise(x, new_height, new_width) for x in noise])

    if rp.is_number(size):
        new_height, new_width = int(old_height * size), int(old_width * size)
    else:
        new_height, new_width = size

    assert new_height<=old_height, 'resize_noise: Only useful for shrinking noise, not growing it'
    assert new_width <=old_width , 'resize_noise: Only useful for shrinking noise, not growing it'
    
    x, y = rp.xy_torch_matrices(
        old_height,
        old_width,
        max_x=new_width,
        max_y=new_height,
    )

    if alpha is not None:
        #Prepend the alpha
        assert alpha.ndim==2,alpha.shape
        assert alpha.shape==noise.shape[1:],(alpha.shape,noise.shape)
        noise=torch.cat((alpha[None],noise))
        
    resized = rp.torch_scatter_add_image(
        noise,
        x,
        y,
        height=new_height,
        width=new_width,
        interp='floor',
        prepend_ones=alpha is None
    )
    
    total, resized = resized[:1], resized[1:]

    adjusted = resized / total**.5

    return adjusted
    
def get_noise_from_video(
    video_path: str,
    noise_channels: int = 3,
    output_folder: str = None,
    visualize: bool = True,
    resize_frames: tuple = None,
    resize_flow: int = 1,
    downscale_factor: int = 1,
    device=None,
    video_preprocessor = None,
    save_files=True,
    progressive_noise_alpha = 0,
    post_noise_alpha = 0,
    remove_background=False,
    visualize_flow_sensitivity=None,
    warp_kwargs=dict(),
):
    """
    Extract noise from a video by warping random noise using optical flow between consecutive frames.

    If running this function in a Jupyter notebook, you'll see a live preview of the noise and visualization as it calculates.
  
    Args:
        video_path (str): Path to the input video file (MP4), 
                          a folder containing image frames,
                          or a glob pattern like "/path/to/images/*.png",
                          or a URL pointing to a video file online.
                          You can also pass a numpy video array in THWC form
        noise_channels (int, optional): Number of channels in the generated noise. Defaults to 3.
        output_folder (str, optional): Folder to save the output noise and visualization.
                                       Defaults to None, in which case the folder name is automatically chosen.
        visualize (bool, optional): Whether to generate visualization images and video. Defaults to True.
        resize_frames (tuple or float, optional): Size to resize the input frames.
                                                  If a tuple (height, width), resizes to the exact dimensions.
                                                  If a float, resizes both dimensions relatively and evenly. Defaults to None.
                                                  This is applied *before* calculating optical flow, so it will make flow calculation slower.
        resize_flow (int): Resize the optical flows calculated after resize_frames. Controls the underlying noise resolution.
                           Will not make flow calculation slower - effectively uses bilinear interpolation to upsample the flow.
                           Should be an integer > 0
        downscale_factor (int): Factor by which to downscale the generated noise.
                                Larger factor --> smaller noise image.
                                This factor should evenly divide the height and width of the video frames.
        video_preprocessor (callable): A function that takes a THW3 RGB uint8 video numpy array, and returns another THW3 numpy array
                                       This can be used for removing watermarks or resizing the video or any other preprocessing steps
                                       This is applied to the input video, directly after loading it from video_path
        save_files (bool): If True, will save files to disk.
        progressive_noise_alpha: For ryan, don't worry about it
        post_noise_alpha: For ryan, don't worry about it
        remove_background: If true will attempt to matte out the background and composite the computed noise on that background
        visualize_flow_sensitivity (float, optional): If specified, will normalize flow with the assumption that this given value is the max magnitude
        warp_kwargs (dict, optional): For experimental features. Don't worry about this if you're not Ryan Burgert.

    Returns:
        EasyDict: A dict containing the following keys:
            - 'numpy_noises' (np.ndarray): Generated noise with form [T, H, W, C].
            - 'numpy_flows' (np.ndarray): The (dx, dy)'s with form [T-1, 2, H, W]
            - 'vis_frames' (np.ndarray): Visualization frames with form [T, H, W, C].
            - 'output_folder' (str): The path to the folder where outputs are saved (if save_files)

    Examples:
        # Command line usage
        >>> python noise_warp.py --video_path /path/to/video.mp4 --noise_channels 3 --output_folder /path/to/output
        >>> python noise_warp.py --video_path /path/to/frames_folder --resize_frames 0.5 --downscale_factor 2
        >>> python noise_warp.py --video_path "/path/to/frames/frame_*.png" --resize_frames (256, 256)

        # Function call
        >>> from rp.git.CommonSource import *
        >>> numpy_noises, vis_frames = get_noise_from_video(
        ...     video_path="/path/to/video.mp4",
        ...     noise_channels=3,
        ...     output_folder="/path/to/output",
        ...     visualize=True,
        ...     resize_frames=0.5,
        ...     downscale_factor=2,
        ... )
        >>> video_demo("/root/CleanCode/Projects/flow_noise_warping/outputs/water/waves_bilinear.mp4", downscale_factor=4)
        >>> video_demo("/efs/users/ryan.burgert/public/sharing/KevinSpinnerNoiseWarping/diffuse_images_360", downscale_factor=8, resize_frames=.5)
        >>> video_demo("/root/CleanCode/Projects/flow_noise_warping/outputs/kevin_spinner/kevin_vps7.mp4", downscale_factor=4, resize_frames=.5)

    EXAMPLE - Jupyter Notebook Soup-to-nuts:

       >>> #Ryan Burgert 2024
       ... #Run this in a Jupyter notebook code cell for a realtime preview!
       ... 
       ... #Setup:
       ... #    Run this in a Jupyter Notebook on a computer with at least one GPU
       ... #        `sudo apt install ffmpeg git`
       ... #        `pip install rp`
       ... #    The first time you run this it might be a bit slow (it will download necessary models)
       ... #    The `rp` package will take care of installing the rest of the python packages for you
       ... 
       ... import rp
       ... 
       ... rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
       ... import rp.git.CommonSource.noise_warp as nw
       ... 
       ... FRAME = 2**-1 #We immediately resize the input frames by this factor, before calculating optical flow
       ...               #The flow is calulated at (input size) × FRAME resolution.
       ...               #Higher FLOW values result in slower optical flow calculation and higher intermediate noise resolution
       ...               #Larger is not always better - watch the preview in Jupyter to see if it looks good!
       ... FLOW = 2**4   #Then, we use bilinear interpolation to upscale the flow by this factor
       ...               #We warp the noise at (input size) × FRAME × FLOW resolution
       ...               #The noise is then downsampled back to (input size)
       ...               #Higher FLOW values result in more temporally consistent noise warping at the cost of higher VRAM usage and slower inference time
       ... LATENT = 8    #We further downsample the outputs by this amount - because 8 pixels wide corresponds to one latent wide in Stable Diffusion
       ...               #The final output size is (input size) ÷ LATENT regardless of FRAME and FLOW
       ... 
       ... LATENT = 2    #Uncomment this line for a prettier visualization! But for any Stable-Diffusion based model, use LATENT=8
       ... 
       ... 
       ... #You can pass a glob of image files
       ... images = "/path_to_images/cam_*_color.png"
       ... 
       ... #You can also use video files or URLs
       ... images = "https://www.shutterstock.com/shutterstock/videos/1100085499/preview/stock-footage-bremen-germany-october-old-style-carousel-moving-on-square-in-city-horses-on-traditional.webm"
       ... 
       ... output_folder = rp.get_folder_name(rp.get_parent_folder(images))
       ... 
       ... #See this function's docstring for more information!
       ... output = nw.get_noise_from_video(
       ...     images,
       ...     remove_background=False, #Set this to True to matte the foreground - and force the background to have no flow
       ...     visualize=True,          #Generates nice visualization videos and previews in Jupyter notebook
       ...     save_files=True,         #Set this to False if you just want the noises without saving to a numpy file
       ...     
       ...     noise_channels=4,
       ...     output_folder=output_folder,
       ...     resize_frames=FRAME,
       ...     resize_flow=FLOW,
       ...     downscale_factor=round(FRAME * FLOW) * LATENT,
       ... );
       ... 
       ... print("Noise shape:"  ,output.numpy_noises.shape)
       ... print("Flow shape:"   ,output.numpy_flows .shape)
       ... print("Output folder:",output.output_folder)

    """

    #Input assertions
    assert isinstance(resize_flow, int) and resize_flow >= 1, resize_flow

    if device is None:
        if rp.currently_running_mac():
            device = 'cpu'
        else:
            device = rp.select_torch_device(prefer_used=True)
    
    raft_model = raft.RaftOpticalFlow(device, "large")

    # Load video frames into a [T, H, W, C] numpy array, where C=3 and values are between 0 and 1
    # Can be specified as an MP4, a folder that contains images, or a glob like /path/to/*.png
    assert rp.is_numpy_array(video_path) or isinstance(video_path, str), type(video_path)
    if rp.is_video_file(video_path) or rp.is_valid_url(video_path):
        video_frames = rp.load_video(video_path)
    elif rp.is_numpy_array(video_path):
        #We can also pass a numpy video as an input in THWC form
        video_frames = video_path
        assert video_frames.ndim==4, video_frames.ndim
        video_path = rp.get_unique_copy_path('noisewarp_video.mp4')
    else:
        if rp.is_a_folder(video_path):
            frame_paths = rp.get_all_image_files(video_path, sort_by='number')
        else:
            frame_paths = glob.glob(video_path)
            frame_paths = sorted(sorted(frame_paths),key=len)
            if not frame_paths:
                raise ValueError(video_path + " is not a video file, a folder of images, or a glob containing images")
        video_frames = rp.load_images(frame_paths, show_progress=True)

    if video_preprocessor is not None:
        assert callable(video_preprocessor), type(video_preprocessor)
        video_frames = rp.as_numpy_array(video_frames)
        video_frames = video_preprocessor(video_frames)

    #If resize_frames is specified, resize all frames to that (height, width)
    if resize_frames is not None:
        rp.fansi_print("Resizing all input frames to size %s"%str(resize_frames), 'yellow')
        video_frames=rp.resize_images(video_frames, size=resize_frames, interp='area')

    if remove_background:
        alphas = []

        background_remover = BackgroundRemover(device)

        if visualize and rp.running_in_jupyter_notebook():
            alpha_display_channel = rp.JupyterDisplayChannel()
            alpha_display_channel.display()

        for video_frame in rp.eta(video_frames, title='Removing Backgrounds'):
            rgba_image = background_remover(video_frame)
            alpha = rp.get_alpha_channel(rgba_image)
            alpha = rp.as_float_image(alpha)
            alphas.append(alpha)

            if visualize and rp.running_in_jupyter_notebook():
                alpha_display_channel.update(
                    rp.horizontally_concatenated_images(
                        rp.with_alpha_checkerboard(rgba_image), alpha
                    )
                )

        del background_remover #Free GPU usage
        
        
    video_frames = rp.as_rgb_images(video_frames)
    video_frames = np.stack(video_frames)
    video_frames = video_frames.astype(np.float16)/255
    _, h, w, _ = video_frames.shape
    rp.fansi_print(f"Input video shape: {video_frames.shape}", 'yellow')
    print('check 1', video_frames.shape)

    if h%downscale_factor or w%downscale_factor:
        rp.fansi_print("WARNING: height {h} or width{w} is not divisible by the downscale_factor {downscale_factor}. This will lead to artifacts in the noise.")

    def downscale_noise(noise):
        down_noise = rp.torch_resize_image(noise, 1/downscale_factor, interp='area') #Avg pooling
        down_noise = down_noise * downscale_factor #Adjust for STD
        return down_noise

    # Decide the location of and create the output folder
    if save_files:
        if output_folder is None:
            output_folder = "outputs/" + rp.get_file_name(video_path, include_file_extension=False)
        output_folder = rp.make_directory(rp.get_unique_copy_path(output_folder))
        rp.fansi_print("Output folder: " + output_folder, "green")

    with torch.no_grad():

        if visualize and rp.running_in_jupyter_notebook():
            # For previewing results in Jupyter notebooks, if applicable
            display_channel = rp.JupyterDisplayChannel()
            display_channel.display()

        warper = NoiseWarper(
            c = noise_channels,
            h = resize_flow * h,
            w = resize_flow * w,
            device = device,
            post_noise_alpha = post_noise_alpha,
            progressive_noise_alpha = progressive_noise_alpha,
            warp_kwargs = warp_kwargs,
        )

        prev_video_frame = video_frames[0]
        print('check 2', prev_video_frame.shape)
        noise = warper.noise

        down_noise = downscale_noise(noise)
        numpy_noise = rp.as_numpy_image(down_noise).astype(np.float16) # In HWC form. Using float16 to save RAM, but it might cause problems on come CPU

        numpy_noises = [numpy_noise]
        numpy_flows = []
        vis_frames = []

        try:
            for index, video_frame in enumerate(tqdm(video_frames[1:])):

                dx, dy = raft_model(prev_video_frame, video_frame)
                print('check 3', dx.shape)
                noise = warper(dx, dy).noise
                prev_video_frame = video_frame

                numpy_flow = np.stack(
                    [
                        rp.as_numpy_array(dx).astype(np.float16),
                        rp.as_numpy_array(dy).astype(np.float16),
                    ]
                )
                numpy_flows.append(numpy_flow)

                down_noise = downscale_noise(noise)

                numpy_noise = rp.as_numpy_image(down_noise).astype(np.float16)

                if remove_background:
                    if 'background_noise' not in dir():
                        background_noise = np.random.randn(*numpy_noise.shape)
                    numpy_noise_alpha = alphas[index]
                    numpy_noise_alpha = rp.cv_resize_image(numpy_noise_alpha, numpy_noise.shape[:2])
                    numpy_noise = blend_noise(background_noise, numpy_noise, numpy_noise_alpha[:,:,None])
                        
                numpy_noises.append(numpy_noise)

                if visualize:
                    flow_rgb = rp.optical_flow_to_image(dx, dy, sensitivity = visualize_flow_sensitivity)

                    #Turn the noise into a numpy HWC RGB array
                    down_noise_image = np.zeros((*numpy_noise.shape[:2], 3))
                    down_noise_image_c = min(noise_channels,3)
                    down_noise_image[:,:,:down_noise_image_c]=numpy_noise[:,:,:down_noise_image_c]


                    down_size = rp.get_image_dimensions(down_noise_image)
                    down_video_frame, down_flow_rgb = rp.resize_images(video_frame, flow_rgb, size=down_size)

                    optional_images = []
                    optional_labels = []
                    if remove_background:
                        alpha = alphas[index]
                        down_alpha = rp.cv_resize_image(alpha, down_size)

                        optional_images.append(down_alpha)
                        optional_labels.append('Alpha')

                        optional_images.append(rp.with_alpha_checkerboard(rp.with_image_alpha(down_video_frame,down_alpha)))
                        optional_labels.append('RGBA')

                    visualization = rp.as_byte_image(
                        rp.tiled_images(
                            rp.labeled_images(
                                [
                                    down_noise_image / 3 + 0.5,
                                    down_video_frame,
                                    down_flow_rgb,
                                    down_noise_image / 5 + down_video_frame,
                                ] + optional_images,
                                [
                                    "Warped Noise",
                                    "Input Video",
                                    "Optical Flow",
                                    "Overlaid",
                                ] + optional_labels,
                                font = 'G:Zilla Slab',
                            )
                        )
                    )

                    if rp.running_in_jupyter_notebook():
                        display_channel.update(visualization)

                    vis_frames.append(visualization)

        except KeyboardInterrupt:
            rp.fansi_print("Interrupted! Returning %i noises" % len(numpy_noises), "cyan", "bold")
            pass

    numpy_noises = np.stack(numpy_noises).astype(np.float16)
    numpy_flows = np.stack(numpy_flows).astype(np.float16)
    if vis_frames: vis_frames = np.stack(vis_frames)

    if save_files and len(vis_frames):
        vis_img_folder = rp.make_directory(output_folder + "/visualization_images")
        vis_img_paths = rp.path_join(vis_img_folder, "visual_%05i.png")
        rp.save_images(vis_frames, vis_img_paths, show_progress=True)

        if "ffmpeg" in rp.get_system_commands():
            vis_mp4_path = rp.path_join(output_folder, "visualization_video.mp4")
            noise_mp4_path = rp.path_join(output_folder, "noise_video.mp4")
            rp.save_video_mp4(
                vis_frames,
                vis_mp4_path,
                video_bitrate="max",
                framerate=30,
            )
            rp.save_video_mp4(
                (numpy_noises / 4 + 0.5)[:,:,:,:3],
                noise_mp4_path,
                video_bitrate="max",
                framerate=30,
            )
            if rp.is_video_file(video_path):
                try:
                    #If possible, try to add the original audio and framerate back again
                    #Only makes sense if the input was an MP4 file and not a folder of images etc
                    for output_video_path in [vis_mp4_path, noise_mp4_path]:
                        rp.fansi_print(
                            "Added audio to output at: "
                            + rp.add_audio_to_video_file(
                                rp.printed(
                                    rp.change_video_file_framerate(
                                        output_video_path,
                                        rp.get_video_file_framerate(video_path),
                                    )
                                ),
                                video_path,
                            ),
                            "green",
                            "bold",
                        )
                except Exception:
                    rp.print_stack_trace()
        else:
            rp.fansi_print("Please install ffmpeg! We won't save an MP4 this time - please try again.")

    if save_files:
        noises_path = rp.path_join(output_folder, "noises.npy")
        flows_path = rp.path_join(output_folder, "flows_dxdy.npy")
        np.save(noises_path, numpy_noises)
        rp.fansi_print("Saved " + noises_path + " with shape " + str(numpy_noises.shape), "green")
        np.save(flows_path, numpy_flows)
        rp.fansi_print("Saved " + flows_path + " with shape " + str(numpy_flows.shape), "green")
        
        rp.fansi_print(rp.get_file_name(__file__)+": Done warping noise, results are at " + rp.get_absolute_path(output_folder), "green", "bold")

    return rp.gather_vars('numpy_noises numpy_flows vis_frames output_folder')

if __name__ == '__main__':
    fire.Fire(dict(rp.gather_vars('get_noise_from_video demo_webcam_noise_warp')))
