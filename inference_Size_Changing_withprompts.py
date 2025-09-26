import os

import torch
import numpy as np
from pathlib import Path
from matplotlib.path import Path as MplPath
from models.nnd import NewtonODELatent
import rp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("")


config_list = [
    dict(
        z0=[6.4833, 5.5226, 0.0, 0.0, 0.0, 0.0, 0.2008, 0.2008, 0.0632],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_a"
    ),
    dict(
        z0=[7.1445, 6.5251, 0.0, 0.0, 0.0, 0.0, 0.4196, 0.4196, 0.2766],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_b"
    ),
    dict(
        z0=[7.7653, 5.6317, 0.0, 0.0, 0.0, 0.0, 0.5246, 0.5246, 0.4322],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_c"
    ),
    dict(
        z0=[7.8777, 5.4010, 0.0, 0.0, 0.0, 0.0, 0.3218, 0.3218, 0.1628],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_d"
    ),
    dict(
        z0=[8.3346, 6.7342, 0.0, 0.0, 0.0, 0.0, 0.5936, 0.5936, 0.5534],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_e"
    ),
    dict(
        z0=[9.2809, 5.9461, 0.0, 0.0, 0.0, 0.0, 0.1276, 0.1276, 1.9976],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_f"
    ),
    dict(
        z0=[10.0759, 5.0603, 0.0, 0.0, 0.0, 0.0, 0.4828, 0.4828, 0.366],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_g"
    ),
    dict(
        z0=[11.4364, 5.0559, 0.0, 0.0, 0.0, 0.0, 0.2457, 0.2457, 0.328],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_h"
    ),
    dict(
        z0=[11.4629, 6.2376, 0.0, 0.0, 0.0, 0.0, 1.12, 1.12, 1.9702],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_i"
    ),
    dict(
        z0=[11.9719, 5.7067, 0.0, 0.0, 0.0, 0.0, 0.124, 0.124, 2.2332],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_j"
    ),
]


T_pred = 48
H, W = 240, 360

model = NewtonODELatent().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# 遍历每个配置
for cfg in config_list:
    print(f"Processing config: {cfg['output_name']}")

    z0_vals = cfg['z0']
    DT = cfg['DT']
    METER_PER_PX = cfg['METER_PER_PX']
    chosen_shape = cfg['chosen_shape']

    z0 = torch.tensor([z0_vals], dtype=torch.float32, device=DEVICE)
    ts = torch.arange(T_pred, dtype=torch.float32, device=DEVICE) * DT

    # ---------------- STEP 1
    with torch.no_grad():
        dynamics = model(z0, ts)
    dynamics = dynamics.squeeze(0).cpu().numpy()

    out_dir = Path(f"inference/size/{cfg['output_name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(dynamics), out_dir / f"inference_dynamics_{cfg['output_name']}.pt")
    np.save(out_dir / f"dynamics_{cfg['output_name']}_world.npy", dynamics[:, :9])

    traj_world = dynamics[:, :4]
    traj_px = np.zeros_like(traj_world)
    traj_px[:, 0] = traj_world[:, 0] / METER_PER_PX
    traj_px[:, 1] = H - (traj_world[:, 1] / METER_PER_PX)
    traj_px[:, 2] = traj_world[:, 2] / METER_PER_PX
    traj_px[:, 3] = -traj_world[:, 3] / METER_PER_PX
    np.save(out_dir / f"traj_pixel_{cfg['output_name']}.npy", traj_px)

    def make_mask(shape, X, Y, cx, cy, scale, theta=0.0):
        if shape == "circle":
            return (X - cx) ** 2 + (Y - cy) ** 2 <= scale**2
        elif shape == "square":
            return (np.abs(X - cx) <= scale) & (np.abs(Y - cy) <= scale)
        elif shape == "rectangle":
            short_edge, long_edge = scale

            Xr = X - cx
            Yr = Y - cy
            X_rot = Xr * np.cos(theta) - Yr * np.sin(theta)
            Y_rot = Xr * np.sin(theta) + Yr * np.cos(theta)
            return (np.abs(X_rot) <= long_edge/2) & (np.abs(Y_rot) <= short_edge/2)
        elif shape == "ellipse":
            short_axis, long_axis = scale

            Xr = X - cx
            Yr = Y - cy
            X_rot = Xr * np.cos(theta) - Yr * np.sin(theta)
            Y_rot = Xr * np.sin(theta) + Yr * np.cos(theta)
            return (X_rot**2)/(long_axis**2) + (Y_rot**2)/(short_axis**2) <= 1
        elif shape == "triangle":
            Xr, Yr = X - cx, Y - cy
            h = np.sqrt(3) * scale
            return (Yr >= -h/2) & (Yr <= h/2) & (np.abs(Xr) <= (h/2 - Yr/np.sqrt(3)))
        elif shape == "diamond":
            return np.abs(X - cx) + np.abs(Y - cy) <= scale
        elif shape in ["pentagon","hexagon"]:
            n = 5 if shape=="pentagon" else 6
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            verts = np.stack([cx + scale*np.cos(angles), cy + scale*np.sin(angles)], axis=-1)
            path = MplPath(verts)
            coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
            return path.contains_points(coords).reshape(X.shape)
        else:
            raise ValueError(f"Unknown shape: {shape}")

    flows = np.zeros((T_pred, 2, H, W), dtype=np.float32)
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    s_param = dynamics[:, 6]
    l_param = dynamics[:, 7]
    theta = dynamics[:, 4]


    flows = np.zeros((T_pred, 2, H, W), dtype=np.float32) 
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    speed_factor = 2.0

    # radius_seq = dynamics[:, 6] / METER_PER_PX
    radius_seq = s_param / METER_PER_PX

    for t in range(T_pred - 1):
        cx, cy = traj_px[t, 0], traj_px[t, 1]
        radius = radius_seq[t]

        dist_x = X - cx
        dist_y = Y - cy
        mask = dist_x**2 + dist_y**2 <= radius**2

        flows[t, 0][mask] = dist_x[mask] / (radius + 1e-5) * speed_factor
        flows[t, 1][mask] = dist_y[mask] / (radius + 1e-5) * speed_factor


    np.save(out_dir / f"flows_dxdy_{cfg['output_name']}.npy", flows)


    # ---------------- STEP 2: NoiseWarp ----------------
    import rp.git.CommonSource.noise_warp_new as nw
    flows_path = str(out_dir / f"flows_dxdy_{cfg['output_name']}.npy")
    T_minus_1, _, H, W = flows.shape
    video = np.zeros((T_minus_1 + 1, H, W, 3), dtype=np.uint8)
    noisewarp_out = nw.get_noise_from_video(
        video,
        remove_background=False,
        visualize=True,
        save_files=True,
        noise_channels=16,
        output_folder=f"inference/size/NoiseWarp_{cfg['output_name']}",
        resize_frames=1,
        resize_flow=1,
        downscale_factor=4,
        external_flows_path=flows_path,
    )




import torch
import numpy as np
import einops

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video, load_image
from icecream import ic
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel 
from transformers import T5EncoderModel

import rp.git.CommonSource.noise_warp as nw

import random

seed = 100
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



#pipe_ids = dict(
#    T2V5B="THUDM/CogVideoX-5b",
#)

# From a bird's-eye view, a serene scene unfolds: a herd of deer gracefully navigates shallow, warm-hued waters, their silhouettes stark against the earthy tones. The deer, spread across the frame, cast elongated, well-defined shadows that accentuate their antlers, creating a mesmerizing play of light and dark. This aerial perspective captures the tranquil essence of the setting, emphasizing the harmonious contrast between the deer and their mirror-like reflections on the water's surface. The composition exudes a peaceful stillness, yet the subtle movement suggested by the shadows adds a dynamic layer to the natural beauty and symmetry of the moment.
#base_url = "https://huggingface.co/Eyeline-Research/Go-with-the-Flow/resolve/main/"
#lora_urls = dict(
#    T2V5B_blendnorm_i18000_DATASET_lora_weights   = base_url+'T2V5B_blendnorm_i18000_DATASET_lora_weights.safetensors',
#)

pipe_ids = dict(
    T2V5B="/home/yuan418/data/project/THUT2V5b/ckpts/",
)

# From a bird's-eye view, a serene scene unfolds: a herd of deer gracefully navigates shallow, warm-hued waters, their silhouettes stark against the earthy tones. The deer, spread across the frame, cast elongated, well-defined shadows that accentuate their antlers, creating a mesmerizing play of light and dark. This aerial perspective captures the tranquil essence of the setting, emphasizing the harmonious contrast between the deer and their mirror-like reflections on the water's surface. The composition exudes a peaceful stillness, yet the subtle movement suggested by the shadows adds a dynamic layer to the natural beauty and symmetry of the moment.
lora_urls = dict(
    T2V5B_blendnorm_i18000_DATASET_lora_weights   = '/home/yuan418/data/project/goflow/lora_models/T2V5B_blendnorm_i18000_DATASET_lora_weights.safetensors',
)


dtype=torch.bfloat16

#https://medium.com/@ChatGLM/open-sourcing-cogvideox-a-step-towards-revolutionizing-video-generation-28fa4812699d
B, F, C, H, W = 1, 13, 16, 60, 90  # The defaults
num_frames=(F-1)*4+1 #https://miro.medium.com/v2/resize:fit:1400/format:webp/0*zxsAG1xks9pFIsoM
#Possible num_frames: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49
assert num_frames==49

@rp.memoized #Torch never manages to unload it from memory anyway
def get_pipe(model_name, device=None, low_vram=True):
    """
    model_name is like "I2V5B", "T2V2B", or "T2V5B", or a LoRA name like "T2V2B_RDeg_i30000_lora_weights"
    device is automatically selected if unspecified
    low_vram, if True, will make the pipeline use CPU offloading
    """

    if model_name in pipe_ids:
        lora_name = None
        pipe_name = model_name
    else:
        #By convention, we have lora_paths that start with the pipe names
        rp.fansi_print(f"Getting pipe name from model_name={model_name}",'cyan','bold')
        lora_name = model_name
        pipe_name = lora_name.split('_')[0]

    is_i2v = "I2V" in pipe_name  # This is a convention I'm using right now

    pipe_id = pipe_ids[pipe_name]
    print(f"LOADING PIPE WITH device={device} pipe_name={pipe_name} pipe_id={pipe_id} lora_name={lora_name}" )
    
    hub_model_id = pipe_ids[pipe_name]

    transformer = CogVideoXTransformer3DModel.from_pretrained(hub_model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    text_encoder = T5EncoderModel.from_pretrained(hub_model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLCogVideoX.from_pretrained(hub_model_id, subfolder="vae", torch_dtype=torch.bfloat16)

    PipeClass = CogVideoXImageToVideoPipeline if is_i2v else CogVideoXPipeline
    pipe = PipeClass.from_pretrained(hub_model_id, torch_dtype=torch.bfloat16, vae=vae,transformer=transformer,text_encoder=text_encoder)

    if lora_name is not None:
        lora_folder = rp.make_directory('lora_models')
        lora_url = lora_urls[lora_name]

        lora_path = lora_url  # 直接使用本地路径
        assert rp.file_exists(lora_path), (lora_name, lora_path)


        print(end="\tLOADING LORA WEIGHTS...",flush=True)
        pipe.load_lora_weights(lora_path)
        print("DONE!")

    if device is None:
        device = rp.select_torch_device()

    if not low_vram:
        print("\tUSING PIPE DEVICE", device)
        pipe = pipe.to(device)
    else:
        print("\tUSING PIPE DEVICE WITH CPU OFFLOADING",device)
        pipe=pipe.to('cpu')
        pipe.enable_sequential_cpu_offload(device=device)

    # pipe.vae.enable_tiling()
    # pipe.vae.enable_slicing()

    # Metadata
    pipe.lora_name = lora_name
    pipe.pipe_name = pipe_name
    pipe.is_i2v    = is_i2v
    # pipe.is_v2v    = is_v2v
    
    return pipe

def get_downtemp_noise(noise, noise_downtemp_interp):
    assert noise_downtemp_interp in {'nearest', 'blend', 'blend_norm', 'randn'}, noise_downtemp_interp
    if   noise_downtemp_interp == 'nearest'    : return                  rp.resize_list(noise, 13)
    elif noise_downtemp_interp == 'blend'      : return                   downsamp_mean(noise, 13)
    elif noise_downtemp_interp == 'blend_norm' : return normalized_noises(downsamp_mean(noise, 13))
    elif noise_downtemp_interp == 'randn'      : return torch.randn_like(rp.resize_list(noise, 13)) #Basically no warped noise, just r
    else: assert False, 'impossible'

def downsamp_mean(x, l=13):
    return torch.stack([rp.mean(u) for u in rp.split_into_n_sublists(x, l)])

def normalized_noises(noises):
    #Noises is in TCHW form
    return torch.stack([x / x.std(1, keepdim=True) for x in noises])


@rp.memoized
def load_sample_cartridge(
    sample_path: str,
    degradation=0,
    noise_downtemp_interp='nearest',
    image=None,
    prompt=None,
    num_inference_steps=30,
    guidance_scale=6,
):
    noise=None
    video=None

    if rp.is_a_folder(sample_path):

        print(end="LOADING CARTRIDGE FOLDER "+sample_path+"...")

        noise_file = rp.path_join(sample_path, 'noises.npy')
        instance_noise = np.load(noise_file)
        instance_noise = torch.tensor(instance_noise)
        instance_noise = einops.rearrange(instance_noise, 'F H W C -> F C H W')

        sample = rp.as_easydict(
            instance_prompt = '',
            instance_noise  = instance_noise,
            instance_video  = None,
        )

        print("DONE!")

    else:
        print(end="LOADING CARTRIDGE FILE "+sample_path+"...")
        sample = rp.file_to_object(sample_path)
        print("DONE!")

    # -
    sample_noise  = sample["instance_noise"].to(dtype)
    sample_prompt = sample["instance_prompt"]

    downtemp_noise = get_downtemp_noise(
        sample_noise,
        noise_downtemp_interp=noise_downtemp_interp,
    )
    downtemp_noise = downtemp_noise[None]
    downtemp_noise = nw.mix_new_noise(downtemp_noise, degradation)

    assert downtemp_noise.shape == (B, F, C, H, W), (downtemp_noise.shape, (B, F, C, H, W))


    if image is None:
        dummy = (sample_noise[0, :3] / 2 + 0.5).clamp(0, 1)
        sample_image = rp.as_pil_image(rp.as_numpy_image(dummy))
    elif isinstance(image, str):
        sample_image = rp.as_pil_image(rp.as_rgb_image(rp.load_image(image)))
    else:
        sample_image = rp.as_pil_image(rp.as_rgb_image(image))

    metadata = rp.gather_vars('sample_path degradation downtemp_noise sample_noise noise_downtemp_interp')
    settings = rp.gather_vars('num_inference_steps guidance_scale')

    if noise  is None: noise  = downtemp_noise
    if video  is None: video  = None
    if image  is None: image  = sample_image
    if prompt is None: prompt = sample_prompt

    assert noise.shape == (B, F, C, H, W), (noise.shape, (B, F, C, H, W))

    return rp.gather_vars('prompt noise image video metadata settings')



def dict_to_name(d=None, **kwargs):
    """
    Used to generate MP4 file names
    
    EXAMPLE:
        >>> dict_to_name(dict(a=5,b='hello',c=None))
        ans = a=5,b=hello,c=None
        >>> name_to_dict(ans)
        ans = {'a': '5', 'b': 'hello', 'c': 'None'}
    """
    if d is None:
        d = {}
    d.update(kwargs)
    return ",".join("=".join(map(str, [key, value])) for key, value in d.items())


def get_output_path(pipe, cartridge, subfolder:str, output_root:str):
    """
    Generates a unique output path for saving a generated video.

    Args:
        pipe: The video generation pipeline used.
        cartridge: Data used for generating the video.
        subfolder (str): Subfolder for saving the video.
        output_root (str): Root directory for output videos.

    Returns:
        String representing the unique path to save the video.
    """

    time = rp.millis()

    output_name = (
        dict_to_name(
            t=time,
            pipe=pipe.pipe_name,
            lora=pipe.lora_name,
            steps    =               cartridge.settings.num_inference_steps,
            # strength =               cartridge.settings.v2v_strength,
            degrad   =               cartridge.metadata.degradation,
            downtemp =               cartridge.metadata.noise_downtemp_interp,
            samp     = rp.get_file_name(rp.get_parent_folder(cartridge.metadata.sample_path), False),
        )
        + ".mp4"
    )

    output_path = rp.get_unique_copy_path(
        rp.path_join(
            rp.make_directory(
                rp.path_join(output_root, subfolder),
            ),
            output_name,
        ),
    )

    rp.fansi_print(f"OUTPUT PATH: {rp.fansi_highlight_path(output_path)}", "blue", "bold")

    return output_path

def run_pipe(
    pipe,
    cartridge,
    subfolder="first_subfolder",
    output_root: str = "infer_outputs",
    output_mp4_path = None, #This overrides subfolder and output_root if specified
):
    # output_mp4_path = output_mp4_path or get_output_path(pipe, cartridge, subfolder, output_root)

    if rp.file_exists(output_mp4_path):
        raise RuntimeError("{output_mp4_path} already exists! Please choose a different output file or delete that one. This script is designed not to clobber previous results.")
    
    if pipe.is_i2v:
        image = cartridge.image
        if isinstance(image, str):
            image = rp.load_image(image,use_cache=True)
        image = rp.as_pil_image(rp.as_rgb_image(image))

    # if pipe.is_v2v:
    #     print("Making v2v video...")
    #     v2v_video=cartridge.video
    #     v2v_video=rp.as_numpy_images(v2v_video) / 2 + .5
    #     v2v_video=rp.as_pil_images(v2v_video)

    print("NOISE SHAPE",cartridge.noise.shape)
    #print("IMAGE",image)

    video = pipe(
        prompt=cartridge.prompt,
        **(dict(image   =image                          ) if pipe.is_i2v else {}),
        # **(dict(strength=cartridge.settings.v2v_strength) if pipe.is_v2v else {}),
        # **(dict(video   =v2v_video                      ) if pipe.is_v2v else {}),
        num_inference_steps=cartridge.settings.num_inference_steps,
        latents=cartridge.noise,

        guidance_scale=cartridge.settings.guidance_scale,
        # generator=torch.Generator(device=device).manual_seed(42),
    ).frames[0]

    export_to_video(video, output_mp4_path, fps=8)

   # sample_gif=rp.load_video(cartridge.metadata.sample_gif_path)
    video=rp.as_numpy_images(video)

    return rp.gather_vars('video output_mp4_path  cartridge subfolder')


def main(
    sample_path,
    output_mp4_path:str,
    prompt=None,
    degradation=.5,
    model_name='I2V5B_final_i38800_nearest_lora_weights',

    low_vram=True,
    device:str=None,
    
    #BROADCASTABLE:
    noise_downtemp_interp='nearest',
    image=None,
    num_inference_steps=30,
    guidance_scale=6,
):
    """
    Main function to run the video generation pipeline with specified parameters.

    Args:
        model_name (str): Name of the pipeline to use ('T2V5B', 'T2V2B', 'I2V5B', etc).
        device (str or int, optional): Device to run the model on (e.g., 'cuda:0' or 0). If unspecified, the GPU with the  most free VRAM will be chosen.
        low_vram (bool): Set to True if you have less than 32GB of VRAM. In enables model cpu offloading, which slows down inference but needs much less vram.
        sample_path (str or list, optional): Broadcastable. Path(s) to the sample `.pkl` file(s) or folders containing (noise.npy and input.mp4 files)
        degradation (float or list): Broadcastable. Degradation level(s) for the noise warp (float between 0 and 1).
        noise_downtemp_interp (str or list): Broadcastable. Interpolation method(s) for down-temporal noise. Options: 'nearest', 'blend', 'blend_norm'.
        image (str, PIL.Image, or list, optional): Broadcastable. Image(s) to use as the initial frame(s). Can be a URL or a path to an image.
        prompt (str or list, optional): Broadcastable. Text prompt(s) for video generation.
        num_inference_steps (int or list): Broadcastable. Number of inference steps for the pipeline.
    """
    output_root='infer_outputs', # output_root (str): Root directory where output videos will be saved.
    subfolder='default_subfolder', # subfolder (str): Subfolder within output_root to save outputs.

    if device is None:
        device = rp.select_torch_device(reserve=True, prefer_used=True)
        rp.fansi_print(f"Selected torch device: {device}")


    cartridge_kwargs = rp.broadcast_kwargs(
        rp.gather_vars(
            "sample_path",
            "degradation",
            "noise_downtemp_interp",
            "image",
            "prompt",
            "num_inference_steps",
            "guidance_scale",
        )
    )

    rp.fansi_print("cartridge_kwargs:", "cyan", "bold")
    print(
        rp.indentify(
            rp.with_line_numbers(
                rp.fansi_pygments(
                    rp.autoformat_json(cartridge_kwargs),
                    "json",
                ),
                align=True,
            )
        ),
    )

    # cartridges = [load_sample_cartridge(**x) for x in cartridge_kwargs]
    cartridges = rp.load_files(lambda x:load_sample_cartridge(**x), cartridge_kwargs, show_progress='eta:Loading Cartridges')

    pipe = get_pipe(model_name, device, low_vram=low_vram)

    output=[]
    for cartridge in cartridges:
        pipe_out = run_pipe(
            pipe=pipe,
            cartridge=cartridge,
            output_root=output_root,
            subfolder=subfolder,
            output_mp4_path=output_mp4_path,
        )

        output.append(
            rp.as_easydict(
                rp.gather(
                    pipe_out,
                    [
                        "output_mp4_path",
                    ],
                    as_dict=True,
                )
            )
        )
    return output





if __name__ == "__main__":

    prompt_list = [
    "A red helium balloon gradually inflating in a sunny park, children playing in the background, trees casting soft shadows, captured from a stationary side camera.",
    "A blue water balloon slowly expanding on a kitchen counter, sunlight streaming through the window, utensils and fruit bowl nearby, observed from a fixed overhead camera.",
    "A yellow birthday balloon inflating on a party table, colorful confetti scattered around, party decorations hanging in the background, viewed from a side-angle camera.",
    "A green balloon gradually inflating on a street vendor's cart, urban buildings and pedestrians passing by, sunlight reflecting off nearby windows, captured from a sidewalk camera.",
    "A pink balloon inflating on a balcony, city skyline visible in the distance, potted plants and railings in the foreground, observed from a fixed camera at eye level.",
    "A transparent water balloon expanding in a laboratory, scientific instruments and glassware around, bright fluorescent lights overhead, captured from a fixed top-down camera.",
    "An orange balloon gradually inflating in a backyard, grass and flowers around, a dog running in the background, sunlight creating natural shadows, viewed from a side-angle camera.",
    "A purple balloon slowly inflating on a wooden picnic table near a lake, reflections of trees and water on the table, distant mountains visible, captured from a fixed camera on the shore.",
    "A white balloon inflating in a cozy living room, sunlight coming through curtains, sofa and bookshelf visible in the background, observed from a fixed corner camera.",
    "A metallic silver balloon gradually expanding in a festive hall, fairy lights and streamers in the background, colorful reflections on the floor, captured from a stationary overhead camera.",
    "A multicolored balloon inflating in a school playground, children running around, slides and swings visible, sunlight casting long shadows, observed from a high-angle camera.",
    "A large red party balloon slowly inflating on a balcony terrace, city rooftops and distant skyscrapers visible, potted plants and railing in the foreground, captured from a side camera."
    ]


    outputs = []

    for cfg in config_list:

        sample_path = f"inference/size/NoiseWarp_{cfg['output_name']}" 

        for i, prompt in enumerate(prompt_list):

            output_mp4_path = f"inference/size/{cfg['output_name']}_prompt{i+1}.mp4"

            print(f"Processing config {cfg['output_name']} with prompt {i+1}")

            out = main(
                sample_path=sample_path,
                output_mp4_path=output_mp4_path,
                prompt=prompt,
                degradation=0.5,
                model_name="T2V5B_blendnorm_i18000_DATASET_lora_weights",
                low_vram=True,
                device=None,
                noise_downtemp_interp="nearest",
                image=None,
                num_inference_steps=30,
                guidance_scale=6,
            )

            outputs.append(out)

    print("All videos generated.")



