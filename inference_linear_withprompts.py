import os
os.environ['HF_HOME'] = ''
os.environ['TRANSFORMERS_CACHE'] = ''

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
        z0=[6.9901, 9.3459, 5.558, -4.8493, 0.0, 0.0, 0.594, 0.7497, 0.4453],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_a"
    ),
    dict(
        z0=[7.2499, 9.9899, 5.4111, -2.2692, 0.0, 0.0, 0.5469, 0.6043, 0.3305],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="rectangle",
        output_name="set_b"
    ),
    dict(
        z0=[7.3491, 8.542, 4.4844, -2.0627, 0.0, 0.0, 0.6509, 0.745, 0.4849],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="circle",
        output_name="set_c"
    ),
    dict(
        z0=[8.266, 10.675, 4.2731, -4.0035, 0.0, 0.0, 0.6826, 0.7036, 0.4803],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="rectangle",
        output_name="set_d"
    ),
    dict(
        z0=[2.2178, 9.8073, 5.7966, -2.0545, 0.0, 0.0, 0.6929, 0.6425, 0.4452],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="rectangle",
        output_name="set_e"
    ),
    dict(
        z0=[2.6699, 10.4329, 6.7211, -4.0865, 0.0, 0.0, 0.6914, 0.7089, 0.4901],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="rectangle",
        output_name="set_f"
    ),
    dict(
        z0=[3.1774, 10.5559, 6.1732, -4.4105, 0.0, 0.0, 0.5991, 0.6185, 0.3705],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="rectangle",
        output_name="set_g"
    ),
    dict(
        z0=[3.1987, 9.5647, 6.3423, -2.4199, 0.0, 0.0, 0.6423, 0.7866, 0.5053],
        DT=0.02,
        METER_PER_PX=0.05,
        chosen_shape="rectangle",
        output_name="set_h"
    ),
    dict(
        z0=[3.6515, 8.7597, 4.9518, -3.6461, 0.0, 0.0, 0.5793, 0.672, 0.3893],
        DT=0.02,
        METER_PER_PX=0.1,
        chosen_shape="rectangle",
        output_name="set_i"
    ),
    dict(
        z0=[3.9619, 9.5815, 5.4972, -3.1354, 0.0, 0.0, 0.6506, 0.7005, 0.4558],
        DT=0.02,
        METER_PER_PX=0.1,
        chosen_shape="rectangle",
        output_name="set_j"
    ),
]



T_pred = 48
H, W = 240, 360

model = NewtonODELatent().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()


for cfg in config_list:
    print(f"Processing config: {cfg['output_name']}")

    z0_vals = cfg['z0']
    DT = cfg['DT']
    METER_PER_PX = cfg['METER_PER_PX']
    chosen_shape = cfg['chosen_shape']

    z0 = torch.tensor([z0_vals], dtype=torch.float32, device=DEVICE)
    ts = torch.arange(T_pred, dtype=torch.float32, device=DEVICE) * DT


    with torch.no_grad():
        dynamics = model(z0, ts)
    dynamics = dynamics.squeeze(0).cpu().numpy()

    out_dir = Path(f"inference/3dmove/{cfg['output_name']}")
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
            # 旋转坐标
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

    for t in range(T_pred - 1):
        cx, cy = traj_px[t, 0], traj_px[t, 1]
        nx, ny = traj_px[t + 1, 0], traj_px[t + 1, 1]
        dx, dy = nx - cx, ny - cy

        if chosen_shape in ["rectangle", "ellipse"]:
            scale = (s_param[t] / METER_PER_PX, l_param[t] / METER_PER_PX)
        else:
            scale = s_param[t] / METER_PER_PX

        mask = make_mask(chosen_shape, X, Y, cx, cy, scale, theta[t])
        #print(theta,"theta[t]")
        flows[t, 0, mask] = dx
        flows[t, 1, mask] = dy

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
        output_folder=f"inference/3dmove/NoiseWarp_{cfg['output_name']}",
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

    is_i2v = "I2V" in pipe_name

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

        lora_path = lora_url
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

    ## for 3D motion
    prompt_list = [
    "A small metal cube slides from the distance along a laboratory bench towards the camera, reflections visible on the surface, scattered tools in the background, captured from a fixed oblique side camera.",
    "A red rubber ball rolls from the distance along a polished wooden floor towards the camera, pulled by a thin string, scattered papers and books in the background, observed from a fixed oblique side camera.",
    "A blue glass marble moves steadily from the distance along a marble countertop towards the camera, reflections and nearby kitchen utensils visible, captured from a fixed oblique side camera.",
    "A bowling ball rolls from the distance along a polished wooden lane towards the camera, lane markings and pins in the distance, surface reflections visible, captured from a fixed oblique side camera.",
    "An airplane taxis from the distance along the runway towards the camera, airport vehicles and runway markings visible, captured from a fixed oblique side camera.",
    "A fighter jet accelerates slowly from the distance along the runway towards the camera, hangars and runway lights visible in the background, captured from a fixed oblique side camera.",
    "A yellow speedboat cruises from the distance along a calm lake towards the camera, small waves trailing behind, shoreline and distant trees visible, captured from a fixed oblique side camera.",
    "A white yacht moves steadily from the distance along the sea towards the camera, slightly wavy water surface, distant coastline and sailboats visible, observed from a fixed oblique side camera.",
    "A toy car moves from the distance along a wooden floor towards the camera, scattered blocks and a rug in the background, captured from a fixed oblique side camera.",
    "A small toy train travels from the distance along tracks on a living room floor towards the camera, bookshelves and furniture visible in the background, viewed from a fixed oblique side camera.",
    "A cardboard box slides from the distance along a warehouse floor towards the camera, shelves and crates visible in the background, captured from a fixed oblique side camera.",
    "A plastic storage container moves steadily from the distance along a smooth concrete surface in a garage towards the camera, tools and bicycles in the background, observed from a fixed oblique side camera."
    ]

    ## for acceleration

    # prompt_list = [
    #     "A red sedan accelerating in a straight line on a clean highway, the road flat and clear, with only a pale sky and distant horizon in the background, captured from a fixed roadside camera.",
    #     "A black off-road SUV accelerating in a straight line on sandy terrain, with continuous sand dunes in the background, a few white clouds in the sky, sunlight slanting, kicking up fine sand particles, viewed from a stationary side-angle camera.",
    #     "A yellow city bus accelerating in a straight line on an urban road, with simple building outlines and smooth road reflections in the background, observed from a fixed overhead traffic camera.",
    #     "A silver sedan accelerating in a straight line on a rural road, with simple green tree silhouettes and open sky in the background, captured from a fixed roadside camera.",
    #     "A blue off-road SUV accelerating on a muddy road, wheels splashing mud, background of forest and distant mountains, cloudy weather with dappled light, viewed from a fixed side camera.",
    #     "A white speedboat accelerating in a straight line on a lake, white waves trailing behind the boat, lake surface reflecting the sky and nearby trees, distant mountains in view, clear sunlight, captured from a stationary camera on the shore.",
    #     "A red long-distance bus accelerating in a straight line on a highway, with only simple road signs and clear blue sky in the background, observed from a fixed roadside camera.",
    #     "A dark gray sports car accelerating in a straight line on an empty road at dusk, wet road reflecting light, orange sunset casting colors on the car, slight wind rustling the grass along the roadside, viewed from a stationary side-angle camera.",
    #     "A white commercial jet accelerating in a straight line on the runway for takeoff, with clear runway markings and a simple blue sky in the background, captured from a fixed camera at the runway side.",
    #     "A gray fighter jet accelerating in a straight line along a military runway, exhaust flames visible at the tail, with only sky and ground in the background, viewed from a stationary side-angle camera.",
    #     "A green four-wheel-drive SUV accelerating on a rugged mountain road, tires kicking up small dust and pebbles, with steep slopes and sparse vegetation in the background, captured from a fixed roadside camera.",
    #     "A yellow small speedboat accelerating in a straight line on the sea, long trails of water behind the boat, slightly wavy surrounding sea, distant blurred coastline and docked sailboats visible, sunlight reflecting off the water, viewed from a stationary camera on a pier."
    # ]

    ## for Deceleration
    # prompt_list = [
    # "A red sedan brakes and decelerates in a straight line on a wet city street, with lights reflecting on the road and buildings in the background, captured by a fixed side-view camera.",
    # "A silver sports car decelerates in a straight line on a countryside road at dusk, with the wet road reflecting the orange sunset, trees and farmland along the roadside, captured by a fixed side-view camera.",
    # "A yellow bus decelerates in a straight line in front of a traffic light on a city street, with pedestrians crossing nearby, and the wet road reflecting the sky, captured by a fixed side-view camera.",
    # "A red coach brakes and decelerates in a straight line on a highway, with road signs and streetlights nearby and the city skyline visible in the distance, captured by a fixed side-view camera.",
    # "A blue pickup truck decelerates in a straight line on a countryside road, with farmland and trees on the roadside, leaves swaying in the breeze, and sunlight shining from the side, captured by a fixed side-view camera.",
    # "A white pickup truck decelerates in a straight line on a gravel road, with dust rising behind it and slowly settling, hills and sparse vegetation in the background, captured by a fixed side-view camera.",
    # "A white speedboat decelerates in a straight line on a lake, with waves behind the boat gradually calming, distant shoreline and mountains visible, under clear sunlight, captured by a fixed side-view camera.",
    # "A yellow small speedboat brakes and decelerates in a straight line on the sea, leaving a long wake that gradually smooths out, with gentle ripples on the water, and a distant blurred coastline and anchored sailboats, captured by a fixed side-view camera.",
    # "A commercial jet airliner decelerates in a straight line on the runway after landing, with runway lights visible, captured by a fixed side-view camera.",
    # "A fighter jet brakes and decelerates in a straight line on a military runway, with mountains and hangars visible in the distance, captured by a fixed side-view camera.",
    # "A red bowling ball rolls in a straight line on a wooden lane and gradually decelerates, slowing down before reaching the pins, with the polished lane surface reflecting light, captured by a fixed side-view camera.",
    # "A blue bowling ball decelerates in a straight line on a glossy lane, spinning slightly, with lane markings and pins visible in the distance, captured by a fixed side-view camera."
    # ]


    ## Parabolic Motion
    # prompt_list = [
    # "A single apple is thrown at an angle with an initial speed. The camera captures the motion from the side, showing the apple rising, reaching its peak, and then descending under gravity. The scene takes place in a bright open field under a clear blue sky, with soft sunlight casting gentle shadows on the ground. The background shows green grass and distant trees, adding depth and realism.",
    # "A single coconut is thrown at an angle with an initial speed. The camera captures the motion from the side, showing the coconut rising, reaching its peak, and then descending under gravity. The scene takes place on a bright open beach under a clear blue sky. The background shows white sand and the sea, adding depth and realism.",
    # "A soccer ball is kicked at an angle with an initial speed. The camera captures the motion from the side, showing the ball rising, reaching its peak, and descending under gravity. The scene takes place on a sunlit football field with green grass, white boundary lines, and distant goalposts visible in the background, adding depth and realism.",
    # "A basketball is thrown at an angle towards a hoop. The camera captures the side view of the ball rising, reaching its apex, and descending towards the basket. The scene takes place in an indoor gym with polished wooden floors, overhead lights reflecting on the court, and bleachers in the background.",
    # "An orange is tossed at an angle with an initial speed. The camera captures the motion from the side, showing it rising and then falling under gravity. The scene is set in a sunny backyard with a wooden fence and flowerbeds in the background.",
    # "A tennis ball is thrown at an angle, captured from a fixed side camera, showing the parabolic trajectory. The scene takes place on an outdoor tennis court with net, lines, and surrounding trees visible in the distance under bright sunlight.",
    # "A baseball is thrown at an angle with an initial speed. The camera captures its flight from the side, rising and then descending. The scene is set on a baseball field, with dirt infield and green outfield grass, and stadium seats faintly visible in the background.",
    # "An American football is kicked at an angle with initial speed. The camera captures the side view, showing the ball rising and descending along a parabolic trajectory. The scene is set on a sunny football field with goalposts in the distance.",
    # "A rugby ball is thrown at an angle, captured from a stationary side camera. The motion shows it rising, reaching its peak, and descending. The scene takes place in a park with green grass, scattered trees, and people faintly visible in the background.",
    # "A volleyball is served at an angle, captured from the side by a stationary camera. The scene is set on an outdoor beach volleyball court, with sand texture, net, and distant palm trees in view.",
    # "A watermelon is thrown at an angle in a park, captured from a fixed side camera, rising and descending along a parabolic path. The scene shows green grass, a few park benches, and trees in the background under bright daylight.",
    # "A golf ball is hit at an angle with an initial speed. The camera captures its parabolic trajectory from the side. The scene takes place on a sunny golf course with manicured fairways, sand bunkers, and distant trees, adding depth and realism."
    # ]


    # ## for Slope Sliding
    # prompt_list = [
    # "A small metal cube sliding down a laboratory ramp, shiny reflections on its surface, scattered tools and wires in the background, captured from a fixed side camera parallel to the ramp.",
    # "A metal block sliding quickly down an inclined steel ramp in a laboratory, reflections on the shiny surface, equipment and cables in the background, captured from a fixed side camera parallel to the ramp.",
    # "A hardcover book accelerating down a carpeted inclined board in a classroom, chalkboard and desks in the background, captured from a fixed side camera parallel to the ramp.",
    # "A cylindrical tin can rolling down a slightly tilted wooden ramp, small dents and reflections visible, classroom objects in the background, captured from a fixed side camera parallel to the ramp.",
    # "A glass bottle accelerating down a tiled inclined surface, reflections on the bottle and floor, with a kitchen counter and utensils in the background, captured from a fixed side camera parallel to the ramp.",
    # "A plastic storage box sliding down a metal ramp in a warehouse, shelves and stacked boxes in the background, captured from a fixed side camera parallel to the ramp.",
    # "A stack of notebooks accelerating down a wooden ramp in a study room, bookshelves and a table lamp visible in the background, captured from a fixed side camera parallel to the ramp.",
    # "A smartphone sliding quickly down a glossy inclined table surface, reflections from a nearby lamp, computer monitor and cables in the background, captured from a fixed side camera parallel to the ramp.",
    # "A ceramic mug accelerating down a wooden inclined board, kitchen tiles and shelves in the background, natural daylight streaming through a window, captured from a fixed side camera parallel to the ramp.",
    # "A plastic bottle sliding down a short inclined ramp outdoors, grass and gravel visible around, sunlight casting shadows, captured from a fixed side camera parallel to the ramp.",
    # "A thick physics textbook accelerating down a polished wooden slope in a library corner, shelves filled with books behind, sunlight casting stripes through the window blinds, captured from a fixed side camera parallel to the ramp.",
    # "A cardboard box sliding down a metal chute outdoors, patches of grass and concrete ground visible at the bottom, cloudy sky in the background, captured from a fixed side camera parallel to the ramp."
    # ]
    #
    #
    # ## for Uniform
    # prompt_list = [
    # "A small metal cube sliding steadily along a smooth laboratory bench, reflections visible on the surface, scattered tools in the background, captured from a fixed side camera.",
    # "A red rubber ball rolling at constant speed on a polished wooden floor, pulled by a thin string, with scattered papers and books in the background, observed from a fixed side camera.",
    # "A blue glass marble moving at steady speed along a marble countertop, reflections and nearby kitchen utensils visible, captured from a fixed side camera.",
    # "A bowling ball moving steadily down a polished wooden lane, lane markings and pins in the distance, reflections on the surface, observed from a stationary camera above the lane.",
    # "An airplane taxiing steadily along the runway, airport vehicles and markings visible, captured from a fixed side camera near the tarmac.",
    # "A fighter jet accelerating slowly along the runway before takeoff, hangars and runway lights in the background, captured from a fixed side camera.",
    # "A yellow speedboat cruising at constant speed on a calm lake, small waves trailing behind, shoreline and distant trees visible, captured from a fixed camera on the shore.",
    # "A white yacht moving at steady speed along the sea, slightly wavy water surface, distant coastline and sailboats in view, observed from a stationary pier camera.",
    # "A toy car moving at constant speed on a wooden floor, scattered blocks and a rug in the background, captured from a fixed side camera at table height.",
    # "A small toy train traveling steadily along tracks on a living room floor, bookshelves and furniture visible in the background, viewed from a stationary side-angle camera.",
    # "A cardboard box sliding at constant speed on a warehouse floor, shelves and crates visible in the background, captured from a fixed side camera.",
    # "A plastic storage container moving steadily on a smooth concrete surface in a garage, tools and bicycles in the background, observed from a stationary side-angle camera."
    # ]
    #


    outputs = []

    for cfg in config_list:
        sample_path = f"inference/3dmove/NoiseWarp_{cfg['output_name']}" 

        for i, prompt in enumerate(prompt_list):
            output_mp4_path = f"inference/3dmove/{cfg['output_name']}_prompt{i+1}.mp4"

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



