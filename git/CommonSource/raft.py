import rp
import torch
import torchvision.transforms
from torchvision.models.optical_flow import raft_large, raft_small

class RaftOpticalFlow(rp.CachedInstances):
    def __init__(self, device, version='large'):
        """
        Automatically downloads the model you select upon instantiation if not already downloaded
        """

        models = {
            'large' : raft_large,
            'small' : raft_small,
        }
        assert version in models
        model = models[version]
        model = model(pretrained=True, progress=False).to(device)
        model.eval()

        self.version = version
        self.device = device
        self.model = model

    def _preprocess_image(self, image):
        assert rp.is_image(image) or rp.is_torch_image(image), type(image)
        
        if rp.is_image(image):
            image = rp.as_float_image(rp.as_rgb_image(image))
            image = rp.as_torch_image(image)

        image = image.to(self.device)
        image = image.float()

        #Floor height and width to the nearest multpiple of 8
        height, width = rp.get_image_dimensions(image)
        new_height = (height // 8) * 8
        new_width  = (width  // 8) * 8

        #Resize the image
        image = rp.torch_resize_image(image, (new_height, new_width), copy=False)

        #Map [0, 1] to [-1, 1]
        image = image * 2 - 1

        #CHW --> 1CHW
        output = image[None]

        assert rp.is_torch_tensor(output)
        assert output.shape == (1, 3, new_height, new_width)

        return output
    
    def __call__(self, from_image, to_image):
        """
        Calculates the optical flow from from_image to to_image, returned in 2HW form
        In other words, returns (dx, dy) where dx and dy are both HW torch matrices with the same height and width as the input image

        Works best when the image's dimensions are multiple of 8 pixels
        Works fastest when passed torch images on the same device as this model

        Args:
            from_image: Can be an image as defined by rp.is_image, or an RGB torch image (a 3HW torch tensor)
            to_image  : Can be an image as defined by rp.is_image, or an RGB torch image (a 3HW torch tensor)
        """
        assert rp.is_image(from_image)
        assert rp.is_image(to_image)
        assert rp.get_image_dimensions(from_image) == rp.get_image_dimensions(to_image)
        
        height, width = rp.get_image_dimensions(from_image)
        
        with torch.no_grad():
            img1 = self._preprocess_image(from_image)
            img2 = self._preprocess_image(to_image  )
            
            list_of_flows = self.model(img1, img2)
            output_flow = list_of_flows[-1][0]
    
            # Resize the predicted flow back to the original image size
            resize = torchvision.transforms.Resize((height, width))
            output_flow = resize(output_flow[None])[0]

        assert rp.is_torch_tensor(output_flow)
        assert output_flow.shape == (2, height, width)

        return output_flow
