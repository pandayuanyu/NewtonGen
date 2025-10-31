import requests
from rp import (
    encode_image_to_base64,
    gather_args_call,
    is_image,
    load_files,
    replace_if_none,
)


def _get_gpt4v_request_json(image, text, max_tokens):
    base64_image = encode_image_to_base64(image)
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # "text": "What is in this image?"
                        "text": text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": max_tokens,
    }


def _run_gpt4v(image, text, max_tokens, api_key):
    """Processes a single image"""

    #Fill in your own API key here
    default_key = "sk-proj-Z8CO........................................1Ldn"
    api_key = replace_if_none(default_key)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    request_json = gather_args_call(_get_gpt4v_request_json)

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=request_json,
    )

    return response.json()["choices"][0]["message"]["content"]


def run_gpt4v(images, text="", max_tokens=2, api_key=None):
    """
    Asks GPT4V a question about an image, returning a string.
    If given multiple images, will process them in parallel lazily (retuning a generator instead)

    Args:
        image (str or image):
            Can be passed as a single image, or a list of images
            Images can be specified as file paths, urls, numpy arrays, PIL images, or (H,W,3) torch images
        text (str, optional): The question we ask GPT4V
        max_tokens (int, optional): Maximum tokens in the response
        api_key (str, optional): If specified, overwrites the default openai api_key

    Returns:
        (str or generator): GPT4V's response (or a lazy generator of responses if given a list of images)

    Single Image Example:
        >>> print(
                run_gpt4v(
                    "https://cdn.britannica.com/92/212692-050-D53981F5/labradoodle-dog-stick-running-grass.jpg",
                    "What is this?",
                    max_tokens=20,
                )
            )
        This is a photograph of a happy-looking dog, likely a poodle or poodle mix, caught

    Single Image Example (with cropping):
        >>> #This example shows how you can pass in any type of image, not just strings to urls or paths
        >>> from rp import load_image, crop_image, display_image, np
        >>> image = "https://cdn.britannica.com/92/212692-050-D53981F5/labradoodle-dog-stick-running-grass.jpg"
        >>> image = load_image(image)
        >>> image = crop_image(image,height=300,width=500,origin='center')
        >>> display_image(image) # See what we're feeding GPT4V
        >>> assert isinstance(image, np.ndarray)
        >>> print(
                run_gpt4v(
                    image,
                    "What is this?",
                    max_tokens=20,
                )
            )
        This image shows a close-up of a fluffy dog carrying a stick in its mouth. The dog appears

    Multiple Images Example:
        >>> images = [
                "https://cdn.britannica.com/92/212692-050-D53981F5/labradoodle-dog-stick-running-grass.jpg",
                "https://cdn.britannica.com/39/7139-050-A88818BB/Himalayan-chocolate-point.jpg",
                "https://cdn.britannica.com/42/150642-138-2F8611E1/Erik-Gregersen-astronomy-astronaut-Encyclopaedia-Britannica-space.jpg",
                "https://cdn.britannica.com/99/187399-050-8C81D8D4/cedar-tree-regions-Lebanon-Mediterranean-Sea.jpg",
            ]
            for response in run_gpt4v(images,'What is this a picture of? One word only.'):
                #Will print responses as they come, all running in parallel.
                print(response)
        Dog.
        Cat.
        Astronaut.
        Tree.
    """

    if is_image(images) or isinstance(images, str):
        # The case where we give it a single image
        return gather_args_call(_run_gpt4v, images)

    return load_files(
        gather_args_bind(_run_gpt4v),
        images,
        show_progress=True,
        num_threads=10,
        lazy=True,
    )

