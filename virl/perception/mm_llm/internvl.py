import os
from PIL import Image
import sys
import argparse

from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate



def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        path = cfg.MODEL_PATH

        # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
    
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # set the max number of tiles in `max_num`

        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )
        
        self.debug = False

    def predict(self, image: Image, prompt: str) -> str:
        """
        image: PIL.Image
        prompt: str
        """

        pixel_values = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
        response = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)

        if self.debug:
            print('prompt:', prompt)
            print('response:', response)
        return response


if __name__ == '__main__':
    import argparse
    import easydict
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/path/place_centric_data/place_centric_images/ChIJ_____-Bn1moRWGI1BpiLeN8.jpg')
    parser.add_argument('--question', type=str, default='Q: Which human intentions can be accomplished here? Choices: A. Buying furniture for home improvement. B. Participating in a yoga session for fitness. C. Having a quick meal or dining with family or friends. D. Attending a music concert.')
    args = parser.parse_args()

    cfg = argparse.Namespace()
    cfg = {
        'MODEL_PATH': "OpenGVLab/InternVL-Chat-V1-5",
    }
    cfg = easydict.EasyDict(cfg)

    internvl = InternVL(cfg)

    image = Image.open(args.image_path)
    question = args.question
    output = internvl.predict(image, question)
    output = internvl.predict(image, question)
    print(output)
