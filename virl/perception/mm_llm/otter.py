from typing import Union
import mimetypes
import requests
import os
from PIL import Image
import sys
import transformers

import torch

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate


class Otter(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)

        # initialize the otter instance
        sys.path.append(os.path.join(cfg.OTTER_PATH, 'src'))
        from otter_ai import OtterForConditionalGeneration
        load_bit = "bf16"
        precision = {}
        if load_bit == "bf16":
            precision["torch_dtype"] = torch.bfloat16
        elif load_bit == "fp16":
            precision["torch_dtype"] = torch.float16
        elif load_bit == "fp32":
            precision["torch_dtype"] = torch.float32
        self.model = OtterForConditionalGeneration.from_pretrained(
            "luodian/OTTER-Image-MPT7B", device_map="sequential", **precision)
        self.model.text_tokenizer.padding_side = "left"
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = transformers.CLIPImageProcessor()
        self.model.eval()

    def get_formatted_prompt(self, prompt: str) -> str:
        return f"<image>User: {prompt} GPT:<answer>"

    def predict(self, image, prompt: str) -> str:
        """
        image: PIL.Image
        question: str
        """
        input_data = image

        if isinstance(input_data, Image.Image):
            if input_data.size == (224, 224) and not any(input_data.getdata()):  # Check if image is blank 224x224 image
                vision_x = torch.zeros(1, 1, 1, 3, 224, 224, dtype=next(self.model.parameters()).dtype)
            else:
                vision_x = self.image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        else:
            raise ValueError("Invalid input data. Expected PIL Image.")

        lang_x = self.model.text_tokenizer(
            [
                self.get_formatted_prompt(prompt),
            ],
            return_tensors="pt",
        )

        model_dtype = next(self.model.parameters()).dtype

        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]

        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x_input_ids.to(self.model.device),
            attention_mask=lang_x_attention_mask.to(self.model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3,
        )
        parsed_output = (
            self.model.text_tokenizer.decode(generated_text[0])
            .split("<answer>")[-1]
            .lstrip()
            .rstrip()
            .split("<|endofchunk|>")[0]
            .lstrip()
            .rstrip()
            .lstrip('"')
            .rstrip('"')
        )
        return parsed_output


def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


def get_image(url: str) -> Union[Image.Image, list]:
    if not url.strip():  # Blank input, return a blank Image
        return Image.new("RGB", (224, 224))  # Assuming 224x224 is the default size for the model. Adjust if needed.
    elif "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    else:
        raise ValueError("Invalid content type. Expected image.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='xxx.jpg')
    parser.add_argument('--question', type=str, default='Q: Which human intentions can be accomplished here? Choices: A. Buying furniture for home improvement. B. Participating in a yoga session for fitness. C. Having a quick meal or dining with family or friends. D. Attending a music concert.')
    args = parser.parse_args()

    cfg = argparse.Namespace()
    cfg.OTTER_PATH = '/home/deng/jihan/Otter'

    otter = Otter(cfg)

    image = Image.open(args.image_path)
    question = args.question
    output = otter.predict(image, question)
    print(output)
