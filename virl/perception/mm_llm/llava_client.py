# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py
from PIL import Image
from transformers import TextStreamer

import torch

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class LLaVA(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)

        # initialize the otter instance
        self.model_name = get_model_name_from_path(cfg.MODEL_PATH)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            cfg.MODEL_PATH, cfg.MODEL_BASE, self.model_name, cfg.LOAD_8BIT, cfg.LOAD_4BIT,
            device='cuda')

        self.temperature = cfg.TEMPERATURE
        self.max_new_tokens = cfg.MAX_NEW_TOKENS

        if 'llama-2' in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if cfg.CONV_MODE is not None and self.conv_mode != cfg.CONV_MODE:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                self.conv_mode, cfg.CONV_MODE, cfg.CONV_MODE))
        else:
            cfg.CONV_MODE = self.conv_mode

        self.debug = False

    def predict(self, image: Image, prompt: str) -> str:
        """
        image: PIL.Image
        question: str
        """
        # initialize
        image = image.convert('RGB')
        self.conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = self.conv.roles

        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, {'image-aspect-ratio': 'pad'})
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inp = f"{self.roles[0]}: {prompt}"

        # print(f"{self.roles[1]}: ", end="")

        # first message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)

        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        self.conv.messages[-1][-1] = outputs

        if self.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        return outputs


if __name__ == '__main__':
    import argparse
    import easydict
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='xxx.jpg')
    parser.add_argument('--question', type=str, default='Q: Which human intentions can be accomplished here? Choices: A. Buying furniture for home improvement. B. Participating in a yoga session for fitness. C. Having a quick meal or dining with family or friends. D. Attending a music concert.')
    args = parser.parse_args()

    cfg = argparse.Namespace()
    cfg = {
        'MODEL_PATH': '/xxx/LLaVA/llava-v1.5-7b',
        'MODEL_BASE': None,
        'LOAD_8BIT': False,
        'LOAD_4BIT': False,
        'CONV_MODE': None,
        'TEMPERATURE': 0.2,
        'MAX_NEW_TOKENS': 512
    }
    cfg = easydict.EasyDict(cfg)

    llava = LLaVA(cfg)

    image = Image.open(args.image_path)
    question = args.question
    output = llava.predict(image, question)
    output = llava.predict(image, question)
    print(output)
