# Modified from https://github.com/dvlab-research/MGM/blob/main/mgm/serve/cli.py
import os
from PIL import Image
import sys

from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import argparse
import torch

try:
    from diffusers import StableDiffusionXLPipeline
except:
    print('please install diffusers==0.26.3')

try:
    from paddleocr import PaddleOCR
except:
    print('please install paddleocr following https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md')


from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class MGM(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Model
        disable_torch_init()
        
        if cfg.OCR:
            self.ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="ch")
    
        # initialize the model instance
        self.model_name = get_model_name_from_path(cfg.MODEL_PATH)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            cfg.MODEL_PATH, cfg.MODEL_BASE, self.model_name, cfg.LOAD_8BIT, cfg.LOAD_4BIT,
            device='cuda')

        self.temperature = cfg.TEMPERATURE
        self.max_new_tokens = cfg.MAX_NEW_TOKENS

        if '8x7b' in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif '34b' in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif '2b' in self.model_name.lower():
            self.conv_mode = "gemma"
        else:
            self.conv_mode = "vicuna_v1"

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
        
        prompt = 'Please choose the most appropriate answer from the given choices based on the image and the question.\n' + prompt + 'No explanatin and other information should be answered.'

        # save to file
        if hasattr(self, 'ocr'):
            image.save('temp.png')
            result = self.ocr.ocr('temp.png')
            str_in_image = ''
            if result[0] is not None:
                result = [res[1][0] for res in result[0] if res[1][1] > 0.1]
                if len(result) > 0:
                    str_in_image = ', '.join(result)
                    print('OCR Token: ' + str_in_image)

        # initialize
        image = image.convert('RGB')
        if hasattr(self.model.config, 'image_size_aux'):
            if not hasattr(self.image_processor, 'image_size_raw'):
                self.image_processor.image_size_raw = self.image_processor.crop_size.copy()
            self.image_processor.crop_size['height'] = self.model.config.image_size_aux
            self.image_processor.crop_size['width'] = self.model.config.image_size_aux
            self.image_processor.size['shortest_edge'] = self.model.config.image_size_aux

        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, self.model.config)
    
        image_grid = getattr(self.model.config, 'image_grid', 1)
        if hasattr(self.model.config, 'image_size_aux'):
            raw_shape = [self.image_processor.image_size_raw['height'] * image_grid,
                        self.image_processor.image_size_raw['width'] * image_grid]
            image_tensor_aux = image_tensor 
            image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                        size=raw_shape,
                                                        mode='bilinear',
                                                        align_corners=False)
        else:
            image_tensor_aux = []


        if image_grid >= 2:            
            raw_image = image_tensor.reshape(3, 
                                            image_grid,
                                            self.image_processor.image_size_raw['height'],
                                            image_grid,
                                            self.image_processor.image_size_raw['width'])
            raw_image = raw_image.permute(1, 3, 0, 2, 4)
            raw_image = raw_image.reshape(-1, 3,
                                        self.image_processor.image_size_raw['height'],
                                        self.image_processor.image_size_raw['width'])
                    
            if getattr(self.model.config, 'image_global', False):
                global_image = image_tensor
                if len(global_image.shape) == 3:
                    global_image = global_image[None]
                global_image = torch.nn.functional.interpolate(global_image, 
                                                            size=[self.image_processor.image_size_raw['height'],
                                                                    self.image_processor.image_size_raw['width']], 
                                                            mode='bilinear', 
                                                            align_corners=False)
                # [image_crops, image_global]
                raw_image = torch.cat([raw_image, global_image], dim=0)
            image_tensor = raw_image.contiguous()
            image_tensor = image_tensor.unsqueeze(0)
    
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
            image_tensor_aux = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor_aux]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            image_tensor_aux = image_tensor_aux.to(self.model.device, dtype=torch.float16)


        conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            self.roles = ('user', 'assistant')
        else:
            self.roles = conv.roles

        inp = f"{self.roles[0]}: {prompt}"
        
        if hasattr(self, 'ocr') and len(str_in_image) > 0:
            inp = inp + '\nReference OCR Token: ' + str_in_image + '\n'

        # print(f"{self.roles[1]}: ", end="")

        # first message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                bos_token_id=self.tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.pad_token_id,  # Pad token
                streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        conv.messages[-1][-1] = outputs

        if self.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
        return outputs


if __name__ == '__main__':
    import argparse
    import easydict
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/path/place_centric_data/place_centric_images/ChIJ_____-Bn1moRWGI1BpiLeN8.jpg')
    parser.add_argument('--question', type=str, default='Q: Which human intentions can be accomplished here? Choices: A. Buying furniture for home improvement. B. Participating in a yoga session for fitness. C. Having a quick meal or dining with family or friends. D. Attending a music concert.')
    args = parser.parse_args()

    cfg = argparse.Namespace()
    cfg = {
        'MODEL_PATH': '/home/deng/ry/repo/MGM/MGM-7B',
        'MODEL_BASE': '/home/deng/ry/repo/MGM/MGM-7B',
        'LOAD_8BIT': False,
        'LOAD_4BIT': False,
        'CONV_MODE': None,
        'OCR': False,
        'TEMPERATURE': 0.2,
        'MAX_NEW_TOKENS': 512
    }
    cfg = easydict.EasyDict(cfg)

    mgm = MGM(cfg)

    image = Image.open(args.image_path)
    question = args.question
    output = mgm.predict(image, question)
    output = mgm.predict(image, question)
    print(output)
