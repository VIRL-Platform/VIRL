# modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/cli.py
from PIL import Image
import numpy as np
from transformers import TextStreamer

import argparse
import torch

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate
from virl.config import cfg

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


def int2char(ii):
    return chr(ii + 1 + 64)


def char2int(ii):
    return ord(ii) - 64 - 1


class LLaVAClient(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)

        # initialize the otter instance
        self.model_name = get_model_name_from_path(cfg.MODEL_PATH)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            cfg.MODEL_PATH, cfg.MODEL_BASE, self.model_name, cfg.LOAD_8BIT, cfg.LOAD_4BIT,
            device='cuda')

        # self.temperature = cfg.TEMPERATURE
        # self.max_new_tokens = cfg.MAX_NEW_TOKENS

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

    def form_prompt(self, candidate_answers):
        question = "Q: Which of the following landmarks would be the most accurately identified in the image with a high degree of confidence?"
        choices = "Choices: "
        for ii, aa in enumerate(candidate_answers):
            choices += f'{int2char(ii)}. {aa}; '
        choices += f'{int2char(len(candidate_answers))}. None of the above.'
        return question + ' ' + choices

    def inference(self, image: Image, candidate_answers: str) -> str:
        """
        image: PIL.Image
        candidate_answers: str
        """
        # initialize
        candidate_answers_list = candidate_answers.split(',,')
        image = image.convert('RGB')
        conv = conv_templates[self.conv_mode].copy()

        # Similar operation in model_worker.py
        image_tensor = process_images([image], self.image_processor, {'image-aspect-ratio': 'pad'})
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        # print(image_tensor)
        question = self.form_prompt(candidate_answers_list)
        # question = 'What is the landmark in the image? '

        question_inp = f"{conv.roles[0]}: {question}"
        if self.model.config.mm_use_im_start_end:
            question_inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question_inp
        else:
            question_inp = DEFAULT_IMAGE_TOKEN + '\n' + question_inp
        # inp = f"{conv.roles[0]}: {question}"
        conv.append_message(conv.roles[0], question_inp)
        conv.append_message(conv.roles[1], None)
        question_prompt = conv.get_prompt()
        question_input_ids = tokenizer_image_token(question_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            question_output = self.model(
                question_input_ids,
                images=image_tensor,
                use_cache=True)
        question_logits = question_output.logits
        question_past_key_values = question_output.past_key_values

        loss_list = []
        for ii, option in enumerate(candidate_answers_list + ['None of the above.']):
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], question_inp)
            conv.append_message(conv.roles[1], int2char(ii) + '. ' + option)
            full_prompt = conv.get_prompt()
            full_input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]
            with torch.inference_mode():
                output_option = self.model(input_ids=option_answer_input_ids,
                                           use_cache=True,
                                           attention_mask=torch.ones(1, question_logits.shape[1] + option_answer_input_ids.shape[1], device=full_input_ids.device),
                                           past_key_values=question_past_key_values)
            # print(question_prompt, '\n', full_prompt)
            logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)
            loss_fct = torch.nn.CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.vocab_size)
            labels = option_answer_input_ids.view(-1)
            loss = loss_fct(logits, labels)
            loss_list.append(loss)
        option_chosen = torch.stack(loss_list).argmin()
        # print(loss_list)

        scores = np.zeros(len(candidate_answers_list))
        if option_chosen == len(scores):
            pass
        else:
            for ii in range(len(scores)):
                scores[ii] = 1. / (loss_list[ii].cpu().item() + 1e-6)
        print(loss_list)
        return {'scores': scores}


if __name__ == '__main__':
    import argparse
    import easydict
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='xxx.jpg')
    parser.add_argument('--candidate_answers', type=str, default="republic bank,,wendy's,,xian famous food")
    args = parser.parse_args()

    cfg.update({
        'MODEL_PATH': '/xxx/LLaVA/llava-v1.5-13b/',
        'MODEL_BASE': None,
        'LOAD_8BIT': False,
        'LOAD_4BIT': False,
        'CONV_MODE': None,
    })

    llava = LLaVAClient(cfg)

    image = Image.open(args.image_path)

    path = 'xxx.jpg'
    image = Image.open(path)
    output = llava.inference(image, args.candidate_answers)
    print(output)
    output = llava.inference(image, "republic bank,,xian famous food")
    print(output)
