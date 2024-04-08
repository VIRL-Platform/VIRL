from importlib import import_module

from virl.utils import common_utils


class MultiModalLLM(object):
    def __init__(self, model_cfg, name):
        self.model_cfg = model_cfg

        self.model = self.create_mm_llm(name)

    def create_mm_llm(self, name):
        # Mapping of model names to their respective class imports
        model_mapping = {
            'InstructBLIP': ('.instructblip_client', 'InstructBLIPClient'),
            'Otter': ('.otter', 'Otter'),
            'LLaVA': ('.llava_client', 'LLaVA'),
            'mPLUG_Owl': ('.mplug_owl_client', 'mPLUG_Owl'),
            'internLLM_XComposer': ('.internllm_xcomposer_client', 'InternLLM_XComposer'),
            'BLIP2': ('.blip2_local', 'BLIP2'),
            'InstructBLIPLocal': ('.instructblip_local', 'InstructBLIPLocal'),
            'MiniGPT4': ('.minigpt4_client', 'MiniGPT4Client'),
            'Shikra': ('.shikra_client', 'ShikraClient'),
            'GPT4V': ('.gpt4v', 'GPT4V'),
        }

        if name in model_mapping:
            module_name, class_name = model_mapping[name]
            module = import_module(module_name, package=__package__)
            model_class = getattr(module, class_name)
            model = model_class(getattr(self.model_cfg, name))
        else:
            model = None

        return model

    def check(self, image, question, return_json=False):
        print(f'>>> Check with multi-modal language model.')
        common_utils.print_prompt(question)
        answer = self.model.ask(image, question, return_json=return_json)
        common_utils.print_answer(answer)

        return answer
