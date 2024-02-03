from virl.utils import common_utils


class MultiModalLLM(object):
    def __init__(self, model_cfg, name):
        self.model_cfg = model_cfg

        self.model = self.create_mm_llm(name)

    def create_mm_llm(self, name):
        mm_llm_cfg = getattr(self.model_cfg, name)
        if name == 'InstructBLIP':
            from .instructblip_client import InstructBLIPClient
            model = InstructBLIPClient(mm_llm_cfg)
        elif name == 'Otter':
            from .otter import Otter
            model = Otter(mm_llm_cfg)
        elif name == 'LLaVA':
            from .llava_client import LLaVA
            model = LLaVA(mm_llm_cfg)
        elif name == 'mPLUG_Owl':
            from .mplug_owl_client import mPLUG_Owl
            model = mPLUG_Owl(mm_llm_cfg)
        elif name == 'internLLM_XComposer':
            from .internllm_xcomposer_client import InternLLM_XComposer
            model = InternLLM_XComposer(mm_llm_cfg)
        elif name == 'BLIP2':
            from .blip2_local import BLIP2
            model = BLIP2(mm_llm_cfg)
        elif name == 'InstructBLIPLocal':
            from .instructblip_local import InstructBLIPLocal
            model = InstructBLIPLocal(mm_llm_cfg)
        elif name == 'MiniGPT4':
            from .minigpt4_client import MiniGPT4Client
            model = MiniGPT4Client(mm_llm_cfg)
        elif name == 'Shikra':
            from .shikra_client import ShikraClient
            model = ShikraClient(mm_llm_cfg)
        else:
            model = None

        return model

    def check(self, image, question, return_json=False):
        print(f'>>> Check with multi-modal language model.')
        common_utils.print_prompt(question)
        answer = self.model.ask(image, question, return_json=return_json)
        common_utils.print_answer(answer)

        return answer
