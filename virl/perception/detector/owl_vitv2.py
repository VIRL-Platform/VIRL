from virl.utils import common_utils, vis_utils

from .owl_vit import OWLVIT


class OWLVITv2(OWLVIT):
    def __init__(self, cfg, **kwargs):
        self.model_name = cfg.MODEL_NAME
        
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        self.processor = Owlv2Processor.from_pretrained(self.model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name).cuda()

