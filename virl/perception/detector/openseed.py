import sys
import yaml

sys.path.insert(0, '/home/ryding/OpenSeeD/')

from PIL import Image
import numpy as np
np.random.seed(2)

import torch
from torchvision import transforms

try:
    from detectron2.data import MetadataCatalog
    from detectron2.structures import BitMasks
    from openseed.BaseModel import BaseModel
    from openseed import build_model
    from detectron2.utils.colormap import random_color
except:
    pass

from virl.utils import common_utils


def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v


def load_opt_from_config_files(conf_files):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files (list): a list of config file paths

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    for conf_file in conf_files:
        with open(conf_file, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        load_config_dict_to_opt(opt, config_dict)

    return opt


class OpenSeeD(object):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.THRESH = cfg.THRESH

        # META DATA
        pretrained_pth = cfg.WEIGHT
        opt = load_opt_from_config_files(cfg.CFG_FILE)
        self.model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

        t = []
        t.append(transforms.Resize(800, interpolation=Image.BICUBIC))
        self.transform = transforms.Compose(t)

    def inference(self, image_ori, caption, _, need_draw=False):
        thing_classes = caption.split(',')
        thing_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(thing_classes))]
        thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}

        if MetadataCatalog.get("demo").get("thing_colors") is None:
            MetadataCatalog.get("demo").set(
                thing_colors=thing_colors,
                thing_classes=thing_classes,
                thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            )
            # model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + ["background"], is_eval=False)
            self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes, is_eval=True)
            metadata = MetadataCatalog.get('demo')
            self.model.model.metadata = metadata
            self.model.model.sem_seg_head.num_classes = len(thing_classes)

        with torch.no_grad():
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = self.transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = self.model.forward(batch_inputs)
            # visual = Visualizer(image_ori, metadata=metadata)

        boxes = outputs[-1]['instances'].pred_boxes.tensor.cpu().numpy()
        scores = outputs[-1]['instances'].scores.cpu().numpy()
        class_index = outputs[-1]['instances'].pred_classes.cpu().numpy()
        labels = np.array(thing_classes)[class_index]

        # exclude boxes with low scores
        keep = np.where(scores > self.THRESH)[0]
        answers = {
            'boxes': boxes[keep], 'scores': scores[keep], 'labels': labels[keep], 'class_idx': class_index[keep]
        }

        annotated_frame = common_utils.draw_with_results(np.asarray(image), answers)
        if need_draw:
            Image.fromarray(annotated_frame).save('draw.png', 'PNG')

        return answers, annotated_frame


if __name__ == "__main__":
    from easydict import EasyDict as edict
    cfg = edict({'CFG_FILE': ['/xxx/OpenSeeD/configs/openseed/openseed_swint_lang.yaml'],
                 'WEIGHT': '/xxx/OpenSeeD/model_state_dict_swint_51.2ap.pt'})
    osd = OpenSeeD(cfg)
    img = Image.open('/xxx/streetview_hotel.jpeg')
    prompt = 'bank,restaurant,supermarket,bakery,cafe,pharmacy,hospital,spa,convenience store,school,library,park,lodging,laundry,movie theater,book store,clothing store,jewelry store,gym,bar'
    answers, _ = osd.inference(
        img, prompt, None, need_draw=True)
    print(answers)
