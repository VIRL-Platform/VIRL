from PIL import Image
import torch
import numpy as np

from transformers import OwlViTProcessor, OwlViTForObjectDetection

from virl.utils import common_utils, vis_utils


class OWLVIT(object):
    def __init__(self, cfg, **kwargs):
        self.model_name = cfg.MODEL_NAME

        self.processor = OwlViTProcessor.from_pretrained(self.model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(self.model_name).cuda().eval()

    def inference(self, img, caption, score_thresh=0.7, need_draw=False):
        image = img.convert('RGB')
        texts = [caption.split(',')]

        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        for _input in inputs:
            inputs[_input] = inputs[_input].cuda()
        with torch.no_grad():
            outputs = self.model(**inputs)
        for _output in outputs:
            if isinstance(outputs[_output], torch.Tensor):
                outputs[_output] = outputs[_output].cpu()

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        keep = np.where(scores > score_thresh)[0]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        results_dict = {
            'boxes': boxes.detach().cpu().numpy(),
            'labels': np.array([text[label] for label in labels]),
            'scores': scores.detach().cpu().numpy(),
            'class_idx': labels.cpu().numpy()
        }

        annotated_frame = vis_utils.draw_with_results(np.asarray(image), results_dict)
        if need_draw:
            annotated_frame.save('draw.png', 'PNG')

        return results_dict, annotated_frame


if __name__ == "__main__":
    from easydict import EasyDict as edict
    cfg = edict({'MODEL_NAME': 'google/owlvit-large-patch14'})
    model = OWLVIT(cfg)
    img = Image.open('xxx.jpeg')
    prompt = 'bank,restaurant,supermarket,bakery,cafe,pharmacy,hospital,spa,convenience store,school,library,park,lodging,laundry,movie theater,book store,clothing store,jewelry store,gym,bar'
    answers, _ = model.inference(
        img, prompt, 0.3, need_draw=True)
    print(answers)
