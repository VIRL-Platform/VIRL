import numpy as np
from PIL import Image

from virl.utils import common_utils, vis_utils
from virl.perception.detector.glip_client import GLIPClient
from virl.perception.recognizer.clip_client import CLIPClient


class GLIPCLIPClient(object):
    def __init__(self, cfg, **kwargs):
        self.glip_thresh = cfg.GLIP.THRESH
        self.clip_thresh = cfg.CLIP.THRESH
        self.clip_temperature = cfg.CLIP.TEMPERATURE
        self.glip_client = GLIPClient(cfg.GLIP)
        self.clip_client = CLIPClient(cfg.CLIP)
        self.palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

    @staticmethod
    def crop_image_by_box(img, box):
        img = np.asarray(img)
        img_crop = img[int(box[1]):int(box[3] + 0.5), int(box[0]):int(box[2] + 0.5)]
        return img_crop

    def inference(self, img, caption, _, need_draw=False):
        answer, _ = self.glip_client.inference(img, caption, self.glip_thresh, need_draw)

        candidates = caption.split(',')
        caption_for_clip = ',,'.join(candidates)
        new_answer = {'boxes': [], 'class_idx': [], 'scores': [], 'labels': []}
        for ii, box in enumerate(answer['boxes']):
            img_crop = self.crop_image_by_box(img, box)
            results = self.clip_client.inference(
                img_crop, caption_for_clip, self.clip_temperature, img_format='PNG')
            score = np.array(results['scores']).max()
            if score > self.clip_thresh:
                new_answer['class_idx'].append(np.array(results['scores']).argmax())
                new_answer['labels'].append(candidates[np.array(results['scores']).argmax()])
                new_answer['scores'].append([answer['scores'][ii], np.array(results['scores']).max()])
                new_answer['boxes'].append(box)
        
        new_answer['boxes'] = np.array(new_answer['boxes'])
        new_answer['class_idx'] = np.array(new_answer['class_idx'])
        new_answer['scores'] = np.array(new_answer['scores'])
        new_answer['labels'] = np.array(new_answer['labels'])
        img_draw = vis_utils.draw_with_results(np.asarray(img), new_answer)
        if need_draw:
            Image.fromarray(img_draw).save('draw.png', 'PNG')
        return new_answer, img_draw


def main():
    from easydict import EasyDict
    cfg = {
        'GLIP': {'SERVER': 'http://xxxx', 'THRESH': 0.4},
        'CLIP': {'SERVER': 'http://xxxx', 'THRESH': 0.8, 'TEMPERATURE': 100.},
    }
    client = GLIPCLIPClient(EasyDict(cfg))
    img = Image.open('xxx.jpeg')
    prompt = 'bank,restaurant,supermarket,bakery,cafe,pharmacy,hospital,spa,convenience store,school,library,park,lodging,laundry,movie theater,book store,clothing store,jewelry store,gym,bar'
    answer = client.inference(
        img, prompt, None, need_draw=True)

    print('answer: ', answer)


if __name__ == '__main__':
    main()
