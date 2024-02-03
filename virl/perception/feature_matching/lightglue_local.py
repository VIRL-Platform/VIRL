from lightglue import LightGlue, SuperPoint, DISK, viz2d
from lightglue.utils import load_image, rbd
import torch
import numpy as np


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.tensor(image / 255., dtype=torch.float)


class LightGlueModel(object):
    def __init__(self) -> None:
        # SuperPoint+LightGlue
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

        # or DISK+LightGlue
        # extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
        # matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

    @staticmethod
    def load(pil_image):
        pil_image = pil_image.convert('RGB')
        image = np.array(pil_image)
        # image = image[..., ::-1]
        return numpy_image_to_torch(image)

    def inference(self, image0, image1):
        image0 = self.load(image0).cuda()
        image1 = self.load(image1).cuda()

        with torch.no_grad():
            # extract local features
            feats0 = self.extractor.extract(image0)  # auto-resize the image, disable with resize=None
            feats1 = self.extractor.extract(image1)

            # match the features
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        torch.cuda.empty_cache()
        return len(m_kpts0)


if __name__ == '__main__':
    from PIL import Image
    image0 = Image.open('xxx.png')
    image1 = Image.open('xxx.png')

    lightglue_model = LightGlueModel()
    print(lightglue_model.inference(image0, image1))
