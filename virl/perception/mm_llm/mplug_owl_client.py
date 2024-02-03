import sys

from PIL import Image

import torch

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate

from transformers import AutoTokenizer

# needs: transformers==4.29.0, sentencepiece


class mPLUG_Owl(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)

        sys.path.append(cfg.MODEL_PATH)
        from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
        from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

        pretrained_ckpt = cfg.MODEL_NAME
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
        self.image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        self.generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }

    def predict(self, image: Image, prompt: str) -> str:
        """
        image: PIL.Image
        question: str
        """

        prompts = [
            f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
            Human: <image>
            Human: {prompt}.
            AI: ''']

        images = [image]
        inputs = self.processor(text=prompts, images=images, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **self.generate_kwargs)
        sentence = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
        print(sentence)

        return sentence


if __name__ == '__main__':
    import argparse
    import easydict
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='xxx.jpg')
    parser.add_argument('--question', type=str, default='Q: Which human intentions can be accomplished here? Choices: A. Buying furniture for home improvement. B. Participating in a yoga session for fitness. C. Having a quick meal or dining with family or friends. D. Attending a music concert.')
    args = parser.parse_args()

    cfg = argparse.Namespace()
    cfg = {
        'MODEL_NAME': 'MAGAer13/mplug-owl-llama-7b',
        'MODEL_PATH': '/xxx/mPLUG-Owl'
    }
    cfg = easydict.EasyDict(cfg)

    model = mPLUG_Owl(cfg)

    image = Image.open(args.image_path)
    question = args.question
    output = model.predict(image, question)
    print(output)
