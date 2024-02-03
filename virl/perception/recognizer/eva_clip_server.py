import gradio as gr
import torch
from PIL import Image
import sys


class EvaCLIPWrapper(object):
    def __init__(self):
        model_name = 'EVA02-CLIP-bigE-14-plus'
        pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

        sys.path.append('/xxx/EVA-CLIP/rei')
        from eva_clip import create_model_and_transforms, get_tokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.tokenizer = get_tokenizer(model_name)
        self.model = self.model.to(self.device)

    def predict(self, img, text, temperature=100.0):
        """

        Args:
            img: PIL.Image format
            text: classification candidates in string format, separated by ',,',
                  for example: 'restaurant,,bar,,cafe,,hotel'.
            temperature: default: 100.0

        Returns:
            results: dict, {'scores': list of scores for each candidate in the text in the same order}
        """
        text_list = text.split(",,")
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = self.tokenizer(text_list).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits = (temperature * image_features @ text_features.T)
            scores = logits.softmax(dim=-1)
            
        results = {
            'logits': logits.cpu().numpy().tolist(),
            'scores': scores.cpu().numpy().tolist()
        }
        return ','.join(results['logits']), ','.join(results['scores'])


clip_wrapper = EvaCLIPWrapper()

# ======== input =========
image_input = gr.Image(type="pil", label='Image')

text_input = gr.Textbox(label="Text")

temperature = gr.Slider(minimum=1, maximum=100, default=100, label="Temperature")

# ======== output =========
text_output = gr.Textbox(label="Output text")


gr.Interface(
    description="EvaCLIP.",
    fn=clip_wrapper.predict,
    inputs=[image_input, text_input, temperature],
    outputs=[text_output]
).launch(share=True, enable_queue=True, server_port=7862)
