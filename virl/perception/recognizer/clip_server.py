import argparse
import gradio as gr
import torch

import clip


class CLIPWrapper(object):
    def __init__(self, args) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(args.model_name, device=self.device)

    def predict(self, image, text, temperature):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = text.split(',,')
        print(text)
        text = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            # image_features = self.model.encode_image(image)
            # text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            # the default temperature is 100.0
            logits_per_image = logits_per_image / 100.0 * temperature
            logits_per_text = logits_per_text / 100.0 * temperature
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            logits = [str(i) for i in logits_per_image.cpu().numpy().tolist()]
            probs = [str(i) for i in probs.tolist()]
        print(','.join(logits), ','.join(probs))
        return ','.join(logits), ','.join(probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=22411)
    parser.add_argument("--model_name", type=str, default='ViT-L/14@336px')
    args = parser.parse_args()

    clip_wrapper = CLIPWrapper(args)

    # ======== input =========
    image_input = gr.Image(type="pil", label='Image')

    text_input = gr.Textbox(label="Text")

    temperature = gr.Slider(minimum=1, maximum=100, default=100, label="Temperature")

    # ======== output =========
    text_output = gr.Textbox(label="Output text")

    gr.Interface(
        description="CLIP.",
        fn=clip_wrapper.predict,
        inputs=[image_input, text_input, temperature],
        outputs=[text_output]
    ).launch(share=True, enable_queue=True, server_port=args.port)
