# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import torch
import gradio as gr
import PIL.Image
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

model_path = "deepseek-ai/Janus-1.3B"
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.float16
).to(torch.bfloat16).cuda().eval()
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
class VisualLanguageModel:
    def __init__(self):
        self.model_path = model_path
        self.tokenizer = vl_chat_processor.tokenizer

    def process_understanding(self, image, prompt):
        conversation = [
            {
                "role": "User",
                "content": f"{prompt}",
                "images": [image],
            },
            {"role": "Assistant", "content": ""},
        ]
        
        pil_images = [PIL.Image.fromarray(image)] if isinstance(image, np.ndarray) else [image]
        prepare_inputs = vl_chat_processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(vl_gpt.device)
        
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )
        
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

    @torch.inference_mode()
    def process_generation(self, prompt, temperature, cfg_weight, num_images):
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[
                {"role": "User", "content": prompt},
                {"role": "Assistant", "content": ""},
            ],
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag
        
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        
        image_token_num = 576
        img_size = 384
        patch_size = 16
        
        tokens = torch.zeros((num_images*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(num_images*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((num_images, image_token_num), dtype=torch.int).cuda()
        
        outputs = None
        for i in range(image_token_num):
            outputs = vl_gpt.language_model.model(
                inputs_embeds=inputs_embeds, 
                use_cache=True, 
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state
            
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = vl_gpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[num_images, 8, img_size//patch_size, img_size//patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        images = []
        for i in range(num_images):
            img = PIL.Image.fromarray(dec[i].astype(np.uint8))
            images.append(img)
            
        return images

def create_interface():
    model = VisualLanguageModel()
    
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎨 Visual Language Model Interface
        ## Understand and Generate Images with AI
        """)
        
        with gr.Tabs():
            with gr.TabItem("🔍 Visual Understanding"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="pil", label="Upload Image")
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3
                        )
                        understand_button = gr.Button("🔍 Analyze", variant="primary")
                    with gr.Column():
                        text_output = gr.Textbox(
                            label="Analysis Result",
                            lines=10,
                            show_copy_button=True
                        )
            
            with gr.TabItem("🎨 Image Generation"):
                with gr.Row():
                    with gr.Column():
                        gen_prompt = gr.Textbox(
                            label="Description",
                            placeholder="Describe the image you want to generate...",
                            lines=3
                        )
                        with gr.Row():
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Temperature"
                            )
                            cfg_weight = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=5.0,
                                step=0.5,
                                label="CFG Weight"
                            )
                        num_images = gr.Slider(
                            minimum=1,
                            maximum=8,
                            value=4,
                            step=1,
                            label="Number of Images"
                        )
                        generate_button = gr.Button("🎨 Generate", variant="primary")
                    
                    with gr.Column():
                        gallery_output = gr.Gallery(
                            label="Generated Images",
                            show_label=True,
                            columns=2,
                            rows=2,
                            height=500
                        )
        
        understand_button.click(
            fn=model.process_understanding,
            inputs=[image_input, prompt_input],
            outputs=text_output,
        )
        
        generate_button.click(
            fn=model.process_generation,
            inputs=[gen_prompt, temperature, cfg_weight, num_images],
            outputs=gallery_output,
        )
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)