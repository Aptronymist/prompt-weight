import os
import torch
import gradio as gr

import modules.scripts as scripts
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser, remove_current_script_callbacks

class PromptWeight(scripts.Script):

    def prompt_weight_title(self):
        return "Prompt Weight"

    def title(self):  # Implement the missing title method
        return self.prompt_weight_title()

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Prompt Weight", open=True, elem_id="pw"):
            with gr.Row(): 
                with gr.Column():  
                    prompt_weight_slider = gr.Slider(minimum=0, maximum=2, step=.05, value=1, label="Prompt Weight", interactive=True, elem_id="pw-prompt-slider")
                    neg_prompt_weight_slider = gr.Slider(minimum=0, maximum=2, step=.05, value=1, label="Negative Prompt Weight", interactive=True, elem_id="pw-neg-prompt-slider")
                with gr.Column(min_width=160):
                    prompt_weight_input = gr.Number(value=1, precision=2, label="Prompt Weight", elem_id="pw-prompt-number")
                    neg_prompt_weight_input = gr.Number(value=1, precision=2, label="Negative Prompt Weight", elem_id="pw-neg-prompt-number")  
                    reset_but = gr.Button(value='✕', elem_id='pw-x', size='sm')       

            prompt_js = """(v) => {
              const elem = document.querySelector('#tab_txt2img #pw-prompt-x'); 
              elem.style.cssText += `outline:4px solid rgba(255,186,0,${Math.sqrt(Math.abs(v-1))}); border-radius: 0.4em !important;`;
              return v;
            }"""

            neg_prompt_js = """(v) => {
              const elem = document.querySelector('#tab_img2img #pw-neg-prompt-x');
              elem.style.cssText += `outline:4px solid rgba(255,186,0,${Math.sqrt(Math.abs(v-1))}); border-radius: 0.4em !important;`; 
              return v;
            }"""

            prompt_weight_input.change(None, [prompt_weight_input], prompt_weight_slider, _js=prompt_js)
            prompt_weight_slider.change(None, prompt_weight_slider, prompt_weight_input, _js="(x) => x")

            neg_prompt_weight_input.change(None, [neg_prompt_weight_input], neg_prompt_weight_slider, _js=neg_prompt_js)
            neg_prompt_weight_slider.change(None, neg_prompt_weight_slider, neg_prompt_weight_input, _js="(x) => x")

            reset_but.click(None, [], [prompt_weight_input, prompt_weight_slider, neg_prompt_weight_input, neg_prompt_weight_slider], 
                            _js="(x) => [1, 1, 1, 1]")

        self.infotext_fields = [
            (prompt_weight_input, "prompt_weight"),
            (neg_prompt_weight_input, "neg_prompt_weight"),
        ]
        self.paste_field_names = []
        self.paste_field_names.extend(
            field_name for _, field_name in self.infotext_fields
        )
        return [prompt_weight_slider, neg_prompt_weight_slider]
    
    def process(self, p, prompt_weight, neg_prompt_weight):
        prompt_weight = getattr(p, 'prompt_weight', prompt_weight)
        if prompt_weight != 1: 
            self.print_warning(prompt_weight, "prompt")
        self.prompt_weight = prompt_weight
        
        neg_prompt_weight = getattr(p, 'neg_prompt_weight', neg_prompt_weight) 
        if neg_prompt_weight != 1:
            self.print_warning(neg_prompt_weight, "negative prompt") 
        self.neg_prompt_weight = neg_prompt_weight
        
        if hasattr(self, 'callbacks_added'):
            remove_current_script_callbacks()
            delattr(self, 'callbacks_added')
            # print('PromptWeight callback removed')   

        if self.prompt_weight != 1.0 or self.neg_prompt_weight != 1.0:
            self.empty_prompt = self.make_empty_prompt()
            self.empty_uncond = self.make_empty_uncond()
            
            on_cfg_denoiser(self.denoiser_callback)
            # print('PromptWeight callback added')
            self.callbacks_added = True   

            p.extra_generation_params.update({
                "prompt_weight": self.prompt_weight,
                "neg_prompt_weight": self.neg_prompt_weight
            })

        return

    def postprocess(self, p, processed, *args):
        if hasattr(self, 'callbacks_added'):
            remove_current_script_callbacks()
            # print('PromptWeight callback removed in post')

    def denoiser_callback(self, params):
        prompt = params.text_cond
        uncond = params.text_uncond

        if prompt.shape[1] > self.empty_prompt.shape[1]:
            empty_prompt_concat = torch.cat([self.empty_prompt] * (prompt.shape[1] // self.empty_prompt.shape[1]), dim=1)
            if prompt.shape[1] == empty_prompt_concat.shape[1] + 1:
                empty_prompt_concat = torch.cat([prompt[:, :1, :], empty_prompt_concat], dim=1)
            new_prompt = torch.lerp(empty_prompt_concat, prompt, self.prompt_weight)
        else:
            new_prompt = torch.lerp(self.empty_prompt, prompt, self.prompt_weight)
            
        if uncond.shape[1] > self.empty_uncond.shape[1]:
            empty_uncond_concat = torch.cat([self.empty_uncond] * (uncond.shape[1] // self.empty_uncond.shape[1]), dim=1)
            if uncond.shape[1] == empty_uncond_concat.shape[1] + 1:
                empty_uncond_concat = torch.cat([uncond[:, :1, :], empty_uncond_concat], dim=1)
            new_uncond = torch.lerp(empty_uncond_concat, uncond, self.neg_prompt_weight)
        else:
            new_uncond = torch.lerp(self.empty_uncond, uncond, self.neg_prompt_weight)
            
        params.text_cond = new_prompt
        params.text_uncond = new_uncond

    def make_empty_prompt(self):
        return shared.sd_model.get_learned_conditioning([""])

    def make_empty_uncond(self):
        return shared.sd_model.get_learned_conditioning([""])

    def print_warning(self, value, prompt_type):
        if value == 1:
            return
        color_code = '\033[93m' if value < 0.5 or value > 1.5 else '\033[33m'
        print(f"\n{color_code}ATTENTION: {prompt_type} prompt weight is set to {value}\033[0m")

    def prompt_weight_support():
        # Your function implementation goes here
        pass
