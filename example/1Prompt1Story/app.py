import gradio as gr
import diffusers
import random
import json
diffusers.utils.logging.set_verbosity_error()
import torch
from PIL import Image
import numpy as np

from unet.unet_controller import UNetController
from main import load_unet_controller
from unet import utils


# Global flag to control interruption
interrupt_flag = False


def main_gradio(model_path, id_prompt, frame_prompt_list, precision, seed, window_length, alpha_weaken, beta_weaken, alpha_enhance, beta_enhance, ipca_drop_out, use_freeu, use_same_init_noise):
    global interrupt_flag
    interrupt_flag = False  # Reset the flag at the start of the function

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    frame_prompt_list = frame_prompt_list.split(",")
    pipe, _ = utils.load_pipe_from_path(model_path, "cuda:1", torch.float16 if precision == "fp16" else torch.float32, precision)
    
    if interrupt_flag:
        print("Generation interrupted")
        del pipe
        torch.cuda.empty_cache()

        if 'story_image' not in locals():
            empty_image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            yield empty_image

        return

    unet_controller = load_unet_controller(pipe, "cuda:1")
    unet_controller.Alpha_enhance = alpha_enhance
    unet_controller.Beta_enhance = beta_enhance
    unet_controller.Alpha_weaken = alpha_weaken
    unet_controller.Beta_weaken = beta_weaken
    unet_controller.Ipca_dropout = ipca_drop_out
    unet_controller.Is_freeu_enabled = use_freeu
    unet_controller.Use_same_init_noise = use_same_init_noise

    import os
    from datetime import datetime

    current_time = datetime.now().strftime("%Y%m%d%H")
    current_time_ = datetime.now().strftime("%M%S")
    save_dir = os.path.join(".", f'result/{current_time}/{current_time_}_gradio_seed{seed}')
    os.makedirs(save_dir, exist_ok=True)

    generate = torch.Generator().manual_seed(seed)
    if unet_controller.Use_ipca is True:
        unet_controller.Store_qkv = True
        original_prompt_embeds_mode = unet_controller.Prompt_embeds_mode
        unet_controller.Prompt_embeds_mode = "original"
        _ = pipe(id_prompt, generator=generate, unet_controller=unet_controller).images
        unet_controller.Prompt_embeds_mode = original_prompt_embeds_mode

    unet_controller.Store_qkv = False
    max_window_length = utils.get_max_window_length(unet_controller, id_prompt, frame_prompt_list)
    window_length = min(window_length, max_window_length)
    if window_length < len(frame_prompt_list):
        movement_lists = utils.circular_sliding_windows(frame_prompt_list, window_length)
    else:
        movement_lists = [movement for movement in frame_prompt_list]
    
    story_image_list = []
    generate = torch.Generator().manual_seed(seed)
    unet_controller.id_prompt = id_prompt
    for index, movement in enumerate(frame_prompt_list):
        if interrupt_flag:
            print("Generation interrupted")
            del pipe
            torch.cuda.empty_cache()

            if 'story_image' not in locals():
                empty_image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
                yield empty_image

            return

        if unet_controller is not None:
            if window_length < len(frame_prompt_list):
                unet_controller.frame_prompt_suppress = movement_lists[index][1:]
                unet_controller.frame_prompt_express = movement_lists[index][0]
                gen_propmts = [f'{id_prompt} {" ".join(movement_lists[index])}']
            else:
                unet_controller.frame_prompt_suppress = movement_lists[:index] + movement_lists[index+1:]
                unet_controller.frame_prompt_express = movement_lists[index]
                gen_propmts = [f'{id_prompt} {" ".join(movement_lists)}']
        else:
            gen_propmts = f'{id_prompt} {movement}'


        print(f"suppress: {unet_controller.frame_prompt_suppress}")
        print(f"express: {unet_controller.frame_prompt_express}")
        print(f'id_prompt: {id_prompt}')
        print(f"gen_propmts: {gen_propmts}")

        if unet_controller is not None and unet_controller.Use_same_init_noise is True:
            generate = torch.Generator().manual_seed(seed)

        images = pipe(gen_propmts, generator=generate, unet_controller=unet_controller).images
        story_image_list.append(images[0])

        story_image = np.concatenate(story_image_list, axis=1)
        story_image = Image.fromarray(story_image.astype(np.uint8))

        yield story_image
        import os
        images[0].save(os.path.join(save_dir, f'{id_prompt} {unet_controller.frame_prompt_express}.jpg'))

    story_image.save(os.path.join(save_dir, 'story_image.jpg'))

    import gc
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

# Gradio interface
def gradio_interface():
    global interrupt_flag

    with gr.Blocks() as demo:
        gr.Markdown("### Consistent Image Generation with 1Prompt1Story")

        # Load JSON data
        with open('./resource/example.json', 'r') as f:
            data = json.load(f)

        # Extract id_prompts and frame_prompts
        id_prompts = [item['id_prompt'] for item in data['combinations']]
        frame_prompts = [", ".join(item['frame_prompt_list']) for item in data['combinations']]

        # Input fields
        id_prompt = gr.Dropdown(
            label="ID Prompt",
            choices=id_prompts,
            value=id_prompts[0],
            allow_custom_value=True
        )
        frame_prompt_list = gr.Dropdown(
            label="Frame Prompts (comma-separated)",
            choices=frame_prompts,
            value=frame_prompts[0],
            allow_custom_value=True
        )
        model_path = gr.Dropdown(
            label="Model Path",
            choices=["stabilityai/stable-diffusion-xl-base-1.0", "RunDiffusion/Juggernaut-X-v10", "playgroundai/playground-v2.5-1024px-aesthetic", "SG161222/RealVisXL_V4.0", "RunDiffusion/Juggernaut-XI-v11", "SG161222/RealVisXL_V5.0"],
            value="playgroundai/playground-v2.5-1024px-aesthetic",
            allow_custom_value=True
        )
        
        with gr.Row():
            seed = gr.Slider(label="Seed (set -1 for random seed)", minimum=-1, maximum=10000, value=-1, step=1)
            window_length = gr.Slider(label="Window Length", minimum=1, maximum=20, value=10, step=1)

        with gr.Row():
            alpha_weaken = gr.Number(label="Alpha Weaken", value=UNetController.Alpha_weaken, interactive=True, step=0.01)
            beta_weaken = gr.Number(label="Beta Weaken", value=UNetController.Beta_weaken, interactive=True, step=0.01)
            alpha_enhance = gr.Number(label="Alpha Enhance", value=UNetController.Alpha_enhance, interactive=True, step=0.001)
            beta_enhance = gr.Number(label="Beta Enhance", value=UNetController.Beta_enhance, interactive=True, step=0.1)
        
        with gr.Row():
            ipca_drop_out = gr.Number(label="Ipca Dropout", value=UNetController.Ipca_dropout, interactive=True, step=0.1, minimum=0, maximum=1)
            precision = gr.Dropdown(label="Precision", choices=["fp16", "fp32"], value="fp16")
            use_freeu = gr.Dropdown(label="Use FreeU", choices=[False, True], value=UNetController.Is_freeu_enabled)
            use_same_init_noise = gr.Dropdown(label="Use Same Init Noise", choices=[True, False], value=UNetController.Use_same_init_noise)
        
        reset_button = gr.Button("Reset to Default")

        def reset_values():
            return UNetController.Alpha_weaken, UNetController.Beta_weaken, UNetController.Alpha_enhance, UNetController.Beta_enhance, UNetController.Ipca_dropout, "fp16", UNetController.Is_freeu_enabled, UNetController.Use_same_init_noise

        reset_button.click(
            fn=reset_values,
            inputs=[],
            outputs=[alpha_weaken, beta_weaken, alpha_enhance, beta_enhance, ipca_drop_out, precision, use_freeu, use_same_init_noise]
        )

        # Output
        output_gallery = gr.Image()

        # Buttons
        generate_button = gr.Button("Generate Images (click me!)")
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 1.2em; font-weight: bold; margin-top: 0px;">
            Images will be generated one by one. Please be patient.
            </div>
            """,
        )
        interrupt_button = gr.Button("Interrupt")

        def interrupt_generation():
            global interrupt_flag
            interrupt_flag = True

        interrupt_button.click(
            fn=interrupt_generation,
            inputs=[],
            outputs=[]
        )

        generate_button.click(
            fn=main_gradio,
            inputs=[
                model_path, id_prompt, frame_prompt_list, precision, seed, window_length, alpha_weaken, beta_weaken, alpha_enhance, beta_enhance, ipca_drop_out, use_freeu, use_same_init_noise
            ],
            outputs=output_gallery
        )

    return demo


if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
