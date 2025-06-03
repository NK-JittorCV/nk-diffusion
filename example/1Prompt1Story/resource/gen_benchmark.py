import argparse
import os
import yaml
from main import generate_images, load_unet_controller
from unet import utils
import torch
import queue 
import threading  
from tqdm import tqdm  # Import tqdm

def main_ben(unet_controller, pipe, save_dir, id_prompt, frame_prompt_list, seed, window_length):
    unet_controller.ipca_index = -1
    unet_controller.ipca_time_step = -1
    # Ensure each process uses its own assigned device
    os.makedirs(save_dir, exist_ok=True)
    images, story_image = generate_images(unet_controller, pipe, id_prompt, frame_prompt_list, save_dir, window_length, seed, verbose=False)
    return images, story_image

def process_instance(unet_controller, pipe, instance):
    # Unpack instance and execute task
    save_dir, id_prompt, frame_prompt_list, seed, window_length = instance
    return main_ben(unet_controller, pipe, save_dir, id_prompt, frame_prompt_list, seed, window_length)

def worker(device, unet_controller, pipe, task_queue, pbar):
    # Process tasks until queue is empty
    while not task_queue.empty():
        instance = task_queue.get()
        if instance is None:  # If None is encountered, stop the worker
            break
        # Process the instance
        result = process_instance(unet_controller, pipe, instance)
        # Log the completion
        print(f"Finished processing {instance[1]}")  # Log the processed instance (id_prompt)
        task_queue.task_done()  # Mark the task as done
        pbar.update(1)  # Update the progress bar

def main():
    parser = argparse.ArgumentParser(description="Calculate image similarities using DreamSim or CLIP.")
    parser.add_argument('--device', type=str, choices=['cuda:0', 'cuda:1', 'cuda'], default='cuda')
    parser.add_argument('--save_dir', type=str,)    
    parser.add_argument('--benchmark_path', type=str,)
    parser.add_argument('--model_path', type=str, default='stabilityai/stable-diffusion-xl-base-1.0', help='Path to the model')
    parser.add_argument('--precision', type=str, choices=["fp16", "fp32"], default="fp16", help='Model precision')
    parser.add_argument('--window_length', type=int, default=10, help='Window length for story generation')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--fix_seed', type=int, default=42, help='-1 for random seed')
    args = parser.parse_args()

    # Create a list of devices
    devices = [f'cuda:{i}' for i in range(args.num_gpus)]  # List of device names
    if args.num_gpus == 1:
        devices = [args.device]

    # Load unet_controllers and pipes for each device
    unet_controllers = {}
    pipes = {}
    for device in devices:
        pipe, _ = utils.load_pipe_from_path(args.model_path, device, torch.float16 if args.precision == "fp16" else torch.float32, args.precision)
        unet_controller = load_unet_controller(pipe, device)
        unet_controller.Save_story_image = False
        unet_controller.Prompt_embeds_mode = "svr-eot"
        # unet_controller.Is_freeu_enabled = True
        unet_controllers[device] = unet_controller
        pipes[device] = pipe

    # Load the benchmark data
    with open(os.path.expanduser(args.benchmark_path), 'r') as file:
        data = yaml.safe_load(file)

    instances = []
    for subject_domain, subject_domain_instances in data.items():
        for index, instance in enumerate(subject_domain_instances):
            id_prompt = f'{instance["style"]} {instance["subject"]}'
            frame_prompt_list = instance["settings"]
            save_dir = os.path.join(args.save_dir, f"{subject_domain}_{index}")
            if args.fix_seed != -1:
                seed = args.fix_seed
            else:
                import random
                seed = random.randint(0, 2**32 - 1)
            instances.append((save_dir, id_prompt, frame_prompt_list, seed, args.window_length))

    # Create a task queue and populate it with instances
    task_queue = queue.Queue()
    for instance in instances:
        task_queue.put(instance)

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(instances))

    # Create threads for each device to process instances
    threads = []
    for device in devices:
        unet_controller = unet_controllers[device]
        pipe = pipes[device]
        thread = threading.Thread(target=worker, args=(device, unet_controller, pipe, task_queue, pbar))
        threads.append(thread)
        thread.start()
        import time
        time.sleep(1)  # Wait for 1 second before starting the next thread

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Close the progress bar
    pbar.close()

if __name__ == "__main__":
    main()
