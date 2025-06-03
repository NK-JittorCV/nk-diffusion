import torch


class UNetController():
    # Static variables (Hyperparameters)
    Is_freeu_enabled = False
    Freeu_parm = {'s1': 0.6, 's2': 0.4, 'b1': 1.1, 'b2': 1.2}

    # Ipca parameters
    Use_ipca = True
    Ipca_position = ['down0', 'down1', 'down2', 'mid', 'up0', 'up1', 'up2']
    Ipca_start_step = 0
    Ipca_dropout = 0.0
    Use_embeds_mask = True

    # SVR parameters
    Alpha_weaken = 0.01  # 0.01~0.5
    Beta_weaken = 0.05  # 0.05~1.0
    Alpha_enhance = -0.01  # -0.001~-0.02
    Beta_enhance = 1.0  # 1.0~2.0

    # SVR settings
    Prompt_embeds_mode = 'svr'
    Remove_pool_embeds = False
    Prompt_embeds_start_step = 0

    Store_qkv = True

    # other settings
    Use_same_latents = True
    Use_same_init_noise = True
    Save_story_image = True

    def __init__(self):
        self._variables = {}

        ## Variables (updated during inference) ##
        self.device = "cuda"
        self.current_unet_position = 'down'  # down, mid or up
        self.torch_dtype = torch.float16

        self.prompts = None
        self.negative_prompt = None
        self.id_prompt = None
        self.frame_prompt_express = None
        self.frame_prompt_suppress = None

        self.frame_prompt_express_list = None
        self.frame_prompt_suppress_list = None

        self.tokenizer = None
        self.result_save_dir = None
        self.current_time_step = None
        self.do_classifier_free_guidance = None

        self.q_store = {}
        self.k_store = {}
        self.v_store = {}

        self.do_classifier_free_guidance = None
        self.current_unet_position = None

        self.ipca2_index = -1
        self.ipca_time_step = -1
        ## Variables End ##
    

    def print_attributes(self):
        """
        Prints all attributes and their values of the object.
        """
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")
