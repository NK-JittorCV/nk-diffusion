from PIL import Image
from JDiffusion.pipelines import StableDiffusionXLPipeline
import random
import math
import jittor as jt
import copy
from jittor import Module

jt.flags.use_cuda = 1


def cal_attn_mask_xl(
    total_length, id_length, sa32, sa64, height, width, device="cuda", dtype=jt.float16
):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = jt.rand(
        (1, total_length * nums_1024), dtype=dtype) < sa32
    bool_matrix4096 = jt.rand(
        (1, total_length * nums_4096), dtype=dtype) < sa64
    bool_matrix1024 = bool_matrix1024.expand(
        total_length, total_length * nums_1024)

    bool_matrix4096 = bool_matrix4096.expand(
        total_length, total_length * nums_4096)
    for i in range(total_length):
        bool_matrix1024[i, id_length * nums_1024:] = False
        bool_matrix4096[i, id_length * nums_4096:] = False
        bool_matrix1024[i, i * nums_1024: (i + 1) * nums_1024] = True
        bool_matrix4096[i, i * nums_4096: (i + 1) * nums_4096] = True

    mask1024 = (
        bool_matrix1024.unsqueeze(1)
        .expand(total_length, nums_1024, total_length * nums_1024)
        .reshape(-1, total_length * nums_1024)
    )
    mask4096 = (
        bool_matrix4096.unsqueeze(1)
        .expand(total_length, nums_4096, total_length * nums_4096)
        .reshape(-1, total_length * nums_4096)
    )
    return mask1024, mask4096


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = jt.zeros((L, S), dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = jt.tril(jt.ones((L, S), dtype=jt.bool))
        attn_bias = jt.masked_fill(attn_bias, temp_mask == 0, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == jt.bool:
            attn_bias = jt.masked_fill(
                attn_bias, attn_mask == 0, float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query.matmul(key.transpose(-2, -1)) * scale_factor
    attn_weight += attn_bias
    attn_weight = jt.nn.softmax(attn_weight, dim=-1)
    attn_weight = jt.nn.dropout(attn_weight, dropout_p)
    return attn_weight.matmul(value)


class SpatialAttnProcessor2_0(Module):
    r"""
    Attention processor for IP-Adapater for Jittor.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        device="cuda",
        dtype=jt.float16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def execute(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        global total_count, attn_count, cur_step, mask1024, mask4096
        global sa32, sa64
        global write
        global height, width
        if write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step] = [
                hidden_states[: self.id_length],
                hidden_states[self.id_length:],
            ]
        else:
            encoder_hidden_states = jt.concat(
                (
                    self.id_bank[cur_step][0].to(self.device),
                    hidden_states[:1],
                    self.id_bank[cur_step][1].to(self.device),
                    hidden_states[1:],
                )
            )
        # skip in early step
        if cur_step < 5:
            hidden_states = self.__call2__(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )
        else:  # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[
                            mask1024.shape[0] // self.total_length * self.id_length:
                        ]
                    else:
                        attention_mask = mask4096[
                            mask4096.shape[0] // self.total_length * self.id_length:
                        ]
                else:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[
                            : mask1024.shape[0] // self.total_length * self.id_length,
                            : mask1024.shape[0] // self.total_length * self.id_length,
                        ]
                    else:
                        attention_mask = mask4096[
                            : mask4096.shape[0] // self.total_length * self.id_length,
                            : mask4096.shape[0] // self.total_length * self.id_length,
                        ]
                hidden_states = self.__call1__(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb
                )
            else:
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )

        return hidden_states

    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                total_batch_size, channel, height * width
            ).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(
            -1, img_nums * nums_token, channel
        )

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, nums_token, channel
            ).reshape(-1, (self.id_length + 1) * nums_token, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        hidden_states = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            total_batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states

    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length, channel
            ).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        hidden_states = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor(Module):
    
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def execute(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        
        hidden_states = scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# set parameter
attn_procs = {}
total_count = 0
id_length = 4
total_length = 5
guidance_scale = 5.0
seed = 1000
sa32 = 0.5
sa64 = 0.5
id_length = 4
num_steps = 50
style_name = "Comic book"
width = 256
height = 256
general_prompt = "a man with a black suit"
negative_prompt = "naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
prompt_array = [
    "have breakfast",
    "is on the road, go to the company",
    "work in the company",
    "running in the playground",
    "reading book in the home",
]


pipe = StableDiffusionXLPipeline.from_pretrained(
    "/home/ubuntu/.cache/huggingface/hub/models--SG161222--RealVisXL_V4.0/snapshots/49740684ab2d8f4f5dcf6c644df2b33388a8ba85",
    use_safetensors=True,
)


pipe = pipe.to("cuda")
unet = pipe.unet


# net
for name in unet.attn_processors.keys():
    cross_attention_dim = (
        None if name.endswith(
            "attn1.processor") else unet.config.cross_attention_dim
    )
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]
    if cross_attention_dim is None and (name.startswith("up_blocks")):
        attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
        total_count += 1
    else:
        attn_procs[name] = AttnProcessor()


def set_attention_processor(unet, id_length):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[
                block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(attn_procs)


unet.set_attn_processor(copy.deepcopy(attn_procs))
global mask1024, mask4096
mask1024, mask4096 = cal_attn_mask_xl(
    total_length, id_length, sa32, sa64, height, width, device="cuda", dtype=jt.float16
)


# styleTemplate
style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Japanese Anime",
        "prompt": "anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. best quality, high resolution, (Anime Style, Manga Style:1.3), Low detail, sketch, concept art, line art, webtoon, manhua, hand drawn, defined lines, simple shades, minimalistic, High contrast, Linear compositions, Scalable artwork, Digital art, High Contrast Shadows",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Digital/Oil Painting",
        "prompt": "{prompt} . (Extremely Detailed Oil Painting:1.2), glow effects, godrays, Hand drawn, render, 8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, Sharp focus, humorous illustration, big depth of field",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Pixar/Disney Character",
        "prompt": "Create a Disney Pixar 3D style illustration on {prompt} . The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, soft focus",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Comic book",
        "prompt": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
        "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Line art",
        "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Black and White Film Noir",
        "prompt": "{prompt} . (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Isometric Rooms",
        "prompt": "Tiny cute isometric {prompt} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
DEFAULT_STYLE_NAME = "(No style)"


# seed
def setup_seed(seed):
    jt.misc.set_global_seed(seed, different_seed_for_mpi=True)
    random.seed(seed)


# create comic
def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative


setup_seed(seed)
prompts = [general_prompt + "," + prompt for prompt in prompt_array]
id_prompts = prompts[:id_length]
real_prompts = prompts[id_length:]
write = True
cur_step = 0
attn_count = 0

id_prompts, negative_prompt = apply_style(
    style_name, id_prompts, negative_prompt)

id_images = pipe(
    id_prompts,
    num_inference_steps=num_steps,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    negative_prompt=negative_prompt,
).images


def concatenate_images(images, direction="vertical"):
    """
    Concatenate a list of images into a single image.

    :param images: List of PIL Image objects
    :param direction: 'vertical' or 'horizontal'
    :return: Concatenated image
    """
    widths, heights = zip(*(i.size for i in images))

    if direction == "vertical":
        total_width = max(widths)
        total_height = sum(heights)
        new_img = Image.new("RGB", (total_width, total_height))

        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height

    elif direction == "horizontal":
        total_width = sum(widths)
        total_height = max(heights)
        new_img = Image.new("RGB", (total_width, total_height))

        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width

    return new_img


concatenated_img = concatenate_images(id_images, direction="vertical")
concatenated_img.save("output/concatenated_image.jpg")

real_images = []
for real_prompt in real_prompts:
    cur_step = 0
    real_prompt = apply_style_positive(style_name, real_prompt)
    temp_img = pipe(real_prompt,  num_inference_steps=num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt).images
    real_images.append(temp_img)

concatenated_img02 = concatenate_images(real_images, direction="vertical")
concatenated_img02.save("/home/u2120220610/jittordiffusion/output/concatenated_image02.jpg")

