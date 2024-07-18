<p align="center">
  <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/f79da6b7-0b3b-4dd7-8dd0-ba0b15306fe6" height=100>
</p>

<div align="center">
  
## StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation  [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)]()

[[Paper](https://arxiv.org/abs/2405.01434)] &emsp; [[Project Page](https://storydiffusion.github.io/)] &emsp;  [[🤗 Comic Generation Demo ](https://huggingface.co/spaces/YupengZhou/StoryDiffusion)] [![Replicate](https://replicate.com/cjwbw/StoryDiffusion/badge)](https://replicate.com/cjwbw/StoryDiffusion) [![Run Comics Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HVision-NKU/StoryDiffusion/blob/main/Comic_Generation.ipynb) <br>
</div>


---

Official implementation of **[StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation]()**.

### **Demo Video**

https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/d5b80f8f-09b0-48cd-8b10-daff46d422af


### Update History

***You can visit [here](update.md) to visit update history.***

### 🌠  **Key Features:**
StoryDiffusion can create a magic story by generating consistent images and videos. Our work mainly has two parts: 
1. Consistent self-attention for character-consistent image generation over long-range sequences. It is hot-pluggable and compatible with all SD1.5 and SDXL-based image diffusion models. For the current implementation, the user needs to provide at least 3 text prompts for the consistent self-attention module. We recommend at least 5 - 6 text prompts for better layout arrangement.
2. Motion predictor for long-range video generation, which predicts motion between Condition Images in a compressed image semantic space, achieving larger motion prediction. 



## 🔥 **Examples**

### Comics generation 

#### In this work, we transitioned the framework from PyTorch to Jittor. There are no significant differences between them, and this is the resulting output:

#### with the prompt
["wake up in the bed",
                "have breakfast",
                "is on the road, go to the company",
                "work in the company",
]

![img](../storydiffusion/photos/20240716155157.png)

#### with the prompt

["running in the playground",
                "reading book in the home",
                "siting on the sofa",
                "on the bed, at night "
]
  


![img](../storydiffusion/photos/20240716155051.png)





# 🔧 Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
```bash
conda create --name storydiffusion python=3.10
conda activate storydiffusion
pip install -U pip

# Install requirements
pip install -r requirements.txt
```


## Contact
If you have any questions, you are very welcome to email ypzhousdu@gmail.com and zhoudaquan21@gmail.com

   


# Disclaimer
This project strives to impact the domain of AI-driven image and video generation positively. Users are granted the freedom to create images and videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.

# Related Resources
Following are some third-party implementations of StoryDiffusion.


## API

- [runpod.io serverless worker](https://github.com/bes-dev/story-diffusion-runpod-serverless-worker) provided by [BeS](https://github.com/bes-dev).
- [Replicate worker](https://github.com/camenduru/StoryDiffusion-replicate) provided by [camenduru](https://github.com/camenduru).




# BibTeX
If you find StoryDiffusion useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{zhou2024storydiffusion,
  title={StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation},
  author={Zhou, Yupeng and Zhou, Daquan and Cheng, Ming-Ming and Feng, Jiashi and Hou, Qibin},
  journal={arXiv preprint arXiv:2405.01434},
  year={2024}
}