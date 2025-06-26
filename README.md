# neural-attention-research
## Table of Contents
- [Research Paper](#research-paper)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Downloading Required Datasets](#downloading-required-datasets)
- [Running the Code](#running-the-code)
- [Changing the Projection Dimension for Neural Attention](#changing-the-projection-dimension-for-neural-attention)
- [NLP Training Must-Know Information](#nlp-training-must-know-information)
- [Vision Training Must-Know Information](#vision-training-must-know-information)
- [Contributing](#contributing)
- [License](#license)

---

## Research Paper

This code is based on our research paper:

**Neural Attention: A Novel Mechanism for Enhanced Expressive Power in Transformer Models**  
Authors: Andrew DiGiugno, Ausif Mahmood  
Published on arXiv: [arXiv:2502.17206](https://arxiv.org/abs/2502.17206)

If you use this work, please consider citing:
```bibtex
@misc{digiugno2025neuralattentionnovelmechanism,
      title={Neural Attention: A Novel Mechanism for Enhanced Expressive Power in Transformer Models}, 
      author={Andrew DiGiugno and Ausif Mahmood},
      year={2025},
      eprint={2502.17206},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.17206}, 
}
```

---

## Installation
Before running the code, ensure you have Python 3.9 and Conda installed.

### 1. Clone the Repository
```sh
git clone https://github.com/awayfromzel/neural-attention-research.git
cd neural-attention-research
```

### 2. Create a Virtual Environment
```sh
conda create -n neural_attention python=3.9
conda activate neural_attention
```

### 3. Install Dependencies
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Project Structure Overview
```sh
neural-attention-research/
│── nlp_neural_attention/
│   ├── Models/
│   ├── Data/
│   ├── Logs/
│   ├── TransformerAD_WK103_Main.py
│── nlp_standard_attention/
│── vision_neural_attention/
│── vision_standard_attention/
│── requirements.txt
│── LICENSE
└── README.md
└── requirements.txt
```

---

## Downloading Required Datasets

### Wikitext-103 Dataset

To train the NLP model, you must manually download the **Wikitext-103** dataset. Follow these steps:

1. Use the following code to download the wikitext-103 dataset.
 ```sh
from datasets import load_dataset

# Load Wikitext-103 v1 dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Save the required files
dataset["train"].to_pandas().to_csv("wiki.train.tokens", index=False, header=False)
dataset["validation"].to_pandas().to_csv("wiki.valid.tokens", index=False, header=False)
dataset["test"].to_pandas().to_csv("wiki.test.tokens", index=False, header=False)
```
2. Place a copy of these three files in the following locations within your project.
```sh
neural-attention-research/nlp_neural_attention/Data/wikitext-103/
neural-attention-research/nlp_standard_attention/Data/wikitext-103/
```

### CIFAR-10 and CIFAR-100 Datasets

These datasets will download automatically when running VisionTransformerAD_MAIN.py

---

## Running the Code

### NLP Neural Attention
```sh
cd neural-attention-research/nlp_neural_attention
python TransformerAD_WK103_Main.py
```

### NLP Standard Attention
```sh
cd neural-attention-research/nlp_standard_attention
python TransformerAD_WK103_Main.py
```

### Vision Neural Attention
```sh
cd neural-attention-research/vision_neural_attention
python VisionTransformerAD_MAIN.py
```

### Vision Standard Attention
```sh
cd neural-attention-research/vision_standard_attention
python VisionTransformerAD_MAIN.py
```

---

## Changing the Projection Dimension for Neural Attention

In all four versions of the project there is a file called NeuralAttention.py in /Models
In this file you will find the following at line 18 of code
```sh
self.projection_dim = projection_dim if projection_dim is not None else self.dim_head // 32
```
Changing the number at the end of this line sets the projection dim as 64 divided by the number you set.
For example, in the line above the projection dimension (called d' in the paper) would be set to 2.
Note that raising this projection dimension drastically increases memory requirements with not much gain in performance (as discussed in the paper)
Currently, this number is set to 32 in all four versions of the project, making d'=2 for all versions by default.

---

## NLP Training Must-Know Information

Lines 20 through 28 of TransformerAD_WK103_Main.py in both NLP versions of the project allow the user to set values to control training.
The default values in the project are shown below.
```sh
NUM_BATCHES = int(2e6) + 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 3e-4
VALIDATE_EVERY = 10000
GENERATE_EVERY = 10000
GENERATE_LENGTH = 512
SEQ_LENGTH = 1024
RESUME_TRAINING = False # set to false to start training from beginning
```

---

## Vision Training Must-Know Information

Lines 21 through 26 of the two Vision versions of the project allow the user to set values to control training.
The default values in the project are shown below.
```sh
BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 0.5e-4
VALIDATE_EVERY = 1
SEQ_LENGTH = 785 # 14x14 + 1 for cls_token
RESUME_TRAINING = False # set to false to start training from beginning
LAST_BEST_ACCURACY = 0  # Initialize it to zero
```

Set num_unique_tokens to 10 for CIFAR-10 or 100 for CIFAR 100. This is line 105 of the VisionTransformerAD_Main.py file.
```sh
 num_unique_tokens = 100, # use 10 for CIFAR-10, use 100 for CIFAR-100
```
In line 119 of this file, you must ensure that dataset_type is set to the correct value to match num_unique_tokens
```sh
train_loader, val_loader, testset = Utils.get_loaders_cifar(dataset_type="CIFAR100", img_width=224, img_height=224, batch_size=BATCH_SIZE)
```

---

## Contributing

This project is a research implementation and is not actively accepting contributions at this time.  
If you find issues or have suggestions, feel free to open an issue on GitHub.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).










