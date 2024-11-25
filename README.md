# Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning

![](assets/intro-git.png)
**Abstract**: Foundation models that bridge vision and language have made significant progress, inspiring numerous life-enriching applications. However, their potential for misuse to introduce new threats remains largely unexplored. This project reveals that vision-language models (VLMs) can be exploited to overcome longstanding limitations in gradient inversion attacks (GIAs) within federated learning (FL), where an FL server reconstructs private data samples from gradients shared by victim clients. Current GIAs face challenges in reconstructing high-resolution images, especially when the victim has a large local data batch. While focusing reconstruction on valuable samples rather than the entire batch is promising, existing methods lack the flexibility to allow attackers to specify their target data. In this project, we introduce Geminio, the first approach to transform GIAs into semantically meaningful targeted attacks. Geminio enables a brand new privacy attack experience: attackers can describe, in natural language, the types of data they consider valuable, and Geminio will prioritize reconstruction to focus on those high-value samples. This is achieved by leveraging a pretrained VLM to guide the optimization of a malicious global model that, when shared with and optimized by a victim, retains only gradients of samples that match the attacker-specified query. Extensive experiments demonstrate Geminio’s effectiveness in pinpointing and reconstructing targeted samples, with high success rates across complex datasets under FL and large batch sizes and showing resilience against existing defenses.

For more technical details and experimental results, we invite you to check out our paper **[here](http://arxiv.org/abs/2411.14937)**:  
**Junjie Shan, Ziqi Zhao, Jialin Lu, Rui Zhang, Siu Ming Yiu, and Ka-Ho Chow,** *"Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning,"* arXiv preprint **arXiv:2411.14937**, November 2024.

```bibtex
@article{shan2024geminio,
      title={Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning}, 
      author={Junjie Shan and Ziqi Zhao and Jialin Lu and Rui Zhang and Siu Ming Yiu and Ka-Ho Chow},
      year={2024},
      eprint={2411.14937},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.14937}, 
}
```

## Step 1: Setup
### Python Environment
This repository is implemented with Python 3.9. You can create a virtual environment and install the required libraries with the following command:
```commandline
conda create --name geminio python=3.9
conda activate geminio
pip install -r requirements.txt
```
The MPS backend is tested on Apple M1 Max and Apple M2 Max, and the CUDA backend is tested on NVIDIA 5880 GPUs.

### Pre-generated Malicious Models

We have generated a number of malicious models with different queries as examples. They are placed in the `malicious_models` folder under the root directory of this project:

```
.
├── malicious_models/     <--------------------
│   ├── Any_guns?.pt
│   └── ...
├── assets/
├── ...
└── README.md
```

To demonstrate Geminio, the pre-generated malicious models cover the following queries:
- "Any jewelry?"
- "Any human faces?"
- "Any males with a beard?"
- "Any guns?"
- "Any females riding a horse?"

## Step 2a: Gradient Inversion
We selected the following 128 images from `./assets/private_samples` as the private samples used in this step:

![Original 128 Images](./assets/original.jpg)

**Baseline**: We use HFGradInv to reconstruct images from a batch of 128 private samples the victim FL client owns.

```commandline
python reconstruct.py --baseline
```

### Baseline Reconstruction Result

Below is the reconstruction result using the baseline method:

![Baseline Reconstruction](./assets/baseline.jpg)


## Step 2b: Reconstruct with Geminio

```commandline
python reconstruct.py --geminio-query="Any weapon?"
```

### Reconstruction Results for Queries

Below are example reconstruction results for each query. These illustrate the reconstructed outputs for the corresponding queries:

- **Query: "Any jewelry?"**
  ![Reconstruction for Any Jewelry](./assets/Any_jewelry.jpg)

- **Query: "Any human faces?"**
  ![Reconstruction for Any Human Faces](./assets/Any_human_faces.jpg)

- **Query: "Any males with a beard?"**
  ![Reconstruction for Any Males with a Beard](./assets/Any_males_with_a_beard.jpg)

- **Query: "Any guns?"**
  ![Reconstruction for Any Guns](./assets/Any_guns.jpg)

- **Query: "Any females riding a horse?"** 
  ![Reconstruction for Any Females Riding a Horse](./assets/Any_females_riding_a_horse.jpg)

## Acknowledgement
We would like to acknowledge the repositories below.
* [breaching](https://github.com/JonasGeiping/breaching)
* [2023YeHFGradInv](https://github.com/MiLab-HITSZ/2023YeHFGradInv)