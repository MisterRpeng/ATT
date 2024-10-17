# Supplementary Materials for "Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning"

## Dependencies

- Python 3.6.13
- Pytorch 1.7.1
- Torchvision 0.8.2
- Numpy 1.19.5
- Pillow 8.4.0
- Timm 0.5.4
- Scipy 1.5.4

## Usage Instructions

##### 1. Models and Datasets

ViT models are all available in [timm](https://github.com/huggingface/pytorch-image-models) library. We consider four surrogate models (vit_base_patch16_224, pit_b_224, cait_s24_224, and visformer_small) and four additional target models (deit_base_distilled_patch16_224, levit_256, convit_base, tnt_s_patch16_224).

The dataset can be downloaded from the link https://github.com/jpzhang1810/TGR, which is provided by TGR's authors, and all the image files should be placed in the directory `. /clean_resized_images`.

To evaluate the transferability of adversarial examples on CNN models, please download the converted pretrained models from the link https://github.com/ylhz/tf_to_pytorch_model before running the code, and these model checkpoint files should be placed in the directory `./models`.

##### 2. Source Code

- `methods.py` : the code of the implementation for our proposed Adaptive Token Tuning (ATT) attack method.

- `evaluate.py` : the code for evaluating the transferability of generated adversarial examples on different ViT models.

- `evaluate_cnn.py` : the code for evaluating the transferability of generated adversarial examples on different CNN models.

##### 3. Quickstart

- You also can use the following command to get all our best results directly!

```
bash train_and_test_all.sh
```

##### 4. Training

- Generate adversarial examples via the proposed ATT attack on ViT models. You can also modify the hyperparameters to match the experimental settings in our paper.

```
python attack.py --attack ATT --source_model vit_base_patch16_224
```

##### 5. Testing

- Evaluate the transferability of adversarial examples on ViT models

```
bash evaluate_ViTs.sh vit_base_patch16_224
```

- Evaluate the transferability of adversarial examples on CNN models

```
python evaluate_cnn.py --source_model vit_base_patch16_224
```

#### 5. Running Memory and Time

GPU memory usage: 4GB.

Time: average 1.8s per image with RTX 3090.

## Acknowledgments

The code refer to: [TGR](https://github.com/jpzhang1810/TGR) and [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model).

We thanks the authors for sharing sincerely.
