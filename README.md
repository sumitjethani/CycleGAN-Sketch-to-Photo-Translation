# CycleGAN — Sketch ↔ Photo Translation

An unpaired image-to-image translation project using **CycleGAN** to translate between sketch and photo domains in both directions — Sketch → Photo and Photo → Sketch.

---

## Demo

🤗 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/Sumit-Jethani/CycleGAN-Sketch-to-Photo-Translation)

---

## Model Architecture

### Generator (ResNet-based)
- ReflectionPad + Conv7x7 initial layer
- 2 Downsampling Conv layers
- 6 Residual Blocks
- 2 Upsampling ConvTranspose layers
- ReflectionPad + Conv7x7 output layer with Tanh

### Discriminator (PatchGAN)
- 5 Convolution layers
- InstanceNorm + LeakyReLU
- Patch-level real/fake classification

### CycleGAN Components
- **G_AB** — Sketch → Photo Generator
- **G_BA** — Photo → Sketch Generator
- **D_A** — Sketch Discriminator
- **D_B** — Photo Discriminator

---

## Training Details

| Parameter | Value |
|---|---|
| Dataset | Sketch-to-Image (256×256) |
| Image Size | 128×128 |
| Batch Size | 8 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Betas | (0.5, 0.999) |
| Lambda Cycle | 10.0 |
| Lambda Identity | 5.0 |

---

## Loss Functions

- **GAN Loss** — MSE Loss for adversarial training
- **Cycle Consistency Loss** — L1 Loss (λ=10) to preserve structure
- **Identity Loss** — L1 Loss (λ=5) to preserve color/style

---

## Download Model

```bash
wget https://huggingface.co/spaces/Sumit-Jethani/CycleGAN-Sketch-to-Photo-Translation/resolve/main/cyclegan_final_generators.pth
```

---

## Project Structure

```
├── app.py                          # Gradio app for HuggingFace deployment
├── requirements.txt                # Dependencies
├── cyclegan_final_generators.pth   # Pretrained model weights (G_AB + G_BA)
└── CycleGAN-Sketch-to-Photo-Translation.ipynb                        # Training notebook
```

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/sumitjethani/CycleGAN-Sketch-to-Photo-Translation.git
cd CycleGAN-Sketch-to-Photo-Translation

# Install dependencies
pip install -r requirements.txt

# Download model
wget https://huggingface.co/spaces/Sumit-Jethani/CycleGAN-Sketch-to-Photo-Translation/resolve/main/cyclegan_final_generators.pth

# Run the app
python app.py
```

---

## Requirements

```
torch==2.1.0
torchvision==0.16.0
gradio==4.44.0
Pillow==10.3.0
numpy==1.26.4
```

---

## Dataset

[Sketch to Image Dataset](https://www.kaggle.com/datasets/ankitsheoran23/sketch-to-image) — Kaggle

---

## Author

**Sumit Jethani**
- GitHub: [github.com/sumitjethani](https://github.com/sumitjethani)
- LinkedIn: [linkedin.com/in/sumit-jethani](https://linkedin.com/in/sumit-jethani)
- HuggingFace: [huggingface.co/Sumit-Jethani](https://huggingface.co/Sumit-Jethani)
