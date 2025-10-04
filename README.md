# 🏠 ArchTextGen - AI-Powered Architecture Image Generation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-green.svg)](https://huggingface.co/runwayml/stable-diffusion-v1-5)

An AI-powered text-to-image generation system specialized in creating architectural imagery. This project combines Stable Diffusion with LoRA fine-tuning and house style classification to generate high-quality architectural renderings from text descriptions.

## 🎬 Demo

Check out the demo video to see ArchTextGen in action:

![Demo](demo.mp4)

*Watch the full demo to see how ArchTextGen generates stunning architectural images from text prompts.*

## ✨ Features

- 🎨 **Text-to-Image Generation**: Generate architectural images from natural language descriptions
- 🏗️ **Multiple Architectural Styles**: Support for Contemporary, Modern, Traditional, Rustic, and Farmhouse styles
- 🔧 **LoRA Fine-tuning**: Custom LoRA adapters trained on architectural datasets for style-specific generation
- 🤖 **House Style Classification**: MobileNetV2-based classifier to identify architectural styles (91.2%+ accuracy)
- 🌐 **Web Interface**: Flask-based REST API with CORS support for easy integration
- 📊 **Data Pipeline**: Complete pipeline for crawling, preprocessing, and preparing architectural image datasets from ArchDaily
- 🖼️ **Batch Generation**: Generate multiple images in a single request
- ⚙️ **Customizable Parameters**: Control inference steps, guidance scale, image dimensions, and more

## 📁 Project Structure

```
ArchTextGen/
├── crawl_archdaily/          # Data collection and preprocessing
│   ├── caption_file/         # Image captions and metadata
│   │   ├── exterior/         # Exterior images by style
│   │   └── interior/         # Interior images by style
│   ├── card_links.json       # ArchDaily project links
│   └── experiment.ipynb      # Data exploration notebooks
│
├── model/                    # Model training and inference
│   ├── stable_diffusion_finetune/
│   │   ├── train.py          # Full model fine-tuning
│   │   ├── train_lora.py     # LoRA training script
│   │   └── sd_config.py      # Configuration parameters
│   │
│   ├── house_style_detection/
│   │   ├── mobilenet_model.py      # MobileNetV2 classifier
│   │   ├── train_model.py          # Training script
│   │   ├── predict.py              # Inference script
│   │   └── house_style_mobilenetv2_final.h5  # Trained model
│   │
│   ├── exterior_lora/        # LoRA weights for exterior styles
│   │   ├── contemporary_lora/
│   │   ├── modern_lora/
│   │   ├── rustic_lora/
│   │   ├── farmhouse_lora/
│   │   └── traditional_lora/
│   │
│   └── interior_lora/        # LoRA weights for interior styles
│
├── web/                      # Web application
│   └── backend/
│       ├── app.py            # Flask API server
│       ├── base_model_sd_1.5/      # Stable Diffusion v1.5 base model
│       └── generated_images/       # Output directory
│
├── requirements.txt          # Python dependencies
├── demo.mp4                 # Project demonstration video
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster generation)
- 8GB+ VRAM for Stable Diffusion (40GB+ recommended for training)
- Git
- Weights & Biases account (optional, for training monitoring)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/quan23w/ArchTextGen.git
cd ArchTextGen
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the base model**
   - Place the Stable Diffusion v1.5 model in `web/backend/base_model_sd_1.5/`
   - Or download from [Hugging Face](https://huggingface.co/runwayml/stable-diffusion-v1-5)

4. **Set up LoRA weights** (optional)
   - Pre-trained LoRA weights are available in `model/exterior_lora/` and `model/interior_lora/`
   - Copy the desired style folders to `web/backend/` for use with the API

### Running the Web API

1. **Navigate to the backend directory**
```bash
cd web/backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start the Flask server**
```bash
python app.py
```

3. **The API will be available at** `http://localhost:5000`

### API Usage

#### Generate Images

**Endpoint:** `POST /generate`

**Request Body:**
```json
{
  "prompt": "A modern two-story house with large glass windows, surrounded by trees",
  "style": "Exterior Modern",
  "sampling_steps": 30,
  "cfg_scale": 7.5,
  "width": 512,
  "height": 512,
  "format": "png"
}
```

**Response:**
```json
{
  "success": true,
  "images": [
    {
      "filename": "generated_20250104_123456_abc123_1.png",
      "filepath": "generated_images/generated_20250104_123456_abc123_1.png",
      "image_base64": "iVBORw0KGgoAAAANSUhEUgA..."
    }
  ],
  "prompt": "A modern two-story house...",
  "parameters": {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "style": "Exterior Modern"
  }
}
```

#### Available Styles

**Endpoint:** `GET /styles`

Returns list of available architectural styles:
- Exterior Contemporary
- Exterior Modern
- Exterior Traditional
- Exterior Rustic
- Exterior Farmhouse

#### Health Check

**Endpoint:** `GET /health`

Check API status and model availability.

## 🎓 Training Custom Models

### Training LoRA Adapters

1. **Prepare your dataset**
   - Organize images in folders by style
   - Create a metadata.jsonl file with image-caption pairs

2. **Configure training parameters**
```bash
cd model/stable_diffusion_finetune
```

3. **Set up Weights & Biases (optional)**
```bash
wandb login
# Enter your API key when prompted
```

4. **Train LoRA**
```bash
python train_lora.py \
  --pretrained_model_name_or_path="base_model_sd_1.5" \
  --train_data_dir="path/to/your/images" \
  --output_dir="output_lora" \
  --rank=16 \
  --learning_rate=1e-4 \
  --max_train_steps=3000 \
  --train_batch_size=4 \
  --report_to="wandb"
```

**Note**: Training was performed on NVIDIA A40 GPU with 40GB VRAM. Adjust batch size based on your GPU memory. Training progress and metrics are automatically logged to Weights & Biases for real-time monitoring.

### Training Style Classifier

1. **Prepare labeled dataset**
   - Create `labels.csv` with image paths and style labels
   - Organize images in the `all_images` directory

2. **Train the classifier**
```bash
cd model/house_style_detection
python train_model.py
```

3. **Evaluate and predict**
```bash
python predict.py
```

**Training Features:**
- Two-phase training (transfer learning + fine-tuning)
- Data augmentation (rotation, flip, zoom)
- Class balancing with weighted loss
- Achieves 91.2%+ accuracy on test set

## 📊 Dataset

The project includes scripts for crawling and processing architectural images from ArchDaily and Houzz:

- **Exterior Styles**: Contemporary, Modern, Traditional, Rustic, Farmhouse
- **Interior Styles**: Modern, Rustic, Traditional
- **Preprocessing**: Automatic resizing, padding, and filtering
- **Metadata**: JSONL format with image descriptions and captions

### Data Collection

```bash
cd crawl_archdaily
jupyter notebook experiment.ipynb
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Diffusers, Transformers
- **Model Architecture**: Stable Diffusion v1.5, MobileNetV2
- **Fine-tuning**: LoRA (Low-Rank Adaptation), PEFT
- **Training Infrastructure**: NVIDIA A40 GPU (40GB VRAM)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Web Framework**: Flask, Flask-CORS
- **Computer Vision**: OpenCV, Pillow
- **Acceleration**: xFormers, CUDA

## 📈 Model Performance

### Style Classifier
- **Architecture**: MobileNetV2 with custom classification head
- **Accuracy**: 91.2%+ on test set
- **Classes**: Farmhouse, Modern, Rustic
- **Input Size**: 224x224 RGB images

### Text-to-Image Generation
- **Base Model**: Stable Diffusion v1.5
- **LoRA Rank**: 16
- **Training Steps**: 3000+ per style
- **Training Hardware**: NVIDIA A40 GPU (40GB VRAM)
- **Training Monitoring**: Weights & Biases (wandb)
- **Inference Speed**: ~10-15 seconds per image (GPU)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) by Stability AI
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [ArchDaily](https://www.archdaily.com/), [Houzz](https://www.houzz.com/) for architectural inspiration
- MobileNetV2 architecture by Google

## 📧 Contact

Quan - [@quan23w](https://github.com/quan23w)

Project Link: [https://github.com/quan23w/ArchTextGen](https://github.com/quan23w/ArchTextGen)

---

<p align="center">Made with ❤️ for architectural AI generation</p>
