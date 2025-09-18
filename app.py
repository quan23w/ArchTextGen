from flask import Flask, request, jsonify, send_file
from flask import send_from_directory
from flask_cors import CORS
import torch
from diffusers import StableDiffusionPipeline
import io
import base64
from PIL import Image
import os
import uuid
from datetime import datetime
import zipfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store the pipeline
pipe = None

def load_model():
    """Load the Stable Diffusion model"""
    global pipe
    if pipe is None:
        base_model = "base_model_sd_1.5"
        print("Loading Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded on device: {pipe.device}")
    return pipe

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": pipe is not None,
        "cuda_available": torch.cuda.is_available()
    })

@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image from text prompt"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "No prompt provided"}), 400

        prompt = data['prompt']
        # Map frontend fields to backend variables
        guidance_scale = data.get('cfg_scale')
        num_inference_steps = data.get('sampling_steps')
        width = data.get('width')
        height = data.get('height')

        # Load model if not already loaded
        pipeline = load_model()

        

        # Generate images
        print(f"Generating 4 images with prompt: {prompt}")
        result = pipeline(
            prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            num_images_per_prompt=4
        )
        images = result.images

        # Get format (png or jpeg)
        img_format = data.get('format', 'png').lower()
        if img_format not in ['png', 'jpeg', 'jpg']:
            img_format = 'png'
        ext = 'jpg' if img_format in ['jpeg', 'jpg'] else 'png'

        # Save all images and convert to base64
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Create output directory if it doesn't exist
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)

        images_data = []
        for i, image in enumerate(images):
            # Save image with unique filename
            filename = f"generated_{timestamp}_{unique_id}_{i+1}.{ext}"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath, format=img_format.upper() if img_format != 'jpg' else 'JPEG')

            # Convert image to base64
            img_io = io.BytesIO()
            image.save(img_io, format=img_format.upper() if img_format != 'jpg' else 'JPEG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
            
            images_data.append({
                "filename": filename,
                "filepath": filepath,
                "image_base64": img_base64
            })

        return jsonify({
            "success": True,
            "images": images_data,
            "prompt": prompt,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "format": img_format,
                "num_images_per_prompt": 4
            }
        })
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate-file', methods=['POST'])
def generate_image_file():
    """Generate image and return as file"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "No prompt provided"}), 400

        prompt = data['prompt']
        # Map frontend fields to backend variables
        guidance_scale = data.get('cfg_scale', 14.0)
        num_inference_steps = data.get('sampling_steps', 51)
        controlnet_image = data.get('controlnet_image', None)  # Not used in current pipeline
        width = data.get('width', 512)
        height = data.get('height', 512)

        # Load model if not already loaded
        pipeline = load_model()

    

        # Generate images
        print(f"Generating 4 images with prompt: {prompt}")
        result = pipeline(
            prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images_per_prompt=4
        )
        images = result.images

        # Get format (png or jpeg)
        img_format = data.get('format', 'png').lower()
        if img_format not in ['png', 'jpeg', 'jpg']:
            img_format = 'png'
        ext = 'jpg' if img_format in ['jpeg', 'jpg'] else 'png'

        # Create a zip file with all images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        zip_filename = f"generated_images_{timestamp}_{unique_id}.zip"

        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, image in enumerate(images):
                # Convert image to bytes
                img_io = io.BytesIO()
                image.save(img_io, format=img_format.upper() if img_format != 'jpg' else 'JPEG')
                img_bytes = img_io.getvalue()
                
                # Add to zip with filename
                image_filename = f"generated_image_{i+1}.{ext}"
                zip_file.writestr(image_filename, img_bytes)
        
        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if pipe is None:
            return jsonify({"error": "Model not loaded"}), 400
        
        return jsonify({
            "model_loaded": True,
            "device": str(pipe.device),
            "model_type": "Stable Diffusion 1.5",
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def root():
    """Serve the index.html file at root."""
    return send_from_directory('.', 'index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve a blank favicon or return 204 if not present."""
    if os.path.exists('favicon.ico'):
        return send_from_directory('.', 'favicon.ico')
    else:
        return ('', 204)

if __name__ == '__main__':
    print("Starting Flask API server...")
    print("Endpoints available:")
    print("  POST /generate - Generate image and return JSON with base64")
    print("  POST /generate-file - Generate image and return as file")
    print("  GET /health - Health check")
    print("  GET /model-info - Model information")
    
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
