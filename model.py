from ultralytics import YOLO
import cv2
import pytesseract
import os
import google.generativeai as genai
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load the YOLOv10 pre-trained model (update when available)
yolo_model = YOLO('yolov10n.pt')  # Note: YOLOv10 is not yet available, use the latest version

# Set up the Google API Key for Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBNSUOu5kSC9dktdLUB-chXguVAqPusimk') 

# Initialize the GenerativeModel with the API key and model name
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro') 

# Initialize SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Load CLIP model and processor
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = yolo_model(img)
    detected_objects = [result.names[int(cls)] for result in results for cls in result.boxes.cls]
    return detected_objects

def extract_text(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray_img)
    return extracted_text.strip()

def segment_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    
    # Generate automatic masks
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=True,
    )
    
    # Count unique segments
    num_segments = len(masks)
    
    # Create a colored segmentation map
    segmentation_map = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, mask in enumerate(masks):
        segmentation_map[mask] = (i + 1) % 256 

    # Visualize segmentation map using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.imshow(segmentation_map, alpha=0.5) 
    plt.title('Segmentation Map')
    plt.axis('off')
    plt.show() 

    return num_segments, segmentation_map

def generate_clip_description(image_path):
    image = Image.open(image_path)
    inputs = processor_clip(text=["a photo of"], images=image, return_tensors="pt", padding=True)
    outputs = model_clip(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    return probs  # Return the probabilities for analysis

def generate_response(query, context):
    system_prompt = """
    You are an AI assistant with the ability to analyze images. You will be given information about an image, which may include detected objects, extracted text, segmentation details, or a combination of these. Treat this information as if you are directly looking at the image. Respond to queries about the image based on this context. Be descriptive and confident in your answers, as if you can see the image yourself. Keep the answer as informative as possible. If you don't understand anything or lack certain information, focus only on what you know for sure.
    """

    combined_input = f"{system_prompt}\n\nImage Context: {context}\n\nUser Query: {query}\n\nResponse:"

    response = gemini_model.generate_content(combined_input)
    return response.text.strip() if response else "No response received"

def process_image(image_path):
    detected_objects = detect_objects(image_path)
    extracted_text = extract_text(image_path)
    num_segments, _ = segment_image(image_path)
    clip_probs = generate_clip_description(image_path) 

    context = ""
    if detected_objects:
        context += f"Detected Objects: {', '.join(detected_objects)}. "
    else:
        context += "No objects detected. "

    if extracted_text:
        context += f"Extracted Text: {extracted_text}. "
    else:
        context += "No text extracted. "

    context += f"Number of segments: {num_segments}. "
    context += f"CLIP Probabilities: {clip_probs}. "  # Add CLIP probabilities

    return context.strip()

def main():
    image_path = input("Enter the path to your image: ")
    context = process_image(image_path)

    print("\nImage Context:")
    print(context)
    print("\nYou can now ask multiple questions about the image.")

    while True:
        user_query = input("\nWhat would you like to know? (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        response = generate_response(user_query, context)
        print("Response:", response)

if __name__ == "__main__":
    main() 