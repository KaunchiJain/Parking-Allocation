from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load('mobilenet_best_model1.pth'))
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_parking_slot(model, slot_image):
    slot_image = transform(slot_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(slot_image)
        _, predicted = torch.max(output, 1)
    return predicted.item()  # 0 for free, 1 for busy


def parse_image_metadata(image_path):
    parts = image_path.split(os.sep)
    weather = parts[-5]
    capture_date = parts[-3]
    camera_id = parts[-2].replace("camera", "")
    return weather, capture_date, int(camera_id)


def load_and_scale_bboxes(camera_id):
    csv_path = f"/Users/romeshjain/Desktop/DL Parking Allocation/Dataset/CNR-EXT_FULL_IMAGE_1000x750/camera{camera_id}.csv"
    bbox_df = pd.read_csv(csv_path)

    
    scale_x = 1000 / 2592
    scale_y = 750 / 1944
    bbox_df['X'] = (bbox_df['X'] * scale_x).astype(int)
    bbox_df['Y'] = (bbox_df['Y'] * scale_y).astype(int)
    bbox_df['W'] = (bbox_df['W'] * scale_x).astype(int)
    bbox_df['H'] = (bbox_df['H'] * scale_y).astype(int)
    
    # Calculate xmax and ymax
    bbox_df['xmax'] = bbox_df['X'] + bbox_df['W']
    bbox_df['ymax'] = bbox_df['Y'] + bbox_df['H']
    
    return bbox_df


def process_image(image_path, camera_id):
    
    weather, capture_date, camera_id = parse_image_metadata(image_path)
    print(f"Image Metadata -> Weather: {weather}, Capture Date: {capture_date}, Camera ID: {camera_id}")
    
    bbox_df = load_and_scale_bboxes(camera_id)
    full_image = Image.open(image_path).convert("RGB")
    

    for idx, row in bbox_df.iterrows():
        xmin, ymin, xmax, ymax = row['X'], row['Y'], row['xmax'], row['ymax']
        
        # Crop the parking slot region from full image
        parking_slot = full_image.crop((xmin, ymin, xmax, ymax))
        
        # Classify the slot
        if classify_parking_slot(model, parking_slot) == 0:  # 0 means "free"
            print(f"Recommend parking slot at location: {xmin, ymin, xmax, ymax}")
            
            draw = ImageDraw.Draw(full_image)  
            draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=5)  
            output_image_path = os.path.join("static", "output_image.jpg")
            full_image.save(output_image_path)
            return f"/uploads/{os.path.basename(output_image_path)}"  
    print("No free parking slots available.")
    return None

# Route for uploading and processing the image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image_path = request.form['image_path']
        camera_id = request.form['camera_id']
        output_image_path = process_image(image_path, camera_id)
        if output_image_path:
            return render_template('result.html', image_url=output_image_path)
        else:
            return "No free parking slots found."
    return render_template('index.html')

# Route to serve the processed image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
