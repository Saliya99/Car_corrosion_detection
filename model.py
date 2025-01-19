import os
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from collections import OrderedDict


class_labels = ['Coastal', 'Inland']

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best_model.pth"

model = EfficientNet.from_name('efficientnet-b7')

num_ftrs = model._fc.in_features
model._fc = torch.nn.Linear(num_ftrs, len(class_labels))

checkpoint = torch.load(model_path, map_location=device)

state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def preprocess_image(image_path):
    """Preprocess the input image."""
    image = Image.open(image_path).convert('RGB') 
    image = transform(image).unsqueeze(0) 
    return image.to(device)

def predict(image_path):
    """
    Predict the class of the given image.

    Args:
        image_path (str): Path to the image to predict.
    Returns:
        str: Predicted class label.
        torch.Tensor: Probabilities for each class.
    """
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return class_labels[predicted_class], probabilities.squeeze().tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        predicted_label, probabilities = predict(file_path)

        os.remove(file_path)

        return jsonify({
            'predicted_class': predicted_label,
            'probabilities': probabilities
        })

if __name__ == '__main__':
    app.run(debug=True)
