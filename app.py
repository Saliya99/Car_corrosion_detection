from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from model import load_model, preprocess_image

app = Flask(__name__)

# Load your model
model = load_model()

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")  # Serve the HTML form

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        try:
            image_tensor = preprocess_image(file_path)
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted_class = torch.max(output, 1)
                label = "Coastal" if predicted_class.item() == 0 else "Inland"
            return jsonify({"label": label})
        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid file format"})

if __name__ == "__main__":
    app.run(debug=True)
