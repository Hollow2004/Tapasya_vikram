from flask import Flask, flash, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_manager, login_required, logout_user, current_user, LoginManager
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import urllib.request
import os
import uuid
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import pipeline

#-------------------------------------------------------------------------------MODEL-------------------------------------------------------------------------
#------------------------------------TRANSFORMERS PIPELINES----------------------------------------------
pipe_76 = pipeline("image-classification", model="shreyasguha/22class_skindiseases_76acc_possibleoverfit")
pipe_57 = pipeline("image-classification", model="shreyasguha/22class_skindiseases_57acc")
pipe_80 = pipeline("image-classification", model="shreyasguha/22class_skindiseases_80acc")

categories = [
    "Seborrheic Keratoses and other Benign Tumors",
    "Vascular Tumors",
    "Light Diseases and Disorders of Pigmentation",
    "Vasculitis Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Nail Fungus and other Nail Disease",
    "Exanthems and Drug Eruptions",
    "Systemic Disease",
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Lupus and other Connective Tissue diseases",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Eczema Photos",
    "Warts Molluscum and other Viral Infections",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Bullous Disease Photos",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Atopic Dermatitis Photos",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Urticaria Hives",
    "Herpes HPV and other STDs Photos"
]

def preprocess(image_path, mix=False):
    image = Image.open(image_path)
    ans1 = pipe_57(image)
    ans2 = pipe_76(image)
    ans3 = pipe_80(image)

    op = [ans1, ans2, ans3]
    s = [ans[0]['label'] for ans in op]
    ansf = [int(element.replace("LABEL_", "")) for element in s]
    conditions = [categories[ans] for ans in ansf]

    if(conditions[0] == conditions[1] and conditions[1] == conditions[2]):
        return conditions[0]
    if((conditions[0] != conditions[1] and conditions[1] != conditions[2]) or mix==False):
        return conditions[1]
    else:
        if(conditions[0] == conditions[1]):
            return conditions[0]
        elif(conditions[0] == conditions[2]):
            return conditions[0]
        else:
            return conditions[1]


#------------------------------------PREVIOUS MODEL-----------------------------------
# disease_names = {
#     0: "melanoma",
#     1: "benign keratosis-like lesions",
#     2: "basal cell carcinoma",
#     3: "actinic keratoses",
#     4: "squamous cell carcinoma",
#     5: "dermatofibroma",
#     6: "vascular lesions",
#     7: "pigmented benign keratosis"
# }
# class ModelFunction(nn.Module):
#     def __init__(self):
#         super(ModelFunction, self).__init__()
#         self.model = nn.Sequential(
#             # Convolutional layers
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input_shape=(3, 28, 28) because PyTorch expects (channels, height, width)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(128),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(),
#             nn.Dropout(p=0.2),
#             nn.Linear(256 * 1 * 1, 128),  # Adjust the input size here based on the final output size of the conv layers
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.BatchNorm1d(32),
#             nn.Linear(32, 7),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         return self.model(x)

# model = ModelFunction()
# model.load_state_dict(torch.load('/Users/shreyasguha/Downloads/Shreyas/code/SIH_TAPASH/model/skind.pth', map_location=torch.device('cpu')))

# def preprocess(image_path):
#     img = Image.open(image_path)
#     data_transform = transforms.Compose([
#         transforms.Resize(size=(28, 28)),
#         transforms.ToTensor()
#     ])
#     transformed_image = data_transform(img)
#     transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
#     with torch.inference_mode():
#         model.eval()
#         outputs = model(transformed_image)
#         _, predicted = torch.max(outputs, 1)
#         ind = predicted[0].item()
#         maxi = find_max(outputs[0])
#         secondmaxi = find_second_highest(outputs[0])
#         if maxi.item() - secondmaxi.item() > 0.5:
#             return disease_names[ind]
#         return "Model is unsure, please see a doctor."
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

device = "cpu"

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'thisisasecretkey'
# Create uploads folder if it doesn't exist
upload_folder = os.path.join(os.getcwd(), 'static', 'uploads')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

db = SQLAlchemy(app)

bcrypt=Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view="login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

# Create the database tables within an application context
with app.app_context():
    db.create_all()

# UPLOAD_FOLDER = 'D:\\SIH_TAPASH\\SIH_TAPASH\\Penul\\static\\uploads'

class Registerform(FlaskForm):
    username=StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password=PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit=SubmitField("Register")

    def validate_username(self, username):
        existing_user_username=User.query.filter_by(username=username.data).first()
        if existing_user_username:
            raise ValidationError('That username is already taken')


class Loginform(FlaskForm):
    username=StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password=PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit=SubmitField("Login")

app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'webp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form=Loginform()
    if form.validate_on_submit():
        user=User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('index.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register',  methods=['GET', 'POST'])
def register():
    form=Registerform()

    if form.validate_on_submit():
        hashed_password=bcrypt.generate_password_hash(form.password.data)
        new_user=User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/tnc')
def tnc():
    return render_template('tnc.html')




def find_max(values):
    max_value = values[0]
    for value in values:
        if value.item() > max_value.item():
            max_value = value
    return max_value

def find_second_highest(values):
    highest_value = max(values, key=lambda x: x.item())
    second_highest_value = None
    for value in values:
        if value != highest_value and (second_highest_value is None or value.item() > second_highest_value.item()):
            second_highest_value = value
    return second_highest_value



@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Generate unique filename to avoid conflicts
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        flash('Image successfully uploaded and analyzed')

        # Call the preprocess function with the uploaded image path
        prediction = preprocess(file_path)

        return render_template('upload.html', filename=unique_filename, prediction=prediction)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif, webp')
        return redirect(request.url)

# @app.route('/display/<filename>')
# def display_image(filename):
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
