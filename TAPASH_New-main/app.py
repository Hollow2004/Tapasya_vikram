from flask import Flask, flash, request, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_required, logout_user, current_user, LoginManager
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import os
import uuid
from PIL import Image
from transformers import pipeline
import datetime
from datasets import load_dataset

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
    history = db.relationship('UserHistory', backref='user', lazy=True)

class UserHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_filename = db.Column(db.String(120), nullable=False)
    prediction = db.Column(db.String(120), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    symptoms = db.Column(db.Text)
    recommendation = db.Column(db.Text)
    follow_up_date = db.Column(db.DateTime)
    follow_up_completed = db.Column(db.Boolean, default=False)

# Load the healthcare dataset
try:
    print("Loading ChatDoctor dataset...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    ds = None

# Create the database tables within an application context
with app.app_context():
    db.create_all()

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





@app.route('/', methods=['POST'])
@login_required
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

        # Store image and prediction in user history
        user_id = current_user.id
        history = UserHistory(user_id=user_id, image_filename=unique_filename, prediction=prediction)
        db.session.add(history)
        db.session.commit()

        return render_template('upload.html', filename=unique_filename, prediction=prediction)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif, webp')
        return redirect(request.url)

def generate_custom_recommendations(prediction, itching, spreading, duration, additional_responses=None):
    """Generate professional recommendations using the dataset and user responses"""
    recommendations = []
    
    # Use the dataset to generate recommendations
    if ds and 'train' in ds:
        try:
            dataset_entries = ds['train']
            
            # Extract key terms from the prediction to search for relevant entries
            prediction_keywords = extract_condition_keywords(prediction)
            symptom_keywords = []
            
            # Add symptom-specific keywords
            if itching == 'yes':
                symptom_keywords.extend(['itch', 'scratch', 'antihistamine', 'irritation'])
            if spreading == 'yes':
                symptom_keywords.extend(['spread', 'contagious', 'infection', 'growing'])
            if duration in ['2-4_weeks', 'more_than_month']:
                symptom_keywords.extend(['chronic', 'persistent', 'long-term', 'ongoing'])
            
            relevant_entries = []
            
            # Search through dataset for relevant Q&A pairs
            for i, entry in enumerate(dataset_entries):
                if i > 5000:  # Increased limit for better coverage
                    break
                    
                instruction = entry.get('instruction', '').lower()
                output = entry.get('output', '')
                
                # Check if entry is relevant to the condition or symptoms
                relevance_score = 0
                
                # Check for condition-specific keywords
                for keyword in prediction_keywords:
                    if keyword in instruction:
                        relevance_score += 3
                    if keyword in output.lower():
                        relevance_score += 2
                
                # Check for symptom keywords
                for keyword in symptom_keywords:
                    if keyword in instruction or keyword in output.lower():
                        relevance_score += 1
                
                # If relevant, add to list
                if relevance_score >= 3:  # Threshold for relevance
                    relevant_entries.append({
                        'instruction': instruction,
                        'output': output,
                        'score': relevance_score
                    })
            
            # Sort by relevance score
            relevant_entries.sort(key=lambda x: x['score'], reverse=True)
            
            # Extract recommendations from the most relevant entries
            for entry in relevant_entries[:10]:  # Top 10 most relevant
                output_text = entry['output']
                
                # Extract actionable recommendations from the output
                extracted_recs = extract_recommendations_from_text(output_text)
                
                for rec in extracted_recs:
                    cleaned_rec = clean_medical_response(rec)
                    if cleaned_rec and len(cleaned_rec) > 30:
                        recommendations.append(cleaned_rec)
                    
                    if len(recommendations) >= 5:
                        break
                
                if len(recommendations) >= 5:
                    break
            
            # Remove duplicates and select best recommendations
            if recommendations:
                unique_recommendations = list(set(recommendations))
                recommendations = select_best_recommendations(unique_recommendations, prediction, itching, spreading, duration)
            
        except Exception as e:
            print(f"Error processing dataset for recommendations: {e}")
    
    # Fallback professional recommendations if dataset processing fails or insufficient recommendations
    if len(recommendations) < 3:
        fallback_recs = generate_fallback_recommendations(prediction, itching, spreading, duration)
        for rec in fallback_recs:
            if rec not in recommendations:
                recommendations.append(rec)
    
    return recommendations[:3]  # Return top 3 recommendations

def extract_condition_keywords(prediction):
    """Extract relevant keywords from the condition prediction"""
    keywords = []
    prediction_lower = prediction.lower()
    
    # Map conditions to relevant search keywords
    condition_keyword_map = {
        'seborrheic keratoses': ['seborrheic', 'keratosis', 'benign tumor', 'skin growth'],
        'vascular': ['vascular', 'blood vessel', 'hemangioma', 'angioma'],
        'pigmentation': ['pigmentation', 'hyperpigmentation', 'melasma', 'dark spots'],
        'vasculitis': ['vasculitis', 'inflammation', 'blood vessel inflammation'],
        'cellulitis': ['cellulitis', 'bacterial infection', 'skin infection', 'erysipelas'],
        'impetigo': ['impetigo', 'bacterial', 'contagious', 'crusted'],
        'tinea': ['tinea', 'ringworm', 'fungal', 'dermatophyte'],
        'candidiasis': ['candida', 'yeast', 'fungal infection', 'thrush'],
        'nail fungus': ['nail fungus', 'onychomycosis', 'fungal nail'],
        'drug eruption': ['drug reaction', 'medication rash', 'allergic reaction'],
        'acne': ['acne', 'pimples', 'comedones', 'breakout'],
        'rosacea': ['rosacea', 'facial redness', 'flushing'],
        'actinic keratosis': ['actinic keratosis', 'solar keratosis', 'precancerous'],
        'basal cell': ['basal cell', 'carcinoma', 'skin cancer', 'bcc'],
        'lupus': ['lupus', 'autoimmune', 'butterfly rash', 'sle'],
        'alopecia': ['alopecia', 'hair loss', 'baldness', 'hair fall'],
        'melanoma': ['melanoma', 'malignant', 'skin cancer', 'mole cancer'],
        'eczema': ['eczema', 'atopic dermatitis', 'dermatitis', 'skin inflammation'],
        'warts': ['wart', 'verruca', 'hpv', 'viral growth'],
        'molluscum': ['molluscum', 'contagiosum', 'viral infection'],
        'scabies': ['scabies', 'mites', 'itchy rash', 'contagious'],
        'lyme': ['lyme disease', 'tick bite', 'bullseye rash'],
        'bullous': ['bullous', 'blister', 'pemphigus', 'pemphigoid'],
        'poison ivy': ['contact dermatitis', 'allergic contact', 'poison ivy', 'allergen'],
        'psoriasis': ['psoriasis', 'scaly patches', 'chronic skin', 'plaque'],
        'lichen planus': ['lichen planus', 'purple patches', 'inflammatory'],
        'urticaria': ['urticaria', 'hives', 'wheals', 'allergic'],
        'herpes': ['herpes', 'hsv', 'blister', 'viral infection'],
        'hpv': ['hpv', 'human papillomavirus', 'wart virus'],
        'std': ['sexually transmitted', 'std', 'sti']
    }
    
    # Find matching keywords
    for condition, kw_list in condition_keyword_map.items():
        if condition in prediction_lower:
            keywords.extend(kw_list)
    
    # Also add words from the prediction itself
    important_words = prediction.split()
    for word in important_words:
        if len(word) > 4 and word.lower() not in ['photos', 'other', 'diseases', 'disorders']:
            keywords.append(word.lower())
    
    return list(set(keywords))  # Remove duplicates

def extract_recommendations_from_text(text):
    """Extract actionable recommendations from dataset output text"""
    recommendations = []
    
    if not text:
        return recommendations
    
    # Split text into sentences
    sentences = text.replace('\n', '. ').split('.')
    
    # Keywords that indicate actionable recommendations
    action_keywords = [
        'recommend', 'suggest', 'advise', 'should', 'must', 'need to',
        'apply', 'use', 'take', 'avoid', 'consult', 'see', 'visit',
        'maintain', 'keep', 'try', 'consider', 'important to', 'beneficial',
        'help', 'treat', 'prevent', 'reduce', 'manage'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()
        
        # Check if sentence contains actionable advice
        if any(keyword in sentence_lower for keyword in action_keywords):
            # Skip very short or very long sentences
            if 20 < len(sentence) < 200:
                # Skip sentences that are questions or greetings
                if not any(skip in sentence_lower for skip in ['?', 'hello', 'hi,', 'dear', 'thank']):
                    recommendations.append(sentence)
    
    return recommendations

def clean_medical_response(text):
    """Clean medical responses by removing references to ChatDoctor, personal names, etc."""
    if not text:
        return ""
    
    # Remove common problematic phrases
    remove_phrases = [
        'chat doctor', 'chatdoctor', 'thank you for contacting', 'thanks for your question',
        'i can understand', 'by your history', 'hi,', 'hello,', 'dear patient',
        'i understand your concern', 'i hope this helps', 'feel free to ask',
        'get back to me', 'wish you good health'
    ]
    
    cleaned = text.lower()
    for phrase in remove_phrases:
        cleaned = cleaned.replace(phrase, '')
    
    # Remove sentences that start with personal pronouns
    sentences = cleaned.split('.')
    professional_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and not sentence.startswith(('i ', 'you ', 'your ', 'my ', 'we ')):
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            professional_sentences.append(sentence)
    
    result = '. '.join(professional_sentences).strip()
    if result and not result.endswith('.'):
        result += '.'
    
    return result

def select_best_recommendations(recommendations, prediction, itching, spreading, duration):
    """Select the most appropriate recommendations based on symptoms"""
    scored_recommendations = []
    
    for rec in recommendations:
        score = 0
        rec_lower = rec.lower()
        
        # Score based on symptom relevance
        if itching == 'yes' and any(word in rec_lower for word in ['itch', 'scratch', 'antihistamine']):
            score += 2
        if spreading == 'yes' and any(word in rec_lower for word in ['spread', 'infection', 'doctor']):
            score += 2
        if duration in ['2-4_weeks', 'more_than_month'] and any(word in rec_lower for word in ['chronic', 'persistent', 'specialist']):
            score += 2
        
        # Score based on condition type
        if 'acne' in prediction.lower() and any(word in rec_lower for word in ['acne', 'comedogenic', 'facial']):
            score += 1
        if 'eczema' in prediction.lower() and any(word in rec_lower for word in ['moistur', 'dry', 'trigger']):
            score += 1
        if 'melanoma' in prediction.lower() and any(word in rec_lower for word in ['urgent', 'oncolog', 'biopsy']):
            score += 3
        
        scored_recommendations.append((score, rec))
    
    # Sort by score and return top 3
    scored_recommendations.sort(key=lambda x: x[0], reverse=True)
    return [rec for score, rec in scored_recommendations[:3]]

def generate_fallback_recommendations(prediction, itching, spreading, duration):
    """Generate professional fallback recommendations"""
    recommendations = []
    
    # Base recommendations based on symptoms
    if spreading == 'yes' or duration in ['2-4_weeks', 'more_than_month']:
        recommendations.append("Consult a dermatologist for comprehensive evaluation and appropriate treatment planning.")
        recommendations.append("Maintain detailed documentation of symptom progression with photographs for medical review.")
    elif itching == 'yes':
        recommendations.append("Apply cool compresses to affected areas to reduce inflammation and discomfort.")
        recommendations.append("Use gentle, fragrance-free moisturizers to maintain skin barrier function.")
    else:
        recommendations.append("Continue monitoring the condition while maintaining proper skin hygiene practices.")
        recommendations.append("Protect affected areas from excessive sun exposure and environmental irritants.")
    
    # Condition-specific recommendations
    if 'acne' in prediction.lower():
        recommendations.append("Use non-comedogenic skincare products and avoid over-cleansing the affected areas.")
    elif 'eczema' in prediction.lower():
        recommendations.append("Identify and avoid potential triggers while maintaining consistent moisturizing routine.")
    elif any(word in prediction.lower() for word in ['melanoma', 'carcinoma', 'malignant']):
        recommendations.append("URGENT: Seek immediate consultation with an oncologist or dermatologist for further evaluation.")
    
    return recommendations[:3]

@app.route('/process-symptoms', methods=['POST'])
@login_required
def process_symptoms():
    prediction = request.form.get('prediction')
    filename = request.form.get('filename')
    itching = request.form.get('itching')
    spreading = request.form.get('spreading')
    duration = request.form.get('duration')
    
    # Process symptoms and generate custom recommendations using the dataset
    symptoms = f"Itching: {itching}, Spreading: {spreading}, Duration: {duration}"
    recommendations = generate_custom_recommendations(prediction, itching, spreading, duration)
    recommendation_text = '; '.join(recommendations)
    
    # Update the most recent history record for this user
    history = UserHistory.query.filter_by(user_id=current_user.id, image_filename=filename).first()
    if history:
        history.symptoms = symptoms
        history.recommendation = recommendation_text
        db.session.commit()
    
    flash('Symptoms processed successfully!')
    return render_template('upload.html', filename=filename, prediction=prediction, 
                         symptoms=symptoms, recommendation=recommendation_text)

# Step 6: Schedule Follow-up
@app.route('/schedule-follow-up', methods=['POST'])
@login_required
def schedule_follow_up_route():
    filename = request.form.get('filename')
    schedule_follow_up(current_user.id, filename, days=5)
    return redirect(url_for('upload'))

def schedule_follow_up(user_id, image_filename, days=3):
    """Schedule a follow-up after a set number of days"""
    follow_up_date = datetime.datetime.utcnow() + datetime.timedelta(days=days)
    history = UserHistory.query.filter_by(user_id=user_id, image_filename=image_filename).first()
    if history:
        history.follow_up_date = follow_up_date
        db.session.commit()

    flash('Follow-up scheduled successfully! Check your reminders.')


# Step 7: Monitor Progress
@app.route('/monitor-progress', methods=['POST'])
@login_required
def monitor_progress():
    # This function could involve image similarity analysis or symptom updates
    # For simplicity, we will just update the status based on fixed conditions
    filename = request.form.get('filename')
    progress_update = request.form.get('progress_update')
    history = UserHistory.query.filter_by(user_id=current_user.id, image_filename=filename).first()

    if history:
        if progress_update == 'worsening':
            history.follow_up_completed = True
            db.session.commit()
            flash('Your condition seems to be worsening. A consultation is recommended.')
        else:
            flash('Progress monitored successfully. No change detected.')
    return redirect(url_for('upload'))

# Step 8: Replan or Escalate
@app.route('/escalate-care', methods=['POST'])
@login_required
def escalate_care():
    filename = request.form.get('filename')
    history = UserHistory.query.filter_by(user_id=current_user.id, image_filename=filename).first()

    if history:
        # Follow-up on care plan
        if not history.follow_up_completed:
            flash('Follow up on care is not yet completed.')
        else:
            flash('Consult your doctor for aggressive intervention.')

    return redirect(url_for('upload'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
