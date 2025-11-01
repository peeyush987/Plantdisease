import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from deep_translator import GoogleTranslator

# Page Configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558b2f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-card {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #66bb6a;
        margin: 10px 0;
    }
    .cure-card {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #42a5f5;
        margin: 10px 0;
    }
    .hindi-card {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #4caf50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .language-toggle {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize translator
def translate_to_hindi(text):
    try:
        translation = GoogleTranslator(source='en', target='hi').translate(text)
        return translation
    except:
        return text

# Session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Load model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Disease Information Database
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis. It causes dark, scabby lesions on leaves, fruit, and twigs, leading to premature leaf drop and reduced fruit quality.",
        "symptoms": [
            "Olive-green to brown spots on leaves",
            "Velvety lesions on fruit",
            "Premature leaf drop",
            "Cracked and distorted fruit"
        ],
        "cure": [
            "Remove and destroy infected leaves and fruit",
            "Apply fungicides containing captan or myclobutanil",
            "Prune trees to improve air circulation",
            "Plant resistant apple varieties",
            "Apply fungicides in early spring before bud break"
        ]
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa. It affects leaves, fruit, and bark, causing significant damage to apple trees.",
        "symptoms": [
            "Purple spots on leaves that turn brown",
            "Black, sunken lesions on fruit",
            "Mummified fruit remains on tree",
            "Cankers on branches"
        ],
        "cure": [
            "Prune out dead and diseased branches",
            "Remove mummified fruit from tree and ground",
            "Apply fungicides like captan or thiophanate-methyl",
            "Maintain good tree hygiene",
            "Ensure proper drainage around trees"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "name": "Cedar Apple Rust",
        "description": "Cedar apple rust is caused by Gymnosporangium juniperi-virginianae. It requires both apple and cedar trees to complete its life cycle.",
        "symptoms": [
            "Yellow-orange spots on upper leaf surface",
            "Small, raised, orange lesions on fruit",
            "Premature leaf drop",
            "Reduced fruit quality"
        ],
        "cure": [
            "Remove nearby cedar trees if possible",
            "Apply fungicides containing myclobutanil",
            "Plant resistant apple varieties",
            "Rake and destroy fallen leaves",
            "Apply fungicides from bud break to 4 weeks after petal fall"
        ]
    },
    "Apple___healthy": {
        "name": "Healthy Apple Plant",
        "description": "Your apple plant appears healthy with no signs of disease. Continue maintaining good practices.",
        "symptoms": [
            "Vibrant green leaves",
            "No spots or discoloration",
            "Healthy fruit development"
        ],
        "cure": [
            "Continue regular watering schedule",
            "Apply balanced fertilizer as needed",
            "Prune regularly for air circulation",
            "Monitor for early signs of disease",
            "Maintain mulch around base of tree"
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Corn Gray Leaf Spot",
        "description": "Gray leaf spot is caused by the fungus Cercospora zeae-maydis. It thrives in warm, humid conditions and can significantly reduce yield.",
        "symptoms": [
            "Rectangular gray-brown lesions on leaves",
            "Lesions parallel to leaf veins",
            "Premature leaf death",
            "Reduced photosynthesis"
        ],
        "cure": [
            "Plant resistant corn hybrids",
            "Rotate crops with non-host plants",
            "Apply fungicides containing strobilurins",
            "Till crop residue into soil",
            "Avoid overhead irrigation"
        ]
    },
    "Corn_(maize)___Common_rust": {
        "name": "Corn Common Rust",
        "description": "Common rust is caused by Puccinia sorghi. It appears as small, circular to elongated pustules on leaves.",
        "symptoms": [
            "Reddish-brown pustules on both leaf surfaces",
            "Pustules release rust-colored spores",
            "Yellowing of leaves",
            "Reduced plant vigor"
        ],
        "cure": [
            "Plant resistant corn varieties",
            "Apply fungicides if severe",
            "Scout fields regularly",
            "Remove volunteer corn plants",
            "Ensure adequate plant nutrition"
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Northern Corn Leaf Blight",
        "description": "Northern leaf blight is caused by Exserohilum turcicum. It can cause significant yield losses in susceptible corn varieties.",
        "symptoms": [
            "Long, elliptical gray-green lesions on leaves",
            "Lesions may span the entire leaf width",
            "Premature leaf death",
            "Reduced grain fill"
        ],
        "cure": [
            "Plant resistant hybrids",
            "Rotate crops for 2-3 years",
            "Apply fungicides at early infection stages",
            "Bury crop debris through deep tillage",
            "Scout fields early and often"
        ]
    },
    "Corn_(maize)___healthy": {
        "name": "Healthy Corn Plant",
        "description": "Your corn plant is healthy with no visible disease symptoms. Maintain current management practices.",
        "symptoms": [
            "Dark green, vigorous leaves",
            "No lesions or spots",
            "Good ear development"
        ],
        "cure": [
            "Maintain adequate soil moisture",
            "Apply nitrogen fertilizer as recommended",
            "Control weeds effectively",
            "Monitor for pests and diseases",
            "Ensure proper plant spacing"
        ]
    },
    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "description": "Black rot is caused by Guignardia bidwellii. It's one of the most serious diseases of grapes in humid climates.",
        "symptoms": [
            "Circular tan spots on leaves with dark borders",
            "Black, shriveled, mummified berries",
            "Brown lesions on shoots",
            "Infected fruit drops prematurely"
        ],
        "cure": [
            "Remove and destroy mummified fruit",
            "Prune to improve air circulation",
            "Apply fungicides from bud break through harvest",
            "Remove wild grape vines nearby",
            "Maintain good canopy management"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "name": "Grape Esca (Black Measles)",
        "description": "Esca is a complex disease involving multiple fungi. It affects the wood and vascular system of grapevines.",
        "symptoms": [
            "Tiger stripe pattern on leaves",
            "Sudden wilting of shoots (apoplexy)",
            "Dark streaking in wood",
            "Berry spots and shriveling"
        ],
        "cure": [
            "Remove and destroy infected wood",
            "Avoid wounding during pruning",
            "Delay pruning until late winter",
            "Apply wound protectants after pruning",
            "Currently no effective chemical control"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Grape Leaf Blight",
        "description": "Leaf blight is caused by Pseudocercospora vitis. It causes leaf spots and premature defoliation.",
        "symptoms": [
            "Angular brown spots on leaves",
            "Spots may have yellow halos",
            "Premature leaf drop",
            "Reduced photosynthesis"
        ],
        "cure": [
            "Remove infected leaves",
            "Apply copper-based fungicides",
            "Improve air circulation through pruning",
            "Avoid overhead irrigation",
            "Maintain balanced nutrition"
        ]
    },
    "Grape___healthy": {
        "name": "Healthy Grape Plant",
        "description": "Your grapevine is healthy with no disease symptoms present. Continue good vineyard management.",
        "symptoms": [
            "Lush green foliage",
            "No leaf spots or discoloration",
            "Healthy fruit clusters"
        ],
        "cure": [
            "Continue regular irrigation",
            "Maintain proper canopy management",
            "Apply balanced fertilizers",
            "Monitor for early disease signs",
            "Ensure good air circulation"
        ]
    },
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "description": "Early blight is caused by Alternaria solani. It affects leaves, stems, and tubers of potato plants.",
        "symptoms": [
            "Dark brown spots with concentric rings (target pattern)",
            "Yellowing around lesions",
            "Premature leaf drop",
            "Lesions on stems and tubers"
        ],
        "cure": [
            "Plant resistant varieties",
            "Apply fungicides containing chlorothalonil",
            "Remove infected plant debris",
            "Rotate crops with non-solanaceous plants",
            "Ensure adequate plant spacing"
        ]
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "description": "Late blight is caused by Phytophthora infestans. It's the same pathogen that caused the Irish potato famine.",
        "symptoms": [
            "Water-soaked lesions on leaves",
            "White fungal growth on undersides of leaves",
            "Blackened stems",
            "Brown, firm rot on tubers"
        ],
        "cure": [
            "Apply fungicides preventatively (mancozeb, chlorothalonil)",
            "Destroy infected plants immediately",
            "Plant certified disease-free seed potatoes",
            "Avoid overhead irrigation",
            "Hill up soil to protect tubers"
        ]
    },
    "Potato___healthy": {
        "name": "Healthy Potato Plant",
        "description": "Your potato plant is healthy with no disease symptoms. Maintain current growing practices.",
        "symptoms": [
            "Green, vigorous foliage",
            "No leaf spots or blight",
            "Healthy plant growth"
        ],
        "cure": [
            "Maintain consistent soil moisture",
            "Apply balanced fertilizer",
            "Hill soil around plants",
            "Monitor for Colorado potato beetles",
            "Harvest at proper maturity"
        ]
    },
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "description": "Bacterial spot is caused by Xanthomonas species. It affects leaves, stems, and fruit of tomato plants.",
        "symptoms": [
            "Small, dark, greasy-looking spots on leaves",
            "Raised spots on fruit",
            "Yellow halos around leaf spots",
            "Premature leaf drop"
        ],
        "cure": [
            "Use disease-free seeds and transplants",
            "Apply copper-based bactericides",
            "Remove infected plant material",
            "Avoid overhead watering",
            "Rotate crops for 2-3 years"
        ]
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "description": "Early blight is caused by Alternaria solani. It's common in warm, humid conditions.",
        "symptoms": [
            "Dark spots with concentric rings on lower leaves",
            "Yellowing around spots",
            "Defoliation from bottom up",
            "Lesions on fruit (usually at stem end)"
        ],
        "cure": [
            "Remove infected lower leaves",
            "Apply fungicides containing chlorothalonil",
            "Mulch around plants",
            "Stake and prune for air circulation",
            "Water at base of plants"
        ]
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "description": "Late blight is caused by Phytophthora infestans. It can destroy entire tomato crops rapidly.",
        "symptoms": [
            "Large, irregular brown lesions on leaves",
            "White fungal growth on leaf undersides",
            "Brown, greasy-looking spots on fruit",
            "Rapid plant collapse"
        ],
        "cure": [
            "Apply fungicides immediately (copper, chlorothalonil)",
            "Remove and destroy infected plants",
            "Improve air circulation",
            "Avoid wetting foliage",
            "Plant resistant varieties if available"
        ]
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "description": "Leaf mold is caused by Passalora fulva. It's most common in greenhouse and high tunnel production.",
        "symptoms": [
            "Pale green or yellow spots on upper leaf surfaces",
            "Olive-green to gray fuzzy growth on undersides",
            "Curling and browning of leaves",
            "Rarely affects fruit"
        ],
        "cure": [
            "Reduce humidity through ventilation",
            "Space plants for air circulation",
            "Apply fungicides containing chlorothalonil",
            "Remove infected leaves",
            "Plant resistant varieties"
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "description": "Septoria leaf spot is caused by Septoria lycopersici. It's one of the most destructive tomato diseases.",
        "symptoms": [
            "Small, circular spots with gray centers",
            "Dark borders around spots",
            "Tiny black specks in center of spots",
            "Defoliation starting from bottom"
        ],
        "cure": [
            "Remove infected leaves immediately",
            "Apply fungicides containing chlorothalonil or copper",
            "Mulch to prevent soil splash",
            "Rotate crops",
            "Avoid overhead irrigation"
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Two-Spotted Spider Mites",
        "description": "Spider mites are tiny arachnids that feed on plant sap. They thrive in hot, dry conditions.",
        "symptoms": [
            "Stippling or tiny yellow spots on leaves",
            "Fine webbing on plants",
            "Bronzed or silvery leaves",
            "Leaf drop in severe cases"
        ],
        "cure": [
            "Spray with strong water jet to dislodge mites",
            "Apply insecticidal soap or neem oil",
            "Release predatory mites",
            "Maintain adequate soil moisture",
            "Remove heavily infested leaves"
        ]
    },
    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "description": "Target spot is caused by Corynespora cassiicola. It affects tomato leaves, stems, and fruit.",
        "symptoms": [
            "Brown spots with concentric rings (target pattern)",
            "Lesions larger than early blight",
            "Defoliation",
            "Fruit lesions can occur"
        ],
        "cure": [
            "Apply fungicides containing chlorothalonil",
            "Remove infected plant debris",
            "Improve air circulation",
            "Avoid overhead watering",
            "Rotate with non-host crops"
        ]
    },
    "Tomato___Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": "TYLCV is transmitted by whiteflies. It's a serious viral disease in warm climates.",
        "symptoms": [
            "Upward curling of leaves",
            "Yellowing of leaf margins",
            "Stunted plant growth",
            "Reduced fruit production"
        ],
        "cure": [
            "Control whiteflies with insecticides or yellow sticky traps",
            "Remove infected plants immediately",
            "Use reflective mulches",
            "Plant virus-resistant varieties",
            "Use insect-proof netting in greenhouses"
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "Tomato mosaic virus (ToMV) causes mottled foliage and reduced yield. It spreads through contact and tools.",
        "symptoms": [
            "Mottled light and dark green pattern on leaves",
            "Distorted leaves",
            "Stunted growth",
            "Reduced fruit set and quality"
        ],
        "cure": [
            "Remove and destroy infected plants",
            "Sanitize tools and hands",
            "Plant resistant varieties",
            "Control weeds that harbor virus",
            "Avoid handling plants when wet"
        ]
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato Plant",
        "description": "Your tomato plant is healthy with no disease or pest issues. Continue your current care routine.",
        "symptoms": [
            "Dark green, lush foliage",
            "No spots or discoloration",
            "Good fruit set and development"
        ],
        "cure": [
            "Water consistently at base of plant",
            "Apply balanced fertilizer regularly",
            "Stake or cage plants for support",
            "Prune suckers for better air flow",
            "Monitor regularly for early issues"
        ]
    }
}

# Translation content
TRANSLATIONS = {
    "English": {
        "title": "🌿 Plant Disease Classifier",
        "subtitle": "Upload a plant leaf image for AI-powered disease detection and treatment recommendations",
        "language_label": "Select Language / भाषा चुनें",
        "about_header": "About",
        "about_text": """
        This application uses deep learning to identify plant diseases from leaf images.
        
        **Supported Plants:**
        - Apple
        - Corn (Maize)
        - Grape
        - Potato
        - Tomato
        
        **Features:**
        - Disease identification
        - Treatment recommendations
        - Hindi translations
        """,
        "instructions_header": "Instructions",
        "instructions_text": """
        1. Upload a clear image of the plant leaf
        2. Click 'Analyze Disease'
        3. View results and recommendations
        """,
        "upload_header": "📤 Upload Plant Leaf Image",
        "upload_prompt": "Choose an image...",
        "upload_info": "👆 Please upload an image to get started",
        "analyze_button": "🔬 Analyze Disease",
        "analyzing": "🔍 Analyzing image...",
        "analysis_complete": "✅ Analysis Complete!",
        "detected_condition": "Detected Condition",
        "confidence": "Confidence",
        "disease_info_header": "📋 Disease Information",
        "symptoms_header": "🔍 Symptoms",
        "treatment_header": "💊 Treatment & Management",
        "footer_text": "🌱 Plant Disease Classifier ",
        "disclaimer": "For educational purposes only. Consult agricultural experts for serious infestations."
    },
    "Hindi": {
        "title": "🌿 पौधे रोग पहचान प्रणाली",
        "subtitle": "एआई-संचालित रोग पहचान और उपचार सिफारिशों के लिए पौधे की पत्ती की छवि अपलोड करें",
        "language_label": "भाषा चुनें / Select Language",
        "about_header": "परिचय",
        "about_text": """
        यह एप्लिकेशन पत्तियों की छवियों से पौधों की बीमारियों की पहचान करने के लिए डीप लर्निंग का उपयोग करता है।
        
        **समर्थित पौधे:**
        - सेब
        - मक्का
        - अंगूर
        - आलू
        - टमाटर
        
        **विशेषताएं:**
        - रोग की पहचान
        - उपचार की सिफारिशें
        - हिंदी अनुवाद
        """,
        "instructions_header": "निर्देश",
        "instructions_text": """
        1. पौधे की पत्ती की स्पष्ट तस्वीर अपलोड करें
        2. 'रोग का विश्लेषण करें' पर क्लिक करें
        3. परिणाम और सिफारिशें देखें
        """,
        "upload_header": "📤 पौधे की पत्ती की छवि अपलोड करें",
        "upload_prompt": "एक छवि चुनें...",
        "upload_info": "👆 कृपया शुरू करने के लिए एक छवि अपलोड करें",
        "analyze_button": "🔬 रोग का विश्लेषण करें",
        "analyzing": "🔍 छवि का विश्लेषण किया जा रहा है...",
        "analysis_complete": "✅ विश्लेषण पूर्ण!",
        "detected_condition": "पहचानी गई स्थिति",
        "confidence": "विश्वास स्तर",
        "disease_info_header": "📋 रोग की जानकारी",
        "symptoms_header": "🔍 लक्षण",
        "treatment_header": "💊 उपचार और प्रबंधन",
        "footer_text": "🌱 पौधे रोग पहचान प्रणाली ",
        "disclaimer": "केवल शैक्षिक उद्देश्यों के लिए। गंभीर संक्रमण के लिए कृषि विशेषज्ञों से परामर्श करें।"
    }
}

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name, confidence

def translate_to_hindi(text):
    try:
        translation = translator.translate(text, src='en', dest='hi')
        return translation.text
    except:
        return text

# Get current language
lang = st.session_state.language
t = TRANSLATIONS[lang]

# Language Toggle in Sidebar
with st.sidebar:
    st.markdown('<div class="language-toggle">', unsafe_allow_html=True)
    st.markdown(f"**{TRANSLATIONS['English']['language_label']}**")
    
    language_option = st.radio(
        "",
        options=["English", "Hindi / हिंदी"],
        index=0 if st.session_state.language == "English" else 1,
        label_visibility="collapsed"
    )
    
    if language_option == "Hindi / हिंदी":
        st.session_state.language = "Hindi"
    else:
        st.session_state.language = "English"
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Update language after selection
    lang = st.session_state.language
    t = TRANSLATIONS[lang]
    
    if os.path.exists(f"{working_dir}/logo.png"):
        st.image(f"{working_dir}/logo.png", width=100)
    
    st.header(t["about_header"])
    st.info(t["about_text"])
    
    st.header(t["instructions_header"])
    st.markdown(t["instructions_text"])

# Header
st.markdown(f'<p class="main-header">{t["title"]}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{t["subtitle"]}</p>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(t["upload_header"])
    uploaded_image = st.file_uploader(t["upload_prompt"], type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        analyze_button = st.button(t["analyze_button"], use_container_width=True)
    else:
        st.info(t["upload_info"])
        analyze_button = False

with col2:
    if uploaded_image is not None and analyze_button:
        with st.spinner(t["analyzing"]):
            # Predict
            prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
            
            st.success(t["analysis_complete"])
            st.metric(t["detected_condition"], prediction.replace('___', ' - ').replace('_', ' '))
            st.metric(t["confidence"], f"{confidence:.2f}%")
            
            # Get disease info
            disease_info = DISEASE_INFO.get(prediction, None)
            
            if disease_info:
                st.markdown("---")
                
                # Display in selected language
                if lang == "Hindi":
                    # Translate to Hindi
                    disease_name_hi = translate_to_hindi(disease_info['name'])
                    disease_desc_hi = translate_to_hindi(disease_info['description'])
                    
                    st.markdown(f"### {t['disease_info_header']}")
                    st.markdown(f'<div class="hindi-card">', unsafe_allow_html=True)
                    st.markdown(f"**{disease_name_hi}**")
                    st.write(disease_desc_hi)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Symptoms
                    with st.expander(t["symptoms_header"], expanded=True):
                        for symptom in disease_info['symptoms']:
                            symptom_hi = translate_to_hindi(symptom)
                            st.markdown(f"• {symptom_hi}")
                    
                    # Treatment
                    st.markdown(f"### {t['treatment_header']}")
                    st.markdown(f'<div class="hindi-card">', unsafe_allow_html=True)
                    for i, cure in enumerate(disease_info['cure'], 1):
                        cure_hi = translate_to_hindi(cure)
                        st.markdown(f"**{i}.** {cure_hi}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                else:
                    # Display in English
                    st.markdown(f"### {t['disease_info_header']}")
                    st.markdown(f'<div class="disease-card">', unsafe_allow_html=True)
                    st.markdown(f"**{disease_info['name']}**")
                    st.write(disease_info['description'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Symptoms
                    with st.expander(t["symptoms_header"], expanded=True):
                        for symptom in disease_info['symptoms']:
                            st.markdown(f"• {symptom}")
                    
                    # Treatment
                    st.markdown(f"### {t['treatment_header']}")
                    st.markdown(f'<div class="cure-card">', unsafe_allow_html=True)
                    for i, cure in enumerate(disease_info['cure'], 1):
                        st.markdown(f"**{i}.** {cure}")
                    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p>{t["footer_text"]}</p>
    <p style='font-size: 0.8rem;'>{t["disclaimer"]}</p>
</div>
""", unsafe_allow_html=True)