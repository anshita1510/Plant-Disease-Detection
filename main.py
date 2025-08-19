import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# Function to add background from local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1603470348871-5a0a88595e82?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8cGxhbnQlMjBpbiUyMGxpZ2h0JTIwY29sb3IlMjBmb3IlMjBiYWNrZ3JvdW5kJTIwcHVycG9zZXxlbnwwfHwwfHx8MA%3D%3D",{encoded});
            background-size: cover;
            background-attachment: fixed;
            
        }}
        h1, h2, h3, p, label {{
            color: Black !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function (replace with your background image file)
add_bg_from_local("Diseases.png")

# Prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["DISEASE RECOGNITION"])

# Main Banner Image
img = Image.open("Diseases.png")
st.image(img, use_column_width=True)


# Prediction Page
if app_mode == "DISEASE RECOGNITION":
    st.header("🌱 Plant Disease Detection System")
    test_image = st.file_uploader("📷 Choose an Image of a Plant Leaf")
    
    if test_image is not None and st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
    # Predict button
    if test_image is not None and st.button("Predict"):
        st.snow()
        st.write("🔎 Our Prediction")
        result_index = model_prediction(test_image)

        # Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        st.success(f"✅ Model Prediction: **{class_name[result_index]}**")
