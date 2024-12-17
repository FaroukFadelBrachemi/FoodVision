import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
from utils import load_and_prep, get_classes

# Place set_page_config first
st.set_page_config(page_title="Food Vision", page_icon="🍔")

# Cache the model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./models/EfficientNetB1.hdf5")

@st.cache_data
def get_class_names():
    return get_classes()

@st.cache_data
def predicting(image, model):
    # Preprocess the image
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    
    # Model prediction
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    
    # Get top 5 predictions
    top_5_indices = tf.argsort(preds[0], direction='DESCENDING')[:5]
    values = preds[0][top_5_indices] * 100
    labels = [class_names[i] for i in top_5_indices]
    
    # Create DataFrame
    df = pd.DataFrame({"Top 5 Predictions": labels, "F1 Scores": values.numpy()})
    return pred_class, pred_conf.numpy(), df

# Load model and class names
model = load_model()
class_names = get_class_names()

# Sidebar
st.sidebar.title("What's Food Vision?")
st.sidebar.write("""
FoodVision is a **CNN Image Classification Model** that classifies images into **101 food** categories using the **Food101 dataset**.

**Accuracy:** **`85%`**
**Model:** **`EfficientNetB1`**
**Dataset:** **`Food101`**
""")
st.sidebar.markdown("### Contact")
st.sidebar.markdown("""
<a href="https://kaggle.com/faroukfadelbrachemi" target="_blank"><img src="https://shorturl.at/LlbEa" height="40"></a>
<a href="https://linkedin.com/in/farouk-brachemi" target="_blank"><img src="https://bit.ly/3wCl82U" height="40"></a>
<a href="https://github.com/faroukbrachemi" target="_blank"><img src="https://shorturl.at/Nj7gk" height="40"></a>
""", unsafe_allow_html=True)

# Main
st.title("Food Vision 🍔📷")
st.header("Identify what's in your food photos!")
st.write("Check 👉 [**GitHub Repo**](https://github.com/faroukbrachemi/FoodVision)")

file = st.file_uploader("Upload an image of food.", type=["jpg", "jpeg", "png"])
if not file:
    st.warning("Please upload an image")
    st.stop()

# Display image
image = tf.io.decode_image(file.read(), channels=3)
st.image(image, use_column_width=True)

if st.button("Predict"):
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction: {pred_class}\nConfidence: {pred_conf:.2f}%')
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('F1 Scores', title="F1 Scores (%)"),
        y=alt.Y('Top 5 Predictions', sort=None),
        color=alt.value('#EC5953'),
        text=alt.Text('F1 Scores', format='.2f')
    ).properties(width=600, height=400)
    st.altair_chart(chart)
