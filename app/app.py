import os
import sys

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)

sys.path.append(parent)

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from transformers import AutoImageProcessor, pipeline
from model import Classifier

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="chriamue/bird-species-classifier")
pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")


processor = AutoImageProcessor.from_pretrained("chriamue/bird-species-classifier")
model = AutoImageProcessor.from_pretrained("chriamue/bird-species-classifier")



# Define labels
labels = [
    "dog"
]

# Preprocess function
def preprocess(image):
    image = np.array(image)
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()
    transform = A.Compose([resize, normalize, to_tensor])
    image = transform(image=image)["image"]
    return image


# Define the sample images
sample_images = {"dog": "./test_images/dog.jpeg",
}

# Define the function to make predictions on an image
def predict(image):
    try:
        image = preprocess(image).unsqueeze(0)

        # Prediction
        # Make a prediction on the image
        with torch.no_grad():
            output = model(image)
            # convert to probabilities
            probabilities = torch.nn.functional.softmax(output[0])

            topk_prob, topk_label = torch.topk(probabilities, 3)

            # convert the predictions to a list
            predictions = []
            for i in range(topk_prob.size(0)):
                prob = topk_prob[i].item()
                label = topk_label[i].item()
                predictions.append((prob, label))

            return predictions
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []


# Define the Streamlit app
def app():
    st.title("Animal-10 Image Classification")

    # Add a file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    # If an image is uploaded, make a prediction on it
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", width=300)
        predictions = predict(image)


    # Show the top 3 predictions with their probabilities
    if predictions:
        st.write("Top 3 predictions:")
        for i, (prob, label) in enumerate(predictions):
            st.write(f"{i+1}. {labels[label]} ({prob*100:.2f}%)")

            # Show progress bar with probabilities
            st.markdown(
                """
                <style>
                .stProgress .st-b8 {
                    background-color: orange;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.progress(prob)

    else:
        st.write("No predictions.")


# Run the app
if __name__ == "__main__":
    app()
