import streamlit as st
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")


def load_image(img):
    img = Image.open(img)
    img = img.resize((256, 256))
    img = np.array(img)/255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


st.title('Neural Style Transfer')
st.write("Upload a content image and a style image to apply neural style transfer.")

content_file = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])

if content_file is not None and style_file is not None:
    content_img = load_image(content_file)
    style_img = load_image(style_file)


    stylized_image = model(tf.constant(content_img), tf.constant(style_img))[0]


    st.subheader("Content Image")
    st.image(np.squeeze(content_img), use_column_width=True)

    st.subheader("Style Image")
    st.image(np.squeeze(style_img), use_column_width=True)

    st.subheader("Stylized Image")
    st.image(np.squeeze(stylized_image), use_column_width=True)

    st.subheader("Download Stylized Image")
    buf = BytesIO()
    stylized_img_pil = Image.fromarray((np.squeeze(stylized_image) * 255).astype(np.uint8))
    stylized_img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download image", data=byte_im, file_name="stylized_image.png", mime="image/png")
