import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Digital Image Processing App - (for ace filtering, or pseudocoloring, or unsharp masking)", layout="wide")


if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None


def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

def ace_filter_channel(channel):
    return cv2.equalizeHist(channel)

def ace_filter_rgb(img):
    b, g, r = cv2.split(img)
    return cv2.merge([ace_filter_channel(b), ace_filter_channel(g), ace_filter_channel(r)])

def ace_filter_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, ace_filter_channel(gray)

def apply_ace_filtering():
    original = st.session_state.original_image
    if original is None:
        st.error("Please upload an image first!")
        return

    gray, gray_eq = ace_filter_gray(original)
    ace_rgb_combined = ace_filter_rgb(original)
    b, g, r = cv2.split(original)
    b_eq, g_eq, r_eq = cv2.split(ace_rgb_combined)

    images_to_show = [
        ("Original Grayscale", gray),
        ("Original R Channel", r),
        ("Original G Channel", g),
        ("Original B Channel", b),
        ("ACE on Grayscale", gray_eq),
        ("ACE on R Channel", r_eq),
        ("ACE on G Channel", g_eq),
        ("ACE on B Channel", b_eq),
        ("Final ACE RGB Image", ace_rgb_combined),
    ]

    show_images(images_to_show)
    st.session_state.processed_image = ace_rgb_combined

def apply_pseudocoloring():
    original = st.session_state.original_image
    if original is None:
        st.error("Please upload an image first!")
        return

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    pseudocolored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    images_to_show = [
        ("Grayscale Image", gray),
        ("Pseudocolored Image", pseudocolored),
    ]

    show_images(images_to_show)
    st.session_state.processed_image = pseudocolored

def apply_unsharp_masking():
    original = st.session_state.original_image
    if original is None:
        st.error("Please upload an image first!")
        return

    def unsharp_mask(channel):
        blur = cv2.GaussianBlur(channel, (9, 9), 10.0)
        return cv2.addWeighted(channel, 1.5, blur, -0.5, 0)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_usm = unsharp_mask(gray)
    b, g, r = cv2.split(original)
    b_usm = unsharp_mask(b)
    g_usm = unsharp_mask(g)
    r_usm = unsharp_mask(r)
    final_usm_rgb = cv2.merge([b_usm, g_usm, r_usm])

    images_to_show = [
        ("Original Grayscale", gray),
        ("Unsharp Mask on Grayscale", gray_usm),
        ("Original R Channel", r),
        ("Original G Channel", g),
        ("Original B Channel", b),
        ("Unsharp R Channel", r_usm),
        ("Unsharp G Channel", g_usm),
        ("Unsharp B Channel", b_usm),
        ("Final Unsharp RGB", final_usm_rgb),
    ]

    show_images(images_to_show)
    st.session_state.processed_image = final_usm_rgb

def show_images(images_list):
    cols = st.columns(3)
    for i, (title, img) in enumerate(images_list):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb).resize((200, 200))
        cols[i % 3].image(pil_img, caption=title)

def clear_all():
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.rerun()

# Layout
st.title("🖼️ Image Processing App")

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "bmp"])
if uploaded_file is not None:
    st.session_state.original_image = load_image(uploaded_file)
    st.success("Image uploaded successfully!")

if st.session_state.original_image is not None:
    st.subheader("Original Image")
    st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), width=400)

st.divider()

col3, col4, col5 = st.columns(3)
with col3:
    if st.button("Apply ACE Filtering"):
        apply_ace_filtering()
with col4:
    if st.button("Apply Pseudocoloring"):
        apply_pseudocoloring()
with col5:
    if st.button("Apply Unsharp Masking"):
        apply_unsharp_masking()

st.divider()

# Show final processed image after processing
if st.session_state.processed_image is not None:
    st.subheader("Processed Image")
    st.image(
        cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB),
        width=600  
    )
    # Download button
    st.download_button(
        label="Download Processed Image",
        data=cv2.imencode('.png', st.session_state.processed_image)[1].tobytes(),
        file_name="processed_image.png",
        mime="image/png"
    )

# Clear button
if st.button("Clear All"):
    clear_all()
