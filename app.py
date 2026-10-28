# app.py
# Skincare Copilot - Streamlit web app (entry point)
# Run with: streamlit run app.py

import hashlib
import cv2
import numpy as np
import streamlit as st

from face_detector import resize_image, detect_face, get_regions, draw_regions
from skin_analyzer import analyze_skin
from tips import get_tips, skin_type_summary, DISCLAIMER
from chatbot import get_reply

WELCOME_MESSAGE = ("Hi! I am your skincare assistant. Analyze a photo above, then ask me "
                   "\"what about my skin?\", or ask about acne, sunscreen, routines and more.")

NO_FACE_MESSAGE = "No face found. Please use a clear, front-facing photo in good light."

# Must be the first Streamlit command
st.set_page_config(page_title="Skincare Copilot", layout="wide")


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def init_session():
    """Create the values we keep between reruns (only on the first run)."""
    if "analysis" not in st.session_state:
        st.session_state.analysis = None      # latest analysis (dictionary) or None
    if "image_id" not in st.session_state:
        st.session_state.image_id = None      # fingerprint of the last analyzed image
    if "no_face" not in st.session_state:
        st.session_state.no_face = False      # True if the last image had no face
    if "messages" not in st.session_state:
        # Chat history: a list of {"role": ..., "text": ...}
        st.session_state.messages = [{"role": "assistant", "text": WELCOME_MESSAGE}]


def bytes_to_image(data):
    """Turn uploaded/captured image bytes into an OpenCV BGR image (or None)."""
    # The bytes become a 1D numpy array, which cv2.imdecode turns into an image
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    # Same max-width resize as load_image, so thresholds behave the same
    return resize_image(image)


def to_rgb(image):
    """OpenCV uses BGR, Streamlit needs RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def run_analysis(image):
    """Run the full pipeline on one image. Returns a dictionary, or None if no face."""
    face = detect_face(image)
    if face is None:
        return None

    regions = get_regions(image, face)
    result, spot_image = analyze_skin(regions)

    return {
        "result": result,
        "summary": skin_type_summary(result),
        "tips": get_tips(result),
        "boxes_image": draw_regions(image, face, regions),
        "regions": regions,
        "spot_image": spot_image,
    }


def handle_new_image(data):
    """Analyze the image only if it is new (not on every rerun or chat message)."""
    # A fingerprint (hash) of the bytes tells us if this is the same image as before
    image_id = hashlib.md5(data).hexdigest()
    if image_id == st.session_state.image_id:
        return
    st.session_state.image_id = image_id

    image = bytes_to_image(data)
    if image is None:
        st.session_state.analysis = None
        st.session_state.no_face = False
        st.error("Could not read this image. Please try a JPG or PNG file.")
        return

    with st.spinner("Analyzing skin..."):
        analysis = run_analysis(image)

    # Save for later reruns (the chatbot uses this too)
    st.session_state.analysis = analysis
    st.session_state.no_face = analysis is None


# ------------------------------------------------------------
# Page sections
# ------------------------------------------------------------


def show_sidebar():
    """Short explanation of how the app works."""
    st.sidebar.header("How it works")
    st.sidebar.markdown(
        "1. **Skin analysis** - OpenCV finds your face, crops the forehead and cheeks, "
        "and measures tone, oiliness, redness, dark spots and texture.\n"
        "2. **Tips engine** - simple if/else rules turn the measurements into tips.\n"
        "3. **Chatbot** - keyword matching answers skincare questions using your result."
    )
    st.sidebar.header("CV techniques used")
    st.sidebar.markdown(
        "- Haar cascade face detection\n"
        "- ROI cropping (face proportions)\n"
        "- Colour spaces: LAB, HSV, YCrCb\n"
        "- Thresholding\n"
        "- Gaussian blur + image subtraction\n"
        "- Morphological opening\n"
        "- Contours with area filtering\n"
        "- Laplacian variance (texture)"
    )


def show_input():
    """Let the user upload a photo or take a camera snapshot. Returns image bytes or None."""
    mode = st.radio("Choose input", ["Upload Photo", "Live Analyse"], horizontal=True)

    if mode == "Upload Photo":
        file = st.file_uploader("Upload a front-facing photo", type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input("Take a photo (face the camera in good light)")

    if file is None:
        return None
    return file.getvalue()


def show_report(analysis):
    """Show the image with boxes, measurements, tips and disclaimer."""
    result = analysis["result"]
    st.subheader("Your Skin Report")

    left, right = st.columns([1, 1])
    with left:
        st.image(to_rgb(analysis["boxes_image"]), caption="Detected face and skin regions",
                 width="stretch")
    with right:
        st.markdown("**" + analysis["summary"] + "**")

        # Five measurements side by side
        cols = st.columns(5)
        cols[0].metric("Tone", result["tone"])
        cols[1].metric("Oiliness", result["oiliness"])
        cols[2].metric("Redness", result["redness"])
        cols[3].metric("Dark spots", result["spots"])
        cols[4].metric("Texture", result["texture"])

        st.markdown("**Tips for you**")
        st.markdown("\n".join("- " + tip for tip in analysis["tips"]))

    st.caption(DISCLAIMER)


def show_cv_steps(analysis):
    """Show each computer vision step for the demo."""
    with st.expander("Show CV Steps"):
        regions = analysis["regions"]

        # Row 1: detection and the three cropped regions
        cols = st.columns(4)
        cols[0].image(to_rgb(analysis["boxes_image"]), width="stretch",
                      caption="1. Haar cascade face detection + region boxes")
        cols[1].image(to_rgb(regions["forehead"]["roi"]), width="stretch",
                      caption="2. ROI cropping: forehead (LAB, HSV, YCrCb)")
        cols[2].image(to_rgb(regions["left_cheek"]["roi"]), width="stretch",
                      caption="2. ROI cropping: left cheek")
        cols[3].image(to_rgb(regions["right_cheek"]["roi"]), width="stretch",
                      caption="2. ROI cropping: right cheek")

        # Row 2: spot detection and the raw numbers
        cols = st.columns(2)
        if analysis["spot_image"] is not None:
            cols[0].image(to_rgb(analysis["spot_image"]), width="stretch",
                          caption="3. Spot detection: blur subtraction + threshold + "
                                  "morphology + contours (left: circled, right: mask)")
        with cols[1]:
            st.markdown("**Raw measurements**")
            st.json(analysis["result"]["raw"])
            st.caption("lightness_L: LAB L | oily_percent: HSV bright + low saturation | "
                       "red_percent / mean_a: LAB a channel | "
                       "texture_variance: Laplacian variance")


def show_chat():
    """Chatbot with history kept in session_state."""
    st.subheader("Ask the Skincare Chatbot")

    # Latest analysis result for the chatbot (None if nothing analyzed yet)
    last_result = None
    if st.session_state.analysis is not None:
        last_result = st.session_state.analysis["result"]

    # Show the conversation so far
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Two spaces before \n make a line break in Markdown
            st.markdown(message["text"].replace("\n", "  \n"))

    # New question from the user
    question = st.chat_input("Ask about your skin, acne, sunscreen, routines...")
    if question:
        reply = get_reply(question, last_result)

        # Save both messages so they stay after the next rerun
        st.session_state.messages.append({"role": "user", "text": question})
        st.session_state.messages.append({"role": "assistant", "text": reply})

        # Show them now (the loop above only showed older messages)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(reply.replace("\n", "  \n"))


# ------------------------------------------------------------
# Main page
# ------------------------------------------------------------

init_session()
show_sidebar()

# Header
st.title("Skincare Copilot")
st.write("Analyze your skin from a photo with computer vision, get simple tips, "
         "and ask the skincare chatbot.")
st.info(DISCLAIMER)

# Input: analyze only when a new image arrives
data = show_input()
if data is not None:
    handle_new_image(data)

# Report (from session_state, so it stays while chatting)
if st.session_state.no_face:
    st.warning(NO_FACE_MESSAGE)
elif st.session_state.analysis is not None:
    show_report(st.session_state.analysis)
    show_cv_steps(st.session_state.analysis)
else:
    st.write("Upload a photo or take a snapshot to see your skin report.")

st.divider()
show_chat()
