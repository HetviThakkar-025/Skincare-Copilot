# app.py
# Skincare Copilot - Streamlit web app (entry point)
# Run with: streamlit run app.py
# Pages: Home, Analyze, Chatbot, CV Steps (chosen in the sidebar)

import hashlib
import cv2
import numpy as np
import streamlit as st

import skin_analyzer as sa
from face_detector import resize_image, detect_face, get_regions, draw_regions, MAX_IMAGE_WIDTH
from skin_analyzer import analyze_skin, get_skin_mask
from tips import get_tips, skin_type_summary, DISCLAIMER, MANY_SPOTS_COUNT
from chatbot import get_reply

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Page names (the emoji is part of the label shown in the sidebar)
HOME = "🏠 Home"
ANALYZE = "🔬 Analyze"
CHATBOT = "💬 Chatbot"
CV_STEPS = "🧪 CV Steps"
PAGES = [HOME, ANALYZE, CHATBOT, CV_STEPS]

WELCOME_MESSAGE = ("Hi! I am your skincare assistant. Ask me \"what about my skin?\" "
                   "after an analysis, or ask about acne, sunscreen, routines and more.")
NO_FACE_MESSAGE = "No face found. Please use a clear, front-facing photo in good light."
MY_SKIN_QUESTION = "what about my skin?"

# Suggested questions shown as buttons on the Chatbot page
SUGGESTED_QUESTIONS = ["What about my skin?", "How to remove pimples?",
                       "Which sunscreen?", "Morning routine"]

# Badge colour level for each label: good = green, moderate = orange, high = red
LEVELS = {
    "oiliness": {"Normal": "good", "Dry": "moderate", "Oily": "high"},
    "redness": {"Low": "good", "Moderate": "moderate", "High": "high"},
    "texture": {"Smooth": "good", "Normal": "good", "Rough": "high"},
}
BADGE_TEXT = {"good": "Good", "moderate": "Moderate", "high": "Needs care", "info": "Info"}

# Spot count boundaries for the badge (the "many" boundary comes from tips.py)
FEW_SPOTS_COUNT = 4   # fewer than 4 spots = good

# Value that fills the bar to 100% for each measurement
OILY_BAR_MAX = 2 * sa.OILY_NORMAL_MAX_PERCENT    # 10% shiny pixels = full bar
RED_BAR_MAX = 100.0                              # percentage, so 100% = full bar
SPOTS_BAR_MAX = 2 * MANY_SPOTS_COUNT             # 16 spots = full bar
TEXTURE_BAR_MAX = 2 * sa.TEXTURE_NORMAL_MAX      # variance 300 = full bar

# Tip groups: (icon + title, keywords). A tip goes into the first group whose
# keyword appears in it; tips that match nothing go into "Daily habits".
TIP_GROUPS = [
    ("🔍 Dark spots", ["spot", "marks"]),
    ("☀️ Sun protection", ["sun", "spf"]),
    ("🌡️ Redness", ["redness", "fragrance", "scrub"]),
    ("💧 Oil balance", ["oil", "shine", "comedogenic", "cleanser", "moistur", "dry",
                       "lukewarm", "exfoliat", "rough"]),
]
DEFAULT_TIP_GROUP = "🌿 Daily habits"

# CV Steps page: every image is fitted into a tile of this size (4:3), so all line up
TILE_WIDTH = 480
TILE_HEIGHT = 360
TILE_PAD_COLOR = (31, 33, 22)   # BGR of the dark card colour, fills the empty space

# Small CSS block for the card look (dark theme: dark cards, rounded corners, shadow)
CSS = """
<style>
.card {background: #16211f; color: #e6efed; border: 1px solid #24423d; border-radius: 14px;
       padding: 18px; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
       margin-bottom: 14px; overflow-wrap: anywhere;}
.card h3, .card h4 {color: #e6efed;}
.feature-card, .step-card {min-height: 170px;}
.result-card {min-height: 250px;}
.tip-card {min-height: 150px;}
.hero {background: linear-gradient(135deg, #0f3d38, #3a1d2e); border-radius: 18px;
       padding: 36px 32px; margin-bottom: 20px; color: #e6efed;}
.hero h1 {margin: 0 0 8px 0; color: #5eead4;}
.headline {background: #12302c; border-left: 6px solid #2dd4bf; font-size: 1.15rem;
           font-weight: 600;}
.card-title {font-weight: 600; color: #a7bfbb; margin-bottom: 6px;}
.card-value {font-size: 1.6rem; font-weight: 700; margin-bottom: 6px;}
.card-raw {font-size: 0.85rem; color: #a7bfbb; margin-top: 4px;}
.card-note {font-size: 0.8rem; color: #7f9894; margin-top: 6px;}
.badge {display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;}
.badge.good {background: rgba(52, 168, 83, 0.18); color: #6fdc8c;}
.badge.moderate {background: rgba(251, 140, 0, 0.18); color: #ffb74d;}
.badge.high {background: rgba(229, 57, 53, 0.2); color: #ff8a80;}
.badge.info {background: rgba(45, 212, 191, 0.15); color: #5eead4;}
.bar {background: #24332f; border-radius: 999px; height: 8px; margin-top: 10px;}
.bar-fill {height: 8px; border-radius: 999px;}
.bar-fill.good {background: #34a853;}
.bar-fill.moderate {background: #fb8c00;}
.bar-fill.high {background: #e53935;}
.bar-fill.info {background: #2dd4bf;}
.step-number {display: inline-block; width: 30px; height: 30px; line-height: 30px;
              text-align: center; border-radius: 50%; background: #2dd4bf; color: #0e1514;
              font-weight: 700; margin-bottom: 8px;}
.chip {display: inline-block; padding: 4px 12px; margin: 4px; border-radius: 999px;
       background: rgba(45, 212, 191, 0.12); color: #5eead4;
       border: 1px solid rgba(45, 212, 191, 0.3); font-size: 0.85rem;}
.tip-card ul {padding-left: 18px; margin: 6px 0 0 0;}
.tile-title {font-weight: 600; color: #e6efed; margin-bottom: 8px; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis;}
.tile-caption {font-size: 0.8rem; color: #a7bfbb; margin-top: 6px; min-height: 2.6em;}
</style>
"""

# Must be the first Streamlit command
st.set_page_config(page_title="Skincare Copilot", page_icon="🧴", layout="wide")


# ------------------------------------------------------------
# Session state and analysis
# ------------------------------------------------------------


def init_session():
    """Create the values we keep between reruns and page switches (first run only)."""
    defaults = {
        "page": HOME,              # current page (also the sidebar radio's key)
        "analysis": None,          # latest analysis dictionary, or None
        "no_face": False,          # True if the last image had no face
        "upload_id": None,         # fingerprint of the last analyzed upload
        "camera_id": None,         # fingerprint of the last analyzed camera photo
        "pending_question": None,  # question sent from the Analyze page to the chatbot
        "messages": [{"role": "assistant", "text": WELCOME_MESSAGE}],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(page):
    """Button callback: switch page (runs before the next rerun, so it is allowed)."""
    st.session_state.page = page


def ask_about_my_skin():
    """Button callback: open the chatbot with the 'my skin' question ready."""
    st.session_state.page = CHATBOT
    st.session_state.pending_question = MY_SKIN_QUESTION


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

    # Everything the pages need is kept together in one dictionary
    return {
        "image": image,
        "face": face,
        "regions": regions,
        "result": result,
        "spot_image": spot_image,
        "summary": skin_type_summary(result),
        "tips": get_tips(result),
        "boxes_image": draw_regions(image, face, regions),
    }


def handle_new_image(data, source):
    """Analyze the image only if it is new for this source ("upload" or "camera")."""
    # A fingerprint (hash) of the bytes tells us if we already analyzed this image.
    # Each source has its own fingerprint, so the two tabs do not undo each other.
    image_id = hashlib.md5(data).hexdigest()
    id_key = source + "_id"
    if image_id == st.session_state[id_key]:
        return
    st.session_state[id_key] = image_id

    image = bytes_to_image(data)
    if image is None:
        st.error("Could not read this image. Please try a JPG or PNG file.")
        return

    with st.spinner("Analyzing skin..."):
        analysis = run_analysis(image)

    # Save in session_state so it survives reruns and page switches
    st.session_state.analysis = analysis
    st.session_state.no_face = analysis is None


# ------------------------------------------------------------
# Small HTML helpers (cards, chips)
# ------------------------------------------------------------


def html(text):
    """Show a piece of HTML (used for the styled cards)."""
    st.markdown(text, unsafe_allow_html=True)


def card(content, extra_class=""):
    """Wrap content in a styled card."""
    html('<div class="card ' + extra_class + '">' + content + '</div>')


def result_card(emoji, title, label, level, bar_percent, raw_text, note):
    """One measurement card: title, big label, coloured badge, bar and explanation."""
    # Keep the bar between 0% and 100%
    bar_percent = max(0, min(100, bar_percent))
    card(
        '<div class="card-title">' + emoji + " " + title + '</div>'
        '<div class="card-value">' + label + '</div>'
        '<span class="badge ' + level + '">' + BADGE_TEXT[level] + '</span>'
        '<div class="bar"><div class="bar-fill ' + level + '" '
        'style="width:' + str(round(bar_percent)) + '%"></div></div>'
        '<div class="card-raw">' + raw_text + '</div>'
        '<div class="card-note">' + note + '</div>',
        "result-card",
    )


def spots_level(count):
    """Badge level for the number of dark spots."""
    if count < FEW_SPOTS_COUNT:
        return "good"
    if count < MANY_SPOTS_COUNT:
        return "moderate"
    return "high"


def group_tips(tips):
    """Sort tips into groups by simple keyword matching. Returns {group title: [tips]}."""
    groups = {}
    for tip in tips:
        group = DEFAULT_TIP_GROUP
        lower_tip = tip.lower()
        for title, keywords in TIP_GROUPS:
            if any(keyword in lower_tip for keyword in keywords):
                group = title
                break
        groups.setdefault(group, []).append(tip)
    return groups


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------


def show_sidebar_top():
    """App name, tagline and page navigation."""
    st.sidebar.markdown("## 🧴 Skincare Copilot")
    st.sidebar.caption("Computer vision skin check, tips and a skincare chatbot.")
    # The radio's key is "page", so st.session_state.page always holds the current page
    st.sidebar.radio("Navigate", PAGES, key="page")


def show_sidebar_bottom():
    """Analysis status and the disclaimer (drawn after the page, so it is up to date)."""
    st.sidebar.divider()
    if st.session_state.analysis is not None:
        st.sidebar.success("Analysis ready")
    else:
        st.sidebar.info("No analysis yet")
    st.sidebar.caption(DISCLAIMER)


# ------------------------------------------------------------
# Page 1: Home
# ------------------------------------------------------------


def home_page():
    # Hero section
    html('<div class="hero"><h1>🧴 Skincare Copilot</h1>'
         '<p>Check your skin from a photo with computer vision, get simple tips, '
         'and ask the skincare chatbot.</p>'
         '<p><b>Start analysis:</b> open the Analyze page and upload a front-facing photo.</p>'
         '</div>')
    st.button("Start analysis →", type="primary", on_click=go_to, args=(ANALYZE,))

    # Three feature cards
    st.subheader("What it does")
    features = [
        ("🔬", "Skin Analysis", "OpenCV finds your face and measures tone, oiliness, "
                                "redness, dark spots and texture."),
        ("💡", "Personalised Tips", "Simple rules turn your results into practical "
                                    "skincare tips."),
        ("💬", "Skincare Chatbot", "Ask questions about acne, sunscreen or routines, "
                                   "with answers based on your result."),
    ]
    for column, (emoji, title, text) in zip(st.columns(3), features):
        with column:
            card('<h3>' + emoji + " " + title + '</h3><p>' + text + '</p>', "feature-card")

    # How it works: three numbered steps
    st.subheader("How it works")
    steps = [
        ("Upload or capture", "Upload a clear, front-facing photo or take a snapshot "
                              "with your camera."),
        ("OpenCV analyses", "The face is detected, forehead and cheeks are cropped, and "
                            "each region is measured."),
        ("Get tips and ask", "Read your skin report and tips, then ask the chatbot "
                             "anything."),
    ]
    for number, (column, (title, text)) in enumerate(zip(st.columns(3), steps), start=1):
        with column:
            card('<div class="step-number">' + str(number) + '</div>'
                 '<h4>' + title + '</h4><p>' + text + '</p>', "step-card")

    # CV techniques as small chips
    st.subheader("Computer vision techniques used")
    techniques = ["Haar cascade", "ROI cropping", "LAB colour space", "HSV colour space",
                  "YCrCb skin mask", "Thresholding", "Morphology", "Contours",
                  "Laplacian variance"]
    html("".join('<span class="chip">' + t + '</span>' for t in techniques))


# ------------------------------------------------------------
# Page 2: Analyze
# ------------------------------------------------------------


def show_result_cards(result):
    """Five result cards in a 3 + 2 grid."""
    raw = result["raw"]
    cards = [
        ("🎨", "Skin Tone", result["tone"], "info",
         100 * raw["lightness_L"] / 255,
         "Lightness L = " + str(raw["lightness_L"]) + " / 255",
         "Average lightness of the cheeks (LAB colour space)"),
        ("💧", "Oiliness", result["oiliness"],
         LEVELS["oiliness"].get(result["oiliness"], "info"),
         100 * raw["oily_percent"] / OILY_BAR_MAX,
         str(raw["oily_percent"]) + "% shiny pixels",
         "Share of shiny pixels on forehead and cheeks (HSV)"),
        ("🌡️", "Redness", result["redness"],
         LEVELS["redness"].get(result["redness"], "info"),
         100 * raw["red_percent"] / RED_BAR_MAX,
         str(raw["red_percent"]) + "% red pixels",
         "Share of red cheek pixels (LAB a channel)"),
        ("🔍", "Dark Spots", str(result["spots"]) + " found",
         spots_level(result["spots"]),
         100 * result["spots"] / SPOTS_BAR_MAX,
         str(result["spots"]) + " spots in 3 regions",
         "Small dark patches found with threshold + contours"),
        ("✨", "Texture", result["texture"],
         LEVELS["texture"].get(result["texture"], "info"),
         100 * raw["texture_variance"] / TEXTURE_BAR_MAX,
         "Laplacian variance = " + str(raw["texture_variance"]),
         "Amount of fine detail; rough can mean dryness"),
    ]

    # Row 1: three cards. Row 2: two cards (same width, third column left empty)
    for row in [cards[0:3], cards[3:5]]:
        for column, card_data in zip(st.columns(3), row):
            with column:
                result_card(*card_data)


def show_tips(tips):
    """Tips grouped under small headings, as cards (three per row)."""
    st.subheader("Your tips")
    groups = list(group_tips(tips).items())
    for start in range(0, len(groups), 3):
        for column, (title, group_tips_list) in zip(st.columns(3), groups[start:start + 3]):
            with column:
                items = "".join("<li>" + tip + "</li>" for tip in group_tips_list)
                card('<div class="card-title">' + title + '</div><ul>' + items + '</ul>',
                     "tip-card")


def show_report(analysis):
    """Image on the left, skin report on the right, then tips and disclaimer."""
    left, right = st.columns([2, 3])
    with left:
        st.image(to_rgb(analysis["boxes_image"]), width="stretch",
                 caption="Detected face and skin regions")
    with right:
        st.subheader("Your skin report")
        card(analysis["summary"], "headline")
        show_result_cards(analysis["result"])

    show_tips(analysis["tips"])
    st.info(DISCLAIMER)
    st.button("💬 Ask the chatbot about my skin", type="primary", on_click=ask_about_my_skin)


def analyze_page():
    st.title("🔬 Analyze your skin")

    # Two input tabs (the first tab is shown by default)
    upload_tab, camera_tab = st.tabs(["📁 Upload Photo", "📷 Live Analyse"])
    with upload_tab:
        uploaded = st.file_uploader("Upload a clear, front-facing photo",
                                    type=["jpg", "jpeg", "png"])
    with camera_tab:
        captured = st.camera_input("Face the camera in good light and take a photo")

    # Analyze only when a new image arrives
    if uploaded is not None:
        handle_new_image(uploaded.getvalue(), "upload")
    if captured is not None:
        handle_new_image(captured.getvalue(), "camera")

    # Show the latest result (kept in session_state)
    if st.session_state.no_face:
        st.warning(NO_FACE_MESSAGE)
    elif st.session_state.analysis is not None:
        show_report(st.session_state.analysis)
    else:
        st.write("Upload a photo or take a snapshot to see your skin report.")


# ------------------------------------------------------------
# Page 3: Chatbot
# ------------------------------------------------------------


def ask(question):
    """Send a question to the chatbot and save both messages in the history."""
    last_result = None
    if st.session_state.analysis is not None:
        last_result = st.session_state.analysis["result"]

    reply = get_reply(question, last_result)
    st.session_state.messages.append({"role": "user", "text": question})
    st.session_state.messages.append({"role": "assistant", "text": reply})


def clear_chat():
    """Button callback: start the conversation again."""
    st.session_state.messages = [{"role": "assistant", "text": WELCOME_MESSAGE}]


def chatbot_page():
    st.title("💬 Skincare Chatbot")
    if st.session_state.analysis is not None:
        st.caption("✅ Using your latest analysis")
    else:
        st.caption("No analysis yet, upload a photo on the Analyze page for personal answers")

    # A question sent from the Analyze page ("Ask the chatbot about my skin")
    if st.session_state.pending_question is not None:
        ask(st.session_state.pending_question)
        st.session_state.pending_question = None

    # Suggested questions as buttons, plus a clear button
    columns = st.columns(len(SUGGESTED_QUESTIONS) + 1)
    for column, question in zip(columns, SUGGESTED_QUESTIONS):
        if column.button(question, width="stretch"):
            ask(question)
    columns[-1].button("🗑️ Clear chat", on_click=clear_chat, width="stretch")

    # Typed question (the input box stays at the bottom of the page)
    typed = st.chat_input("Ask about your skin, acne, sunscreen, routines...")
    if typed:
        ask(typed)

    # Show the whole conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Two spaces before \n make a line break in Markdown
            st.markdown(message["text"].replace("\n", "  \n"))


# ------------------------------------------------------------
# Page 4: CV Steps
# ------------------------------------------------------------


def threshold_table(result):
    """Rows for the table of raw numbers and the thresholds used."""
    raw = result["raw"]
    return [
        {"Measurement": "Skin tone", "Raw value": "L = " + str(raw["lightness_L"]),
         "Thresholds": "Fair ≥ " + str(sa.TONE_FAIR_MIN_L) + ", Medium ≥ "
                       + str(sa.TONE_MEDIUM_MIN_L) + ", Tan ≥ " + str(sa.TONE_TAN_MIN_L)
                       + ", else Deep",
         "Result": result["tone"]},
        {"Measurement": "Oiliness", "Raw value": str(raw["oily_percent"]) + "% shiny",
         "Thresholds": "shiny = V > " + str(sa.OILY_MIN_BRIGHTNESS) + " and S < "
                       + str(sa.OILY_MAX_SATURATION) + "; Dry < "
                       + str(sa.OILY_DRY_MAX_PERCENT) + "%, Normal ≤ "
                       + str(sa.OILY_NORMAL_MAX_PERCENT) + "%, else Oily",
         "Result": result["oiliness"]},
        {"Measurement": "Redness", "Raw value": str(raw["red_percent"]) + "% red (mean a = "
                                                + str(raw["mean_a"]) + ")",
         "Thresholds": "red = a > " + str(sa.RED_A_THRESHOLD) + "; Low < "
                       + str(sa.RED_LOW_MAX_PERCENT) + "%, Moderate ≤ "
                       + str(sa.RED_MODERATE_MAX_PERCENT) + "%, else High",
         "Result": result["redness"]},
        {"Measurement": "Dark spots", "Raw value": str(raw["spot_count"]) + " spots",
         "Thresholds": "darker than blur by > " + str(sa.SPOT_DIFF_THRESHOLD) + ", area "
                       + str(sa.SPOT_MIN_AREA) + "-" + str(sa.SPOT_MAX_AREA) + " px; many ≥ "
                       + str(MANY_SPOTS_COUNT),
         "Result": str(result["spots"])},
        {"Measurement": "Texture", "Raw value": "variance = " + str(raw["texture_variance"]),
         "Thresholds": "Smooth < " + str(sa.TEXTURE_SMOOTH_MAX) + ", Normal ≤ "
                       + str(sa.TEXTURE_NORMAL_MAX) + ", else Rough",
         "Result": result["texture"]},
    ]


def step_title(number, title):
    """Numbered step heading."""
    html('<span class="step-number">' + str(number) + '</span> <b>' + title + '</b>')


def fit_to_tile(image):
    """Resize an image to fit the tile box (keeping its shape) and pad the rest,
    so every image on the CV Steps page has exactly the same size."""
    # Single-channel images (masks, channels) are turned into 3 channels first
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Scale so the image fits inside TILE_WIDTH x TILE_HEIGHT
    height, width = image.shape[:2]
    scale = min(TILE_WIDTH / width, TILE_HEIGHT / height)
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    image = cv2.resize(image, (new_w, new_h))

    # Put the image in the middle of a tile filled with the card colour
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), TILE_PAD_COLOR, dtype=np.uint8)
    top = (TILE_HEIGHT - new_h) // 2
    left = (TILE_WIDTH - new_w) // 2
    tile[top:top + new_h, left:left + new_w] = image
    return tile


def show_tile(column, image, title, caption):
    """One bordered tile: title, image (always the same size) and a short caption."""
    with column:
        with st.container(border=True):
            html('<div class="tile-title">' + title + '</div>')
            st.image(to_rgb(fit_to_tile(image)), width="stretch")
            html('<div class="tile-caption">' + caption + '</div>')


def crop_box(image, box):
    """Cut a box (x, y, w, h) out of an image, staying inside the borders."""
    x, y, w, h = box
    return image[max(0, y):y + h, max(0, x):x + w]


def make_spot_views(analysis):
    """Spots of all regions drawn on the face: (circled image, black and white mask)."""
    image = analysis["image"]
    circled = image.copy()
    mask_full = np.zeros(image.shape[:2], dtype=np.uint8)

    for region in analysis["regions"].values():
        roi = region["roi"]
        if roi.size == 0:
            continue
        x, y = max(0, region["box"][0]), max(0, region["box"][1])

        # Same spot detection as the analyzer, on this one region
        spots, mask = sa.find_spots_in_roi(roi)

        # Put the region's mask back at its place in the full-size mask
        mask_full[y:y + mask.shape[0], x:x + mask.shape[1]] = mask

        # Circle each spot (shift from ROI coordinates to image coordinates)
        for spot in spots:
            (cx, cy), radius = cv2.minEnclosingCircle(spot)
            cv2.circle(circled, (x + int(cx), y + int(cy)), int(radius) + 4, (0, 0, 255), 2)

    # Show only the face area, so the spots are big enough to see
    face = analysis["face"]
    return crop_box(circled, face), crop_box(mask_full, face)


def make_laplacian_view(roi):
    """Laplacian of a ROI, scaled to 0-255 so the fine detail is visible."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (sa.TEXTURE_BLUR_SIZE, sa.TEXTURE_BLUR_SIZE), 0)
    laplacian = np.absolute(cv2.Laplacian(gray, cv2.CV_64F))
    return cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def cv_steps_page():
    st.title("🧪 Computer Vision Steps")

    analysis = st.session_state.analysis
    if analysis is None:
        st.info("No analysis yet. Open the Analyze page and upload a photo, "
                "then come back here to see every step.")
        st.button("Go to Analyze →", on_click=go_to, args=(ANALYZE,))
        return

    image = analysis["image"]
    regions = analysis["regions"]
    cheek = regions["left_cheek"]["roi"]

    # Images needed for the tiles
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    lab = cv2.cvtColor(cheek, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(cheek, cv2.COLOR_BGR2HSV)
    spots_circled, spots_mask = make_spot_views(analysis)

    # Each row: one step with three tiles of the same size
    rows = [
        (1, "Face detection", [
            (image, "Original image", "Resized to max " + str(MAX_IMAGE_WIDTH) + " px wide"),
            (gray, "Grayscale + equalised", "What the Haar cascade looks at"),
            (analysis["boxes_image"], "Haar cascade result", "Largest face + region boxes"),
        ]),
        (2, "ROI cropping", [
            (regions["forehead"]["roi"], "Forehead", "Top of the face box, middle part"),
            (regions["left_cheek"]["roi"], "Left cheek", "Below the eye, left of the nose"),
            (regions["right_cheek"]["roi"], "Right cheek", "Below the eye, right of the nose"),
        ]),
        (3, "Colour spaces (left cheek)", [
            (lab[:, :, 1], "LAB a channel", "Brighter = redder (used for redness)"),
            (hsv[:, :, 2], "HSV V channel", "Brighter = shinier (used for oiliness)"),
            (get_skin_mask(cheek), "YCrCb skin mask", "White = skin pixels (tone, redness)"),
        ]),
        (4, "Spots and texture", [
            (spots_mask, "Spot mask", "Blur subtraction + threshold + opening"),
            (spots_circled, "Spots found", "Contours with spot-sized area, circled"),
            (make_laplacian_view(cheek), "Laplacian (left cheek)",
             "Brighter = more fine detail (texture)"),
        ]),
    ]

    for number, title, tiles in rows:
        step_title(number, title)
        for column, (tile_image, tile_title, caption) in zip(st.columns(3), tiles):
            show_tile(column, tile_image, tile_title, caption)

    # Raw numbers and thresholds
    st.subheader("Raw numbers and thresholds")
    st.table(threshold_table(analysis["result"]))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

init_session()
html(CSS)
show_sidebar_top()

# Draw the chosen page (each page is a plain function)
if st.session_state.page == HOME:
    home_page()
elif st.session_state.page == ANALYZE:
    analyze_page()
elif st.session_state.page == CHATBOT:
    chatbot_page()
else:
    cv_steps_page()

show_sidebar_bottom()
