# skin_analyzer.py
# All computer vision measurements on the skin regions (ROIs) from face_detector.py.
# Every function takes the regions dictionary: {"forehead": {"roi": ..., "box": ...}, ...}

import cv2
import numpy as np

# ------------------------------------------------------------
# CONFIG - all thresholds and class boundaries are kept here
# ------------------------------------------------------------
# Note: OpenCV stores 8-bit LAB, HSV and YCrCb values in the range 0-255
# (except HSV hue, which is 0-179). In LAB, a = 128 means "neutral" (neither green nor red).

CHEEK_NAMES = ["left_cheek", "right_cheek"]
ALL_REGION_NAMES = ["forehead", "left_cheek", "right_cheek"]

# --- Skin mask (YCrCb) ---
# Skin pixels fall in a well-known Cr/Cb range for most skin tones.
# Pixels outside it (hair, eyebrows, background) are ignored for tone and redness.
SKIN_CR_MIN = 133
SKIN_CR_MAX = 190   # well above the usual 173, so very red (irritated) skin is still counted
SKIN_CB_MIN = 77
SKIN_CB_MAX = 127
# If fewer skin pixels than this are found (strange lighting), use all pixels instead
MIN_SKIN_PIXELS = 50

# --- Skin tone (LAB lightness L, 0 = black, 255 = white) ---
TONE_FAIR_MIN_L = 170     # L >= 170          -> Fair
TONE_MEDIUM_MIN_L = 140   # 140 <= L < 170    -> Medium
TONE_TAN_MIN_L = 110      # 110 <= L < 140    -> Tan
                          # L < 110           -> Deep

# --- Oiliness (HSV shiny highlights) ---
# A shiny (oily) spot reflects light: very bright (high V) and almost white (low S)
# (strict values, so normal fair skin is not mistaken for shine)
OILY_MIN_BRIGHTNESS = 220   # V must be above this
OILY_MAX_SATURATION = 40    # S must be below this
OILY_DRY_MAX_PERCENT = 1.0     # shine < 1%       -> Dry
OILY_NORMAL_MAX_PERCENT = 5.0  # 1% to 5%         -> Normal
                               # shine > 5%       -> Oily

# --- Redness (LAB "a" channel, green-red axis) ---
RED_A_THRESHOLD = 155          # a pixel with a > 155 counts as "red"
RED_LOW_MAX_PERCENT = 10.0       # red pixels < 10%   -> Low
RED_MODERATE_MAX_PERCENT = 30.0  # 10% to 30%         -> Moderate
                                 # red pixels > 30%   -> High

# --- Dark spots ---
SPOT_BLUR_SIZE = 15        # Gaussian blur size (odd), estimates the "normal" skin around each pixel
SPOT_DIFF_THRESHOLD = 15   # a pixel this much darker than its surroundings is a spot pixel
SPOT_KERNEL_SIZE = 3       # morphological opening kernel, removes single-pixel noise
SPOT_MIN_AREA = 4          # smaller blobs are noise (pixels)
SPOT_MAX_AREA = 200        # bigger blobs are not spots (eyebrow, shadow, hair)

# --- Texture (Laplacian variance) ---
TEXTURE_BLUR_SIZE = 3            # light blur first so camera noise is not counted as texture
TEXTURE_SMOOTH_MAX = 50.0        # variance < 50     -> Smooth
TEXTURE_NORMAL_MAX = 150.0       # 50 to 150         -> Normal
                                 # variance > 150    -> Rough (suggests dryness)

# --- Display ---
SPOT_PANEL_WIDTH = 500           # width of each row in the spot window (pixels)
SPOT_CIRCLE_COLOR = (0, 0, 255)  # red (BGR)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------


def get_rois(regions, names):
    """Return the non-empty ROIs for the given region names."""
    rois = []
    for name in names:
        roi = regions[name]["roi"]
        # A region can be empty if the face is at the edge of the image
        if roi.size > 0:
            rois.append(roi)
    return rois


def get_skin_mask(roi):
    """Return a mask (255 = skin, 0 = not skin) using the YCrCb colour space."""
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, SKIN_CR_MIN, SKIN_CB_MIN], dtype=np.uint8)
    upper = np.array([255, SKIN_CR_MAX, SKIN_CB_MAX], dtype=np.uint8)
    # Y (brightness) can be anything, only Cr and Cb decide "skin colour"
    return cv2.inRange(ycrcb, lower, upper)


def get_pixels(regions, names, conversion, skin_only):
    """Convert the ROIs to a colour space and return all their pixels as one N x 3 array."""
    all_pixels = []
    for roi in get_rois(regions, names):
        converted = cv2.cvtColor(roi, conversion)
        pixels = converted.reshape(-1, 3)

        if skin_only:
            # Keep only the pixels the YCrCb mask marks as skin
            mask = get_skin_mask(roi).reshape(-1)
            skin_pixels = pixels[mask > 0]
            # If almost nothing looks like skin (odd lighting), keep every pixel
            if len(skin_pixels) >= MIN_SKIN_PIXELS:
                pixels = skin_pixels

        all_pixels.append(pixels)

    if len(all_pixels) == 0:
        return None
    return np.concatenate(all_pixels)


# ------------------------------------------------------------
# Measurements
# ------------------------------------------------------------


def get_skin_tone(regions):
    """Average cheek lightness (LAB L). Returns (label, L value)."""
    pixels = get_pixels(regions, CHEEK_NAMES, cv2.COLOR_BGR2LAB, skin_only=True)
    if pixels is None:
        return "Unknown", 0.0

    # Column 0 is L (lightness)
    lightness = float(np.mean(pixels[:, 0]))

    if lightness >= TONE_FAIR_MIN_L:
        label = "Fair"
    elif lightness >= TONE_MEDIUM_MIN_L:
        label = "Medium"
    elif lightness >= TONE_TAN_MIN_L:
        label = "Tan"
    else:
        label = "Deep"
    return label, lightness


def get_oiliness(regions):
    """Percentage of shiny pixels (bright + low saturation in HSV). Returns (label, percent)."""
    # Use all pixels here: shiny white highlights do not pass the skin colour mask
    pixels = get_pixels(regions, ALL_REGION_NAMES, cv2.COLOR_BGR2HSV, skin_only=False)
    if pixels is None:
        return "Unknown", 0.0

    saturation = pixels[:, 1]
    brightness = pixels[:, 2]

    # A pixel is "shiny" if it is very bright AND almost colourless
    shiny = (brightness > OILY_MIN_BRIGHTNESS) & (saturation < OILY_MAX_SATURATION)
    percent = 100.0 * np.count_nonzero(shiny) / len(pixels)

    if percent < OILY_DRY_MAX_PERCENT:
        label = "Dry"
    elif percent <= OILY_NORMAL_MAX_PERCENT:
        label = "Normal"
    else:
        label = "Oily"
    return label, percent


def get_redness(regions):
    """Percentage of red cheek pixels (LAB a channel). Returns (label, percent, mean a)."""
    pixels = get_pixels(regions, CHEEK_NAMES, cv2.COLOR_BGR2LAB, skin_only=True)
    if pixels is None:
        return "Unknown", 0.0, 0.0

    # Column 1 is "a": higher value = more red
    a_channel = pixels[:, 1]
    mean_a = float(np.mean(a_channel))

    # Count the pixels above the redness threshold
    percent = 100.0 * np.count_nonzero(a_channel > RED_A_THRESHOLD) / len(pixels)

    if percent < RED_LOW_MAX_PERCENT:
        label = "Low"
    elif percent <= RED_MODERATE_MAX_PERCENT:
        label = "Moderate"
    else:
        label = "High"
    return label, percent, mean_a


def find_spots_in_roi(roi):
    """Find dark spots in one ROI. Returns (list of spot contours, cleaned mask)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # The blurred image shows the "normal" skin brightness around each pixel
    blurred = cv2.GaussianBlur(gray, (SPOT_BLUR_SIZE, SPOT_BLUR_SIZE), 0)

    # blurred - original is large where a pixel is darker than its surroundings.
    # cv2.subtract stops at 0, so brighter pixels just become 0.
    difference = cv2.subtract(blurred, gray)

    # Keep only pixels that are clearly darker than the skin around them
    _, mask = cv2.threshold(difference, SPOT_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Opening (erode then dilate) removes tiny specks of noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPOT_KERNEL_SIZE, SPOT_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Each separate white blob in the mask is a possible spot
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep blobs that are spot-sized (not noise, not eyebrows/shadows)
    spots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if SPOT_MIN_AREA <= area <= SPOT_MAX_AREA:
            spots.append(contour)
    return spots, mask


def make_spot_panel(name, roi, spots, mask):
    """Make one display row: the ROI with spots circled next to its mask."""
    circled = roi.copy()
    for spot in spots:
        # Draw a circle slightly bigger than the spot
        (cx, cy), radius = cv2.minEnclosingCircle(spot)
        cv2.circle(circled, (int(cx), int(cy)), int(radius) + 3, SPOT_CIRCLE_COLOR, 1)

    # Put the ROI and its (black and white) mask side by side
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    row = np.hstack([circled, mask_bgr])

    # Scale the row up so the small ROIs are easy to see
    scale = SPOT_PANEL_WIDTH / row.shape[1]
    new_size = (SPOT_PANEL_WIDTH, max(1, int(row.shape[0] * scale)))
    row = cv2.resize(row, new_size, interpolation=cv2.INTER_NEAREST)

    # Label the row with the region name and number of spots
    text = name + ": " + str(len(spots)) + " spots"
    cv2.putText(row, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # White line at the bottom to separate this row from the next one
    bottom = row.shape[0] - 1
    cv2.line(row, (0, bottom), (row.shape[1], bottom), (255, 255, 255), 1)
    return row


def get_spots(regions):
    """Count dark spots in all regions. Returns (spot count, display image)."""
    total = 0
    panels = []

    for name in ALL_REGION_NAMES:
        roi = regions[name]["roi"]
        if roi.size == 0:
            continue
        spots, mask = find_spots_in_roi(roi)
        total += len(spots)
        panels.append(make_spot_panel(name, roi, spots, mask))

    if len(panels) == 0:
        return 0, None

    # Stack the rows on top of each other (all rows have the same width)
    spot_image = np.vstack(panels)
    return total, spot_image


def get_texture(regions):
    """Average Laplacian variance of the ROIs. Returns (label, variance)."""
    variances = []
    for roi in get_rois(regions, ALL_REGION_NAMES):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Light blur so camera noise is not mistaken for skin texture
        gray = cv2.GaussianBlur(gray, (TEXTURE_BLUR_SIZE, TEXTURE_BLUR_SIZE), 0)

        # The Laplacian responds to edges and fine detail.
        # High variance = lots of fine detail = rough/flaky skin.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(laplacian.var())

    if len(variances) == 0:
        return "Unknown", 0.0

    variance = float(np.mean(variances))

    if variance < TEXTURE_SMOOTH_MAX:
        label = "Smooth"
    elif variance <= TEXTURE_NORMAL_MAX:
        label = "Normal"
    else:
        label = "Rough"
    return label, variance


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------


def analyze_skin(regions):
    """Run every measurement. Returns (results dictionary, spot display image)."""
    tone, lightness = get_skin_tone(regions)
    oiliness, oily_percent = get_oiliness(regions)
    redness, red_percent, mean_a = get_redness(regions)
    spot_count, spot_image = get_spots(regions)
    texture, texture_variance = get_texture(regions)

    results = {
        "tone": tone,
        "oiliness": oiliness,
        "redness": redness,
        "spots": spot_count,
        "texture": texture,
        # Raw numbers, rounded so they are easy to read and explain
        "raw": {
            "lightness_L": round(float(lightness), 1),
            "oily_percent": round(float(oily_percent), 2),
            "red_percent": round(float(red_percent), 2),
            "mean_a": round(float(mean_a), 1),
            "spot_count": spot_count,
            "texture_variance": round(float(texture_variance), 1),
        },
    }
    return results, spot_image
