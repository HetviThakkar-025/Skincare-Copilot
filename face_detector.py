# face_detector.py
# Face detection with a Haar cascade and cropping of skin regions (ROIs).

import cv2

# ------------------------------------------------------------
# CONFIG - all tunable values are kept here
# ------------------------------------------------------------

# Haar cascade file that ships with OpenCV
CASCADE_FILE = "haarcascade_frontalface_default.xml"

# How much the image is shrunk at each detection scale (1.1 = 10% per step).
# Smaller value = more careful search, but slower.
HAAR_SCALE_FACTOR = 1.1

# How many overlapping detections are needed to accept a face.
# Higher value = fewer false faces, but real faces may be missed.
HAAR_MIN_NEIGHBORS = 5

# Smallest face size (in pixels) we care about. Tiny faces are too
# small to measure skin properly anyway.
HAAR_MIN_FACE_SIZE = (80, 80)

# Large photos are resized to this width (aspect ratio kept), so detection
# is fast and all images are measured at a similar scale.
MAX_IMAGE_WIDTH = 800

# Region proportions, as fractions of the face box (0.0 = top/left, 1.0 = bottom/right).
# The Haar face box starts around the middle of the forehead and ends near the chin.

# Forehead: top 5% to 20% of the face height, middle 50% of the width.
# We stop at 20% so the eyebrows (around 25-30% of the height) are not included,
# and keep to the middle so hair at the sides is not included.
FOREHEAD_TOP = 0.05
FOREHEAD_BOTTOM = 0.20
FOREHEAD_LEFT = 0.25
FOREHEAD_RIGHT = 0.75

# Cheeks: 50% to 72% of the face height (below the eyes, above the mouth corners).
CHEEK_TOP = 0.50
CHEEK_BOTTOM = 0.72

# Left cheek (on the left side of the image): 15% to 38% of the width.
# Starts at 15% to skip the background/hair at the face edge.
LEFT_CHEEK_LEFT = 0.15
LEFT_CHEEK_RIGHT = 0.38

# Right cheek (on the right side of the image): 62% to 85% of the width.
# The gap in the middle (38% to 62%) skips the nose.
RIGHT_CHEEK_LEFT = 0.62
RIGHT_CHEEK_RIGHT = 0.85

# Drawing colours (OpenCV uses BGR order)
FACE_BOX_COLOR = (0, 255, 0)      # green
REGION_BOX_COLOR = (255, 128, 0)  # blue
LABEL_COLOR = (255, 255, 255)     # white

# ------------------------------------------------------------

# Load the cascade once when the module is imported
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_FILE)


def resize_image(image):
    """Shrink very large images to MAX_IMAGE_WIDTH, keeping the aspect ratio."""
    height, width = image.shape[:2]
    if width > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / width
        new_size = (MAX_IMAGE_WIDTH, int(height * scale))
        # INTER_AREA gives the best quality when shrinking
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def load_image(path):
    """Read an image file. Returns the image (BGR) or None if it can't be read."""
    image = cv2.imread(path)

    # cv2.imread returns None if the file is missing or not an image
    if image is None:
        return None

    return resize_image(image)


def detect_face(frame):
    """Find the largest face in the frame. Returns (x, y, w, h) or None."""
    # Haar cascades work on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Even out the brightness so detection works better in poor lighting
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBORS,
        minSize=HAAR_MIN_FACE_SIZE,
    )

    if len(faces) == 0:
        return None

    # Keep the biggest face (largest area w * h), usually the person closest to the camera
    largest = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest
    return (int(x), int(y), int(w), int(h))


def make_box(face, left, right, top, bottom):
    """Turn fractions of the face box into a pixel box (x, y, w, h)."""
    fx, fy, fw, fh = face
    x = fx + int(fw * left)
    y = fy + int(fh * top)
    w = int(fw * (right - left))
    h = int(fh * (bottom - top))
    return (x, y, w, h)


def crop(frame, box):
    """Cut a box out of the frame, staying inside the image borders."""
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]

    # Clamp the corners so we never go outside the image
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    # .copy() so later drawing on the frame does not change the ROI
    return frame[y1:y2, x1:x2].copy()


def get_regions(frame, face):
    """Return a dict of skin regions. Each entry has the cropped "roi" and its "box"."""
    boxes = {
        "forehead": make_box(face, FOREHEAD_LEFT, FOREHEAD_RIGHT,
                             FOREHEAD_TOP, FOREHEAD_BOTTOM),
        "left_cheek": make_box(face, LEFT_CHEEK_LEFT, LEFT_CHEEK_RIGHT,
                               CHEEK_TOP, CHEEK_BOTTOM),
        "right_cheek": make_box(face, RIGHT_CHEEK_LEFT, RIGHT_CHEEK_RIGHT,
                                CHEEK_TOP, CHEEK_BOTTOM),
    }

    regions = {}
    for name, box in boxes.items():
        regions[name] = {"roi": crop(frame, box), "box": box}
    return regions


def draw_regions(frame, face, regions):
    """Draw the face box and region boxes with labels. Returns a new image for display."""
    # Draw on a copy so the original frame stays clean for analysis
    output = frame.copy()

    # Face box
    x, y, w, h = face
    cv2.rectangle(output, (x, y), (x + w, y + h), FACE_BOX_COLOR, 2)
    cv2.putText(output, "face", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, FACE_BOX_COLOR, 2)

    # One box + label for each region
    for name, region in regions.items():
        rx, ry, rw, rh = region["box"]
        cv2.rectangle(output, (rx, ry), (rx + rw, ry + rh), REGION_BOX_COLOR, 2)
        cv2.putText(output, name, (rx, ry - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, LABEL_COLOR, 1)

    return output
