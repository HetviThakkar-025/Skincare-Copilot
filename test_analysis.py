# test_analysis.py
# Quick test for skin_analyzer.py
#   Webcam mode: python test_analysis.py   (press "s" to analyze, "q" to quit)
#   Photo mode:  python test_analysis.py path/to/photo.jpg

import sys
import cv2
from face_detector import load_image, detect_face, get_regions, draw_regions
from skin_analyzer import analyze_skin

MAIN_WINDOW = "Skin Analysis Test"
SPOT_WINDOW = "Dark Spots (left: circled, right: mask)"


def print_results(results):
    """Print the labels and all raw numbers in the terminal."""
    print("\n===== Skin Analysis =====")
    print("Skin tone :", results["tone"])
    print("Oiliness  :", results["oiliness"])
    print("Redness   :", results["redness"])
    print("Dark spots:", results["spots"])
    print("Texture   :", results["texture"])
    print("--- Raw values ---")
    for name, value in results["raw"].items():
        print("  " + name + ":", value)
    print("=========================")


def analyze_frame(frame):
    """Detect the face, analyze it, print results and show the spot window.
    Returns the image with boxes drawn (or None if no face)."""
    face = detect_face(frame)
    if face is None:
        print("No face found. Face the camera directly, with good light.")
        return None

    regions = get_regions(frame, face)
    results, spot_image = analyze_skin(regions)
    print_results(results)

    if spot_image is not None:
        cv2.imshow(SPOT_WINDOW, spot_image)
    return draw_regions(frame, face, regions)


def run_webcam():
    """Live webcam. Press 's' to analyze the current frame, 'q' to quit."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: could not open the webcam.")
        return

    print("Webcam started. Press 's' to analyze, 'q' to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Error: could not read a frame from the webcam.")
            break

        # Show the boxes live so the user can position their face
        face = detect_face(frame)
        if face is not None:
            display = draw_regions(frame, face, get_regions(frame, face))
        else:
            display = frame.copy()
        cv2.putText(display, "s = analyze, q = quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(MAIN_WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Analyze the clean frame (without drawings on it)
            analyze_frame(frame)
        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def run_photo(path):
    """Analyze a photo and show the results until any key is pressed."""
    image = load_image(path)
    if image is None:
        print("Error: could not read the image file:", path)
        return

    output = analyze_frame(image)
    if output is None:
        output = image

    print("Press any key on an image window to close.")
    cv2.imshow(MAIN_WINDOW, output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # If a file path is given, use photo mode, otherwise use the webcam
    if len(sys.argv) > 1:
        run_photo(sys.argv[1])
    else:
        run_webcam()
