# test_face.py
# Quick test for face_detector.py
#   Webcam mode: python test_face.py
#   Photo mode:  python test_face.py path/to/photo.jpg

import sys
import cv2
from face_detector import load_image, detect_face, get_regions, draw_regions

WINDOW_NAME = "Face Detection Test"


def process_frame(frame):
    """Detect the face, crop regions and return an image with boxes drawn (or None if no face)."""
    face = detect_face(frame)
    if face is None:
        return None
    regions = get_regions(frame, face)
    return draw_regions(frame, face, regions)


def run_webcam():
    """Show the live webcam with face and region boxes. Press 'q' to quit."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: could not open the webcam.")
        return

    print("Webcam started. Press 'q' to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Error: could not read a frame from the webcam.")
            break

        output = process_frame(frame)
        if output is None:
            # No face: show the plain frame with a message
            output = frame.copy()
            cv2.putText(output, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, output)

        # waitKey(1) waits 1 ms for a key, which also lets the window refresh
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def run_photo(path):
    """Load a photo, draw the boxes, show it until any key is pressed."""
    image = load_image(path)
    if image is None:
        print("Error: could not read the image file:", path)
        return

    output = process_frame(image)
    if output is None:
        print("No face found in the photo. Try a clear, front-facing photo with good light.")
        output = image

    print("Press any key on the image window to close it.")
    cv2.imshow(WINDOW_NAME, output)
    cv2.waitKey(0)  # 0 = wait forever for a key press
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # If a file path is given, use photo mode, otherwise use the webcam
    if len(sys.argv) > 1:
        run_photo(sys.argv[1])
    else:
        run_webcam()
