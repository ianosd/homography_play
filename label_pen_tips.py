import os
import cv2
import numpy as np
import csv
import re
import sys

# Configuration parameters
S = 100  # Size of squares
OVERLAP = S // 2
ZOOM_FACTOR = 4  # Magnification factor for zoomed-in squares

# Global variables for click handling
selected_square = None
selected_point = None
zoomed_square_img = None
original_img = None
current_image_name = None
output_folder = "output"


def get_squares(image, s, overlap):
    """Divide the image into overlapping squares."""
    h, w = image.shape[:2]
    squares = []
    for y in range(0, h - s + 1, s - overlap):
        for x in range(0, w - s + 1, s - overlap):
            square = (x, y, x + s, y + s)
            squares.append(square)
    return squares


def find_closest_square(squares, click_point):
    """Find the square whose center is closest to the click point."""
    x_click, y_click = click_point
    closest_square = None
    min_distance = float('inf')
    for square in squares:
        x1, y1, x2, y2 = square
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        distance = np.sqrt((cx - x_click) ** 2 + (cy - y_click) ** 2)
        if distance < min_distance:
            closest_square = square
            min_distance = distance
    return closest_square


def mouse_callback(event, x, y, flags, param):
    global selected_square, zoomed_square_img, selected_point, original_img

    if event == cv2.EVENT_LBUTTONDOWN:
        if selected_square is None:  # First click: Select the square
            squares = param["squares"]
            selected_square = find_closest_square(squares, (x, y))
            if selected_square:
                x1, y1, x2, y2 = selected_square
                zoomed_square_img = cv2.resize(
                    original_img[y1:y2, x1:x2],
                    (S * ZOOM_FACTOR, S * ZOOM_FACTOR),
                    interpolation=cv2.INTER_LINEAR,
                )
                cv2.imshow("Zoomed Square", zoomed_square_img)
                cv2.setMouseCallback("Zoomed Square", mouse_callback_zoomed_image, {})

def mouse_callback_zoomed_image(event, x, y, flags, param):
    global selected_point, selected_square
    if event == cv2.EVENT_LBUTTONDOWN:
        x_relative = x / ZOOM_FACTOR
        y_relative = y / ZOOM_FACTOR
        x1, y1, _, _ = selected_square
        selected_point = (x1 + x_relative, y1 + y_relative)
        print(f"Selected sub-pixel point: {selected_point}")

def list_frames(folder):
    for file in os.listdir(folder):
        match = re.match(r'(\w+)[_-](\d+)\.(jpg|jpeg|png)', file)
        if not match:
            continue
        yield (match.group(1), int(match.group(2)), file)

def get_frameinfo(filename):
    match = re.match(r"^([a-zA-Z]+)_(\d+)", filename)
    return (match.group(1), int(match.group(2)))
        
def annotate_images(input_folder, output_folder):
    global selected_square, selected_point, zoomed_square_img, original_img, current_image_name

    # Prepare the output folder
    os.makedirs(output_folder, exist_ok=True)
    csv_file = os.path.join(output_folder, "annotations.csv")

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        annotated_frames = [get_frameinfo(line[0]) for line in reader]

    print(annotated_frames)

    # Open CSV for writing annotations
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)

        frames = list_frames(input_folder)
        print(frames)
        # Process each image in the input folder
        for frame_data in sorted(list_frames(input_folder)):
            print(frame_data[0:2])
            if frame_data[0:2] in annotated_frames:
                continue
            image_name = frame_data[2]
            image_path = os.path.join(input_folder, image_name)

            current_image_name = image_name
            original_img = cv2.imread(image_path)
            if original_img is None:
                print(f"Skipping invalid image: {image_name}")
                continue

            squares = get_squares(original_img, S, OVERLAP)
            
            # Show the first image
            cv2.imshow("Image", original_img)
            cv2.setMouseCallback("Image", mouse_callback, {"squares": squares})

            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("a") and selected_point and selected_square:  # Accept annotation
                    selected_squares = [sq for sq in squares if 
                                        sq[0] <= selected_point[0] < sq[2] and 
                                        sq[1] <= selected_point[1] < sq[3]]
                    
                    for square in selected_squares:
                        x1, y1, x2, y2 = square
                        square_image = original_img[y1:y2, x1:x2]
                        square_name = f"{os.path.splitext(image_name)[0]}_{x1}_{y1}.png"
                        square_path = os.path.join(output_folder, square_name)
                        cv2.imwrite(square_path, square_image)

                        writer.writerow([square_name, selected_point[0] - x1, selected_point[1] - y1])
                        print(f"Annotation saved: {square_path}, Point: {selected_point}")

                    selected_square = None
                    selected_point = None
                    cv2.destroyWindow("Zoomed Square")
                    break

                if key == ord("d"):
                    writer.writerow([image_name, -1, -1])
                    print(f"Annotation saved: {image_name} contains no data")
                    selected_square = None
                    selected_point = None
                    cv2.destroyWindow("Zoomed Square")
                    break

                if key == ord("n"):  # Next image
                    selected_square = None
                    selected_point = None
                    cv2.destroyWindow("Zoomed Square")
                    break

                if key == 27:  # ESC key to exit
                    cv2.destroyAllWindows()
                    return

    print("Annotation process completed.")


# Main entry point
if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    annotate_images(input_folder, output_folder)
