import cv2
import numpy as np
import tensorflow as tf
import os
import glob

# --- Configuration --- #
WEED_COLOR_LOWER = np.array([0, 100, 100])      # HSV lower for "weed"
WEED_COLOR_UPPER = np.array([10, 255, 255])
PLANT_COLOR_LOWER = np.array([35, 100, 100])    # HSV lower for "plant"
PLANT_COLOR_UPPER = np.array([85, 255, 255])
ROBOT_SPEED = 20  # (Unused but kept for your context)

# --- Load ML Model --- #
def load_detection_model():
    print("Loading trained model: weed_plant_detector.keras")
    try:
        model = tf.keras.models.load_model("weed_plant_detector.keras")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'weed_plant_detector.keras' is in the same directory as main.py and h5py is installed.")
        return None

# --- Inference/Detection --- #
def predict_objects(image, model):
    if model:
        # Resize and normalize for model
        input_image = cv2.resize(image, (128, 128))
        input_image = np.expand_dims(input_image, axis=0) / 255.0
        predictions = model.predict(input_image)
        # (You can replace with your own post-processing if you have a real detection model)
        # For demo, fallback to color-based detection for visualization
        return simulate_color_based_detection(image)
    else:
        return simulate_color_based_detection(image)

def simulate_color_based_detection(image):
    """Color-based fallback detection for plants and weeds, returns bounding boxes."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    weeds = []
    weed_mask = cv2.inRange(hsv_image, WEED_COLOR_LOWER, WEED_COLOR_UPPER)
    weed_contours, _ = cv2.findContours(weed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in weed_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            weeds.append({"class": "weed", "box": [x, y, w, h]})

    plants = []
    plant_mask = cv2.inRange(hsv_image, PLANT_COLOR_LOWER, PLANT_COLOR_UPPER)
    plant_contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in plant_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:
            plants.append({"class": "plant", "box": [x, y, w, h]})

    return weeds + plants

# --- Robot Navigation (Unused in dashboard, used in simulation) --- #
def robot_navigate(detections, image_width, image_height):
    weeds = [d for d in detections if d["class"] == "weed"]
    plants = [d for d in detections if d["class"] == "plant"]

    if weeds:
        weed_x_center = weeds[0]["box"][0] + weeds[0]["box"][2] / 2
        if weed_x_center < image_width / 3:
            return "turn_left"
        elif weed_x_center > 2 * image_width / 3:
            return "turn_right"
        else:
            return "move_forward"
    elif plants:
        plant_x_centers = [p["box"][0] + p["box"][2] / 2 for p in plants]
        if len(plants) == 1:
            if plant_x_centers[0] < image_width / 2:
                return "turn_right"
            else:
                return "turn_left"
        else:
            return "move_forward"
    else:
        return "move_forward"

# --- Main Simulation Loop --- #
def run_simulation(image_paths, output_dir="./simulation_output"):
    model = load_detection_model()
    os.makedirs(output_dir, exist_ok=True)
    img_width, img_height = 640, 480

    log_file_path = os.path.join(output_dir, "robot_log.txt")
    with open(log_file_path, "w") as log_file:
        log_file.write("Unkaraut Roboter Simulation Log\n")
        log_file.write("-----------------------------------\n\n")

        for i, img_path in enumerate(image_paths):
            log_file.write(f"Processing image: {img_path}\n")
            image = cv2.imread(img_path)
            if image is None:
                log_file.write(f"Error: Could not load image {img_path}\n")
                continue

            image = cv2.resize(image, (img_width, img_height))

            detections = predict_objects(image, model)
            action = robot_navigate(detections, img_width, img_height)

            log_file.write(f"Detected: {detections}, Robot Action: {action}\n")

            display_image = annotate_image_with_detections(image, detections)
            output_img_path = os.path.join(output_dir, f"processed_frame_{i}.jpg")
            cv2.imwrite(output_img_path, display_image)
            log_file.write(f"Saved processed image to {output_img_path}\n\n")

    print("Simulation finished. Check simulation_output directory for results.")

# --- Annotation Helper: Draw boxes and labels (used in both simulation & GUI) --- #
def annotate_image_with_detections(image, detections):
    display_image = image.copy()
    for det in detections:
        x, y, w, h = det["box"]
        color = (0, 0, 255) if det["class"] == "weed" else (0, 255, 0)
        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display_image, det["class"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return display_image

# --- API for Streamlit GUI: Use identical detection/annotation as simulation --- #
def process_image_simulation(image):
    """Public function for dashboard.py: Detects & annotates image exactly like the simulation."""
    model = load_detection_model()
    img_resized = cv2.resize(image, (640, 480))
    detections = predict_objects(img_resized, model)
    annotated = annotate_image_with_detections(img_resized, detections)
    return annotated

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    check_folder_path = os.path.join(current_script_dir, "check")

    image_files = glob.glob(os.path.join(check_folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(check_folder_path, "*.png"))

    if not image_files:
        print(f"No images found in {check_folder_path}. Please ensure your images are there.")
    else:
        run_simulation(image_files)
