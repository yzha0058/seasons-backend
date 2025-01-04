import cv2
from src.BodyShapeAnalyzer import ImageSegmentationProcessor, PoseSegmentationVisualizer

if __name__ == '__main__':
    # Path to the image
    image_path = "1.jpg"
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}. Ensure the file exists and the path is correct.")
    else:
        print("Image loaded successfully.")
    
    # Initialize the PoseSegmentationVisualizer
    try:
        three_d_model = PoseSegmentationVisualizer(image, model_path="selfie_segmenter.tflite")
        print("PoseSegmentationVisualizer initialized successfully.")
        
        # Perform segmentation or visualization as required
        result = three_d_model.process_and_visualize()  # Replace `.visualize()` with the appropriate method if different
        print("Segmentation completed.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
