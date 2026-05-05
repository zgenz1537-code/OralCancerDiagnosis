import os
import cv2

INPUT_DIR = "Data/data"
OUTPUT_DIR = "Data/preprocessed"
IMG_SIZE = (224, 224)  
 
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print(f"Skipping invalid image: {img_path}")
        return None

    img = cv2.resize(img, IMG_SIZE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img

def process_dataset(input_dir, output_dir):
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, file)
                processed = preprocess_image(input_path)

                if processed is not None:
                    cv2.imwrite(output_path, processed)
                    print(f"Saved: {output_path}")
                    
                    
if __name__ == "__main__":
    
    process_dataset(INPUT_DIR, OUTPUT_DIR)
    print("Preprocessing complete.")