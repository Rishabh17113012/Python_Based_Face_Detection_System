import cv2
import os
import sys

# Create a folder to store the images for each person
person_name = input("Enter the person's name: ")
save_path = f"dataset/{person_name}"
os.makedirs(save_path, exist_ok=True)

print(f"[INFO] Saving images to: {save_path}")

# Initialize the webcam with explicit backend for macOS
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("[ERROR] Could not access camera.")
    print("[INFO] Make sure camera permissions are granted in System Settings.")
    sys.exit(1)

# Set camera properties for better compatibility
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("[INFO] Camera initialized successfully")
print("[INFO] Press 'q' to quit or 's' to save current frame")
print("[INFO] Auto-saving every 10th frame...")

count = 0
saved_count = 0
max_images = 100

try:
    while saved_count < 10:  # Collect at least 10 images
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read frame from webcam.")
            break

        # Display the video feed with proper window settings
        try:
            cv2.namedWindow("Face Data Collection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Face Data Collection", 640, 480)
            
            # Add text overlay
            text = f"Saved: {saved_count}/10 | Press 'q' to quit, 's' to save"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow("Face Data Collection", frame)
        except cv2.error as e:
            print(f"[WARNING] Display error (continuing anyway): {e}")
            # Continue without display if imshow fails

        # Auto-save every 10th frame
        if count % 10 == 0 and count > 0:
            img_name = f"{save_path}/{person_name}_{saved_count}.jpg"
            success = cv2.imwrite(img_name, frame)
            if success:
                print(f"[INFO] Saved: {img_name}")
                saved_count += 1
            else:
                print(f"[ERROR] Failed to save: {img_name}")

        count += 1

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Press 's' to manually save
        if key == ord('s'):
            img_name = f"{save_path}/{person_name}_{saved_count}.jpg"
            success = cv2.imwrite(img_name, frame)
            if success:
                print(f"[INFO] Manually saved: {img_name}")
                saved_count += 1
            else:
                print(f"[ERROR] Failed to save: {img_name}")
        
        # Press 'q' to quit
        elif key == ord('q'):
            print("[INFO] Quit requested by user")
            break

        # Safety limit
        if count > max_images:
            print("[INFO] Maximum frame count reached")
            break

except KeyboardInterrupt:
    print("\n[INFO] Collection interrupted by user")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
finally:
    # Release the webcam and close windows
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    print(f"\n[SUMMARY] Collection complete!")
    print(f"  - Total frames processed: {count}")
    print(f"  - Images saved: {saved_count}")
    print(f"  - Save location: {save_path}")
    
    if saved_count == 0:
        print("\n[WARNING] No images were saved!")
        print("  This might be due to camera permission issues.")
        print("  Check: System Settings > Privacy & Security > Camera")
    else:
        print(f"\n[SUCCESS] Collected {saved_count} images for {person_name}")