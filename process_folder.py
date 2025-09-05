from pathlib import Path
import cv2
from ultralytics import YOLO
import math
from moviepy.editor import VideoFileClip, concatenate_videoclips

# --- 1. Configuration: UPDATE THESE VALUES ---

# The folder where you will put all your raw video files.
VIDEO_FOLDER = "raw_footage"

# The name of the final combined highlight video.
OUTPUT_FILE = "practice_highlights.mp4"

# --- Set the output resolution ---
# Set the desired height in pixels. The width will be adjusted automatically.
#   - 1080p (Full HD) = 1080
#   - 1440p (2k QHD)  = 1440
#   - To keep the original resolution of the video, set this to: None
TARGET_RESOLUTION = None 

# --- Detection settings (you can tune these) ---

# CONFIDENCE_THRESHOLD controls the detection's ACCURACY. It's how "sure" the AI
# needs to be before it calls an object a "boat" (from 0.0 to 1.0).
#   - Raise to 0.7 to be more "strict." This reduces false alarms but might miss distant boats.
#   - Lower to 0.4 to be more "lenient." This finds distant boats better but may misidentify other objects.
CONFIDENCE_THRESHOLD = 0.5

# CHECK_EVERY_N_SECONDS controls the script's SPEED vs. PRECISION. It's how
# often the script analyzes the video.
#   - Raise to 2.0 for a much faster analysis. The start/end times of clips may be less precise.
#   - Lower to 0.5 for a slower but more precise analysis that catches very brief moments.
CHECK_EVERY_N_SECONDS = 1

# --- 2. The Analysis Function (with a small optimization) ---

def analyze_video(video_path, model):
    """
    Analyzes a video to find timestamps where a "boat" is visible.
    The YOLO model is passed in to avoid reloading it for every video.
    """
    print(f"\nAnalyzing: {video_path.name}...")
    target_class_id = 8

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path.name}.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_skip = math.floor(fps * CHECK_EVERY_N_SECONDS)
    
    timestamps = []
    boat_is_visible = False
    start_time = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_to_skip == 0:
            current_time_sec = frame_count / fps
            results = model(frame, verbose=False)
            found_boat = any(
                box.cls == target_class_id and box.conf >= CONFIDENCE_THRESHOLD
                for result in results for box in result.boxes
            )
            
            if found_boat and not boat_is_visible:
                boat_is_visible = True
                start_time = current_time_sec
                print(f"  -> Boat spotted at {start_time:.2f}s in {video_path.name}")
            
            elif not found_boat and boat_is_visible:
                boat_is_visible = False
                end_time = current_time_sec
                timestamps.append((start_time, end_time))
                print(f"  -> Segment finished: ({start_time:.2f}s, {end_time:.2f}s)")

        frame_count += 1
    
    if boat_is_visible:
        end_time = frame_count / fps
        timestamps.append((start_time, end_time))
        print(f"  -> Final segment (video ended): ({start_time:.2f}s, {end_time:.2f}s)")
        
    cap.release()
    return timestamps

# --- 3. Main Execution Block ---

def main():
    """
    Main function to run the batch processing.
    """
    # Define the input and output paths.
    input_folder = Path(VIDEO_FOLDER)
    output_path = Path(OUTPUT_FILE)

    # Check if the input folder exists.
    if not input_folder.is_dir():
        print(f"Error: Input folder '{VIDEO_FOLDER}' not found.")
        print("Please create it and put your video files inside.")
        return

    # Find all video files in the folder (add other extensions if needed).
    video_files = list(input_folder.glob("*.mp4")) + list(input_folder.glob("*.MOV"))
    if not video_files:
        print(f"No video files (.mp4, .MOV) found in '{VIDEO_FOLDER}'.")
        return

    print("--- Starting Dragon Boat Highlight Reel Generator ---")
    print(f"Found {len(video_files)} video(s) to process.")

    # Load the YOLO model ONCE to save time.
    print("Loading object detection model...")
    yolo_model = YOLO("yolov8n.pt")

    all_clips = []
    # Loop through each video file.
    for video_path in video_files:
        # Step 1: Analyze the video to get timestamps.
        timestamps = analyze_video(video_path, yolo_model)
        
        # Step 2: Create subclips from the timestamps.
        if timestamps:
            video = VideoFileClip(str(video_path))
            for start, end in timestamps:
                if start < end:
                    subclip = video.subclip(start, end)
                    
                    # NEW: Resize the clip if a target resolution is set.
                    if TARGET_RESOLUTION:
                        subclip = subclip.resize(height=TARGET_RESOLUTION)
                    
                    all_clips.append(subclip)

            # IMPORTANT: Close the video file to free up memory.
            video.close()

    # Step 3: Combine all collected clips into one video.
    if not all_clips:
        print("\nNo boat segments were detected in any videos. No highlight video created.")
        return
        
    print(f"\nFinished analysis. Found {len(all_clips)} total segments.")
    print("Combining all clips into the final video... (This may take a while)")

    final_video = concatenate_videoclips(all_clips)
    final_video.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    print("\n--- All Done! ---")
    print(f"Your final highlight reel is saved as: {output_path.name}")

if __name__ == "__main__":
    main()