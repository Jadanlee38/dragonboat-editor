import cv2
from ultralytics import YOLO
import math
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# --- 1. Configuration: UPDATE THESE VALUES ---

# Enter the exact filename of your video.
# Make sure the video file is in the same folder as this script.
VIDEO_FILE = "your_video.mp4" 

# How sure the model needs to be to count a detection (0.0 to 1.0). 
# 0.5 is 50% sure.
CONFIDENCE_THRESHOLD = 0.5 

# How often to check the video (in seconds). 
# A smaller number is more precise but much slower. 1 or 2 is a good start.
CHECK_EVERY_N_SECONDS = 1

# --- 2. The Analysis Function ---

def analyze_video(video_path):
    """
    Analyzes a video to find timestamps where a "boat" is visible.
    Returns a list of (start_time, end_time) tuples in seconds.
    """
    print(f"Analyzing video: {video_path}...")
    
    # The YOLO model will be downloaded automatically on the first run.
    model = YOLO("yolov8n.pt") 
    target_class_id = 8  # This is the class ID for "boat" in the YOLO model.

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
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

        # Only process frames according to the interval we set.
        if frame_count % frames_to_skip == 0:
            current_time_sec = frame_count / fps
            
            # Run the object detection model on the current frame.
            results = model(frame, verbose=False)
            
            found_boat = False
            for result in results:
                for box in result.boxes:
                    if box.cls == target_class_id and box.conf >= CONFIDENCE_THRESHOLD:
                        found_boat = True
                        break
                if found_boat:
                    break
            
            # Logic to start and stop the timer for each segment.
            if found_boat and not boat_is_visible:
                boat_is_visible = True
                start_time = current_time_sec
                print(f"  -> Boat spotted at {start_time:.2f} seconds.")
            
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
    print("Video analysis complete.")
    return timestamps

# --- 3. The Video Editing Function ---

def create_highlight_video(video_path, timestamps, output_filename):
    """
    Cuts a video based on timestamps and saves the result.
    """
    if not timestamps:
        print("No boat segments found to create a highlight video.")
        return

    print(f"\nCreating highlight video from {video_path}...")
    
    original_video = VideoFileClip(video_path)
    
    clips = []
    for start, end in timestamps:
        # Create a subclip for each detected segment.
        if start < end:
            clips.append(original_video.subclip(start, end))

    if not clips:
        print("No valid clips to concatenate.")
        original_video.close()
        return

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_filename, codec="libx24", audio_codec="aac")
    
    original_video.close()
    print(f"Highlight video successfully saved to: {output_filename}")


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    # Part 1: Analyze the video to get timestamps.
    detected_timestamps = analyze_video(VIDEO_FILE)
    
    # Part 2: Use the timestamps to create the new video file.
    if detected_timestamps:
        output_name = os.path.splitext(VIDEO_FILE)[0] + "_highlights.mp4"
        create_highlight_video(VIDEO_FILE, detected_timestamps, output_filename=output_name)