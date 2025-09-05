# Dragon Boat Practice Auto-Editor 🚣‍♂️

This project is a Python script designed to automate the editing of dragon boat practice footage. It uses computer vision to detect other boats in a collection of video files and automatically combines these segments into a single, consolidated highlight reel.

---

## Features

* **Automatic Boat Detection:** Uses the YOLOv8 object detection model to find boats in video frames.
* **Batch Processing:** Scans a folder and processes all `.mp4` and `.MOV` video files it contains.
* **Automated Editing:** Automatically cuts the segments where boats are detected and merges them into one final video.
* **Configurable:** Easily adjust settings like detection confidence and analysis precision.

---

## Requirements

* Python 3.13+
* Windows 10/11
* An NVIDIA GPU is recommended for faster processing but is not required.

---

## First-Time Setup Instructions

Follow these steps to set up the project for the first time.

**1. Download the Project**
   Download all the files and place them in a folder on your computer.

**2. Install Python**
   If you don't have it, install Python 3.13 from the official website. **Important:** On the first screen of the installer, make sure to check the box that says **"Add python.exe to PATH"**.

**3. Open a Terminal**
   Open a terminal (PowerShell or Command Prompt) in the project's main folder.

**4. Create the Virtual Environment**
   Create an isolated environment for the project by running the command `py -3.13 -m venv venv`.

**5. Activate the Virtual Environment**
   Activate the environment by running `.\venv\Scripts\activate`. You should see `(venv)` appear in your terminal prompt.
   *Note for PowerShell users: If you get a security error, first run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`, press `Y`, and then try activating again.*

**6. Install Required Libraries**
   Install all necessary Python packages by running `pip install -r requirements.txt`.

---

## How to Use

Once the setup is complete, follow this simple workflow to process your videos.

**1. Add Your Footage**
   Place all your raw video files (e.g., `practice_01.mp4`, `gopro_view.MOV`) inside the **`raw_footage`** folder.

**2. Run the Script**
   Make sure your virtual environment is active (`(venv)` is visible) and run the main script with the command `python process_folder.py`.

**3. Get Your Video**
   The script will analyze all the videos and create a single highlight reel. You will find the final video, named **`practice_highlights.mp4`**, in the main project folder.

---

## Configuration

You can fine-tune the script's performance by editing these variables at the top of the `process_folder.py` file:

* **`VIDEO_FOLDER`**: The name of the folder containing your raw footage.
* **`OUTPUT_FILE`**: The filename for the final edited video.
* **`CONFIDENCE_THRESHOLD`**: How "sure" the model needs to be to detect a boat (e.g., `0.5` = 50%). Lower this if it's missing distant boats.
* **`CHECK_EVERY_N_SECONDS`**: How often to analyze a frame. A smaller number (`0.5`) is more precise but much slower. A larger number (`2`) is faster but less precise.