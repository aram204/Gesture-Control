# Face & Gesture Controlled Drone

This project integrates face recognition and gesture control with a Tello drone. The system first recognizes a user's face and then allows them to control the drone using hand gestures. It uses FaceNet for face recognition, Mediapipe for hand gesture detection, and the `djitellopy` library to communicate with the Tello drone.

## Features

- **Face Recognition:** The drone recognizes the user using a pretrained FaceNet model and starts accepting gesture commands only from recognized users.
- **Gesture Control:** Hand gestures are used to control drone movements such as flipping, moving up/down/left/right/forward/back, and landing.
- **Real-time Video Stream:** The drone streams video, which is processed for both face recognition and gesture detection.
- **Safety Check:** The drone takes off automatically if no recognized face is detected within 30 seconds.
- **Logging:** Captures video and stores individual frames for further analysis.

## Requirements

- Python 3.x
- `opencv-python`
- `numpy`
- `tensorflow` (compat.v1)
- `facenet` and `detect_face` modules
- `djitellopy`
- `mediapipe`
- `Pillow`
- `scipy`
- `scikit-learn`
- `imageio`

## Files

- `FaceGestureRecognition.py`: Main script that handles face recognition and gesture control.
- `facenet.py`: Module containing functions for building and managing the FaceNet face recognition model.
- `model/20180402-114759.pb`: Pretrained FaceNet model.
- `class/classifier.pkl`: Classifier trained on the user's images.
- `train_img/`: Directory containing images of users for training.
- `npy/`: Directory used by the MTCNN for temporary files.

## How It Works

1. **Drone Setup:** Connect to the Tello drone and start the video stream.
2. **Face Recognition:**
   - Load the pretrained FaceNet model.
   - Detect faces in the video stream using MTCNN.
   - Compute embeddings for detected faces and classify them using a pretrained classifier.
   - If a recognized face (e.g., `Aram`) is detected, the drone prepares for gesture control.
3. **Gesture Control:**
   - Use Mediapipe to detect hand landmarks.
   - Determine finger positions and orientation.
   - Perform drone actions based on recognized gestures:
     - `[2,3]` -> Flip
     - `[2,3,4]` -> Move Up/Down/Left/Right based on hand orientation
     - `[1,2,3,4]` -> Move Forward/Back or Land based on finger orientation
4. **Logging:** Video frames are saved to `output.avi` and individual frames as `img#.jpg`.

## Running the Project

1. Install all required packages.
2. Ensure that the Tello drone is powered on and connected to your computer.
3. Place the FaceNet model and classifier files in the appropriate directories.
4. Run the main script:

```bash
python FaceGestureRecognition.py
```

5. The drone will:
   - Recognize the user.
   - Start gesture-based control if the recognized face is detected.
   - Automatically take off if no recognized face is detected for 30 seconds.

6. Press `q` to land the drone and stop the program.

## Technical Details

- **Face Recognition:**
  - Uses MTCNN for face detection.
  - Uses FaceNet to compute embeddings.
  - Classifies faces using a pretrained SVM classifier.
  - Only proceeds with drone control if the recognized face probability is above 0.998.
- **Gesture Detection:**
  - Mediapipe detects hand landmarks.
  - Finger positions and hand orientation determine drone commands.
  - Supports flipping, directional movements, forward/back movements, and landing.
- **Drone Control:**
  - `djitellopy` library is used to send movement commands to the Tello drone.

## Safety Considerations

- Always operate the drone in a safe, open space.
- Ensure the drone has enough battery before running the project.
- Be ready to manually land the drone if gesture recognition fails.

## License

MIT License. See the `facenet.py` header for licensing information related to FaceNet functions.
