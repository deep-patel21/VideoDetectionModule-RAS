# VideoDetectionModule-RAS
### Installation:
The __first step__ is to install the required python dependencies using:
``` pip install -r requirements.txt ```

The __second step__ is to install the dlib 68-landmark predictor model:
1. Navigate to https://dlib.net/files/.
2. Download ```"shape_predictor_68_face_landmarks.dat.bz2"```.
3. Decompress the file using ```bzip2 -d shape_predictor_68_face_landmarks.dat.bz2"```.
&emsp;3a. Windows users can alternatively use WinRAR for decompression.
4. Move the file into the directory ```VideoDetectionModule-RAS/models/```.
&emsp;4a. Models placed in other directories need to be referenced with the ```--dlib_model``` command line argument.


### Usage
The script offers various command line arguments for users to customize their experience.

```
usage: detect_drowsiness.py [-h] [-ll {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-v TARGET_FPS] [-m DLIB_MODEL] [-s] [-e EAR_THRESHOLD]      

options:
  -h, --help            show this help message and exit
  -ll {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level to use with logging library
  -v TARGET_FPS, --target_fps TARGET_FPS
                        Target FPS for video processing
  -m DLIB_MODEL, --dlib_model DLIB_MODEL
                        Path to the Dlib landmark prediction model
  -s, --show_simple     Show landmarks on a black canvas instead of raw video
  -e EAR_THRESHOLD, --ear_threshold EAR_THRESHOLD
                        Eye aspect ratio threshold for detecting drowsiness
```




