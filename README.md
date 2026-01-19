# VideoDetectionModule-RAS
### Installation:
The __first step__ is to install the required python dependencies using:
``` pip install -r requirements.txt ```

The __second step__ is to install the dlib 68-landmark predictor model:
1.) Navigate to https://dlib.net/files/.
2.) Download ```"shape_predictor_68_face_landmarks.dat.bz2"```.
3.) Decompress the file using ```bzip2 -d shape_predictor_68_face_landmarks.dat.bz2"```.
&emsp;3a.) Windows users can alternatively use WinRAR for decompression.
4.) Move the file into the directory ```VideoDetectionModule-RAS/models/```.
&emsp;4a.) Models placed in other directories need to be referenced with the ```--dlib_model``` command line argument.


