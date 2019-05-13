import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import rect_to_bb
import glob
import os.path
#In order to run this script, you will need all the dependencies above.
#Other than that, you will also need this trained NN file: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

#Path of the Cohn-kanade original database, from where the images will be read
input_db_path = "C:\\Users\\lavi\\cohn-kanade-images"

#Path of the cropped database, where the images will be saved
output_db_path = "C:\\Users\\lavi\\cohn-kanade-eyes-only"

def split_face(image, shape, margin_w=0):
    #The index of each point can be seen in https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ (Figure 2)
        #The indexes shown in Figure 2 are starting at 1, and in the coding they start at 0, so to access the point 'n' in the image, we must call index 'n-1'
        #shape[n] is a point's coordenate: shape[n][0] is the x-axis and shape[n][1] is the y-axis

    #X-points belonging to the right eye + right eyebrow region
    reye_left = min(shape[42][0], shape[22][0])
    reye_right = max(shape[45][0], shape[26][0])
    reye_width = reye_right - reye_left

    #Y-points belonging to right eye + eyebrow regions
    reye_top = min([y for (x, y) in shape[17:21]])
    reye_bot = max([y for (x, y) in shape[36:47]])
    reye_height = reye_bot - reye_top

    #Cropping region: most left and most right points of the right eye/eyebrow region
    p_left = max(0, int(reye_left - 10))
    p_right = min(len(image)-1, int(reye_right + reye_width*margin_w/100))

    #Height compensation so that the it crops a square
    height_comp = p_right - p_left - reye_height 
    if height_comp < 0:
        height_comp = 0

    #Cropping region: most top and most bottom points of the right eye/eyebrow region
    p_top = max(0, int(reye_top - height_comp*0.5))
    p_bot = min(len(image)-1, int(reye_bot + height_comp*0.5))

    #Cropping the right eye
    reye = image[p_top:p_bot, p_left:p_right] 

    #X-points belonging to the left eye + right eyebrow region
    leye_left = min(shape[36][0], shape[17][0])
    leye_right = max(shape[39][0], shape[21][0])
    leye_width = leye_right - leye_left

    #Y-points belonging to left eye + eyebrow regions
    leye_top = min([y for (x, y) in shape[17:21]])
    leye_bot = max([y for (x, y) in shape[36:47]])
    leye_height = leye_bot - leye_top

    #Cropping region: most left and most right points of the left eye/eyebrow region
    p_left = max(0, int(leye_left - leye_width*margin_w/100))
    p_right = min(len(image)-1, int(leye_right + 10))

    #Height compensation so that the it crops a square
    height_comp = p_right - p_left - leye_height
    if height_comp < 0:
        height_comp = 0

    #Cropping region: most top and most bottom points of the left eye/eyebrow region
    p_top = max(0, int(leye_top - height_comp*0.5))
    p_bot = min(len(image)-1, int(leye_bot + height_comp*0.5))

    #Cropping the left eye
    leye = image[p_top:p_bot, p_left:p_right]

    return leye, reye

for dirpath, dirnames, filenames in os.walk(input_db_path):
    for filename in [f for f in filenames if f.endswith(".png")]:

        path_list = dirpath.split(os.sep)

        #The class name is the name of the folder two levels before the image (that's why the '-2' below), because Cohn-Kanade follows
        # the rule:  "CLASS_NAME/SAMPLE_NUMBER/IMAGE.png".
        #In other databses, the '-2' parameters might have to be modified. 
        class_name = path_list[-2]

        file_path = os.path.join(dirpath, filename)
        file_name = os.path.splitext(filename)[0]


        #Loads the facial detector
        detector = dlib.get_frontal_face_detector()

        #Loads the shape predictor (which marks the 68 points)
        #You can download a trained facial shape predictor from:
        #   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2        
        predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

        #Loads image
        image = cv2.imread(file_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Detects the coordineates of the face
        dets = detector(rgb, 1)

        #If it detects more than one face (or no faces), it ignores the image
        if len(dets) != 1:
            continue

        #Creates the output path, if it doesn't exist
        class_folder = os.path.join(output_db_path, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        #Marks the 68 points
        shape = predictor(rgb, dets[0])
        shape = face_utils.shape_to_np(shape)

        #Crops the right and left eyes, using the width margin as 10% (the cropped area will be 10% larger in the direction opposed to the nose)
        leye, reye = split_face(image, shape, margin_w=10)
        leye_file = file_name + "_le.jpg"
        reye_file = file_name + "_re.jpg"

        #Generate the right eye and the left eye images
        cv2.imwrite(os.path.join(class_folder, leye_file), leye)
        cv2.imwrite(os.path.join(class_folder, reye_file), reye)

        print("Class " + class_name + " - Cropped file: " + file_name)