import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from google.colab.patches import cv2_imshow ## only for colab
from utils.display import *

# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
lm_dict = {
  0:0 , 1:10, 2:12, 3:14, 4:16, 5:11, 6:13, 7:15, 8:24, 9:26, 10:28, 11:23, 12:25, 13:27, 14:5, 15:2, 16:8, 17:7,
}



def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)


    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue, clf = set_video_feed_variables("../demo.mp4")
    


    #Start video feed and run workout
    while cap.isOpened():
        #Getting image from camera
        ret, img = cap.read() 
        #Getting video dimensions
        width  = cap.get(3)  
        height = cap.get(4)  
        
        #Convert from BGR (used by cv2) to RGB (used by Mediapipe)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        #Get pose and draw landmarks
        img = get_pose(img, results, False)
        
        # Get landmark list from mediapipe
        landmark_list = get_position(img, results, height, width, False)
        
        #If landmarks exist, get the relevant workout body angles and run workout. The points used are identifiers for specific joints
        if len(landmark_list) != 0:

            elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle = set_body_angles_from_keypoints(get_angle, img, landmark_list)
            # print(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle)
            workout_name_after_smoothening = set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list)    

            workout_name_after_smoothening = workout_name_after_smoothening.replace("Workout Name:", "").strip()
            pushup_success_percentage, pushup_progress_bar = set_percentage_bar_and_text(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, workout_name_after_smoothening)
        
                    
            #Is the form correct at the start?
            form = check_form(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, form, workout_name_after_smoothening)
        
            #Full workout motion
            
            feedback,count,direction = run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, pushup_success_percentage, feedback, workout_name_after_smoothening)
            
     
            
            #Display workout stats        
            display_workout_stats(count, form, feedback, draw_percentage_progress_bar, display_rep_count, show_workout_feedback, show_workout_name_from_model, img, pushup_success_percentage, pushup_progress_bar, workout_name_after_smoothening)
            
        print(workout_name_after_smoothening,feedback,count)
        # Transparent Overlay
        overlay = img.copy()
        x, y, w, h = 75, 10, 500, 150
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)      
        alpha = 0.8  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)          
        # cv2_imshow(image_new)#use it only on Colab
        #cv2.imshow('DEEPFIT Workout Trainer', image_new) 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
