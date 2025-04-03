import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle = 360-angle

    return round(angle, 2)

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            LEFT_HIP_INDEX = mp_pose.PoseLandmark.LEFT_HIP.value
            LEFT_KNEE_INDEX = mp_pose.PoseLandmark.LEFT_KNEE.value
            LEFT_ANKLE_INDEX = mp_pose.PoseLandmark.LEFT_ANKLE.value
            LEFT_SHOULDER_INDEX = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            LEFT_ELBOW_INDEX = mp_pose.PoseLandmark.LEFT_ELBOW.value
            LEFT_WRIST_INDEX = mp_pose.PoseLandmark.LEFT_WRIST.value

            RIGHT_HIP_INDEX = mp_pose.PoseLandmark.RIGHT_HIP.value
            RIGHT_KNEE_INDEX = mp_pose.PoseLandmark.RIGHT_KNEE.value
            RIGHT_ANKLE_INDEX = mp_pose.PoseLandmark.RIGHT_ANKLE.value
            RIGHT_SHOULDER_INDEX = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            RIGHT_ELBOW_INDEX = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            RIGHT_WRIST_INDEX = mp_pose.PoseLandmark.RIGHT_WRIST.value

            # Get coordinates
            left_hip = [landmarks[LEFT_HIP_INDEX].x,landmarks[LEFT_HIP_INDEX].y]
            left_knee = [landmarks[LEFT_KNEE_INDEX].x,landmarks[LEFT_KNEE_INDEX].y]
            left_ankle = [landmarks[LEFT_ANKLE_INDEX].x,landmarks[LEFT_ANKLE_INDEX].y]
            left_shoulder = [landmarks[LEFT_SHOULDER_INDEX].x,landmarks[LEFT_SHOULDER_INDEX].y]
            left_elbow = [landmarks[LEFT_ELBOW_INDEX].x,landmarks[LEFT_ELBOW_INDEX].y]
            left_wrist = [landmarks[LEFT_WRIST_INDEX].x,landmarks[LEFT_WRIST_INDEX].y]

            right_hip = [landmarks[RIGHT_HIP_INDEX].x,landmarks[RIGHT_HIP_INDEX].y]
            right_knee = [landmarks[RIGHT_KNEE_INDEX].x,landmarks[RIGHT_KNEE_INDEX].y]
            right_ankle = [landmarks[RIGHT_ANKLE_INDEX].x,landmarks[RIGHT_ANKLE_INDEX].y]
            right_shoulder = [landmarks[RIGHT_SHOULDER_INDEX].x,landmarks[RIGHT_SHOULDER_INDEX].y]
            right_elbow = [landmarks[RIGHT_ELBOW_INDEX].x,landmarks[RIGHT_ELBOW_INDEX].y]
            right_wrist = [landmarks[RIGHT_WRIST_INDEX].x,landmarks[RIGHT_WRIST_INDEX].y]

            # Calculate angles
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            right_shoulder_angle = calculate_angle(right_hip, right_shoulder, right_elbow)
            right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angles
            cv2.putText(image, str(left_knee_angle),
                           tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(left_shoulder_angle),
                           tuple(np.multiply(left_shoulder, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(left_elbow_angle),
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(right_knee_angle),
                           tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(right_shoulder_angle),
                           tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, str(right_elbow_angle),
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )

        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
