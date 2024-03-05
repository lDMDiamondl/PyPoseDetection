import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 세 정점의 각도 구하기
def calcAngle (a, b, c):
    radians = np.arctan2 (c[1] - b[1], c[0] - b[0]) - np.arctan2 (a[1] - b[1], a[0] - b[0])
    angle = np.abs (radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle # 각도를 0 ~ 180의 범위로 지정한 뒤 출력

cap = cv2.VideoCapture (0)

# Mediapipe 인스턴스 초기 설정
with mp_pose.Pose (min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened ():
        ret, frame = cap.read ()
        
        # 이미지를 RGB로 변환한 뒤 자세 감지, 이후 원상복구
        image = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process (image)
        image.flags.writeable = True
        image = cv2.cvtColor (image, cv2.COLOR_RGB2BGR)
        
        # 랜드마크 추출
        try:
            landmarks = results.pose_landmarks.landmark

            # Define landmark names
            landmarkName = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", 
                            "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
            
            # 랜드마크 좌표를 저장할 딕셔너리
            landmarksDict = {}

            # 랜드마크의 인덱스와 좌표를 딕셔너리에 저장
            for name in landmarkName:
                landmarkIndex = mp_pose.PoseLandmark[name].value
                landmarkCoords = [landmarks[landmarkIndex].x, landmarks[landmarkIndex].y]
                landmarksDict[name] = landmarkCoords

            # 좌표 추출
            leftShoulder = landmarksDict["LEFT_SHOULDER"]
            leftElbow = landmarksDict["LEFT_ELBOW"]
            leftWrist = landmarksDict["LEFT_WRIST"]
            rightShoulder = landmarksDict["RIGHT_SHOULDER"]
            rightElbow = landmarksDict["RIGHT_ELBOW"]
            rightWrist = landmarksDict["RIGHT_WRIST"]
            leftHip = landmarksDict["LEFT_HIP"]
            leftKnee = landmarksDict["LEFT_KNEE"]
            leftAnkle = landmarksDict["LEFT_ANKLE"]
            rightHip = landmarksDict["RIGHT_HIP"]
            rightKnee = landmarksDict["RIGHT_KNEE"]
            rightAnkle = landmarksDict["RIGHT_ANKLE"]

            # 각도 계산
            leftArmAngle = calcAngle (leftShoulder, leftElbow, leftWrist)
            rightArmAngle = calcAngle (rightShoulder, rightElbow, rightWrist)
            leftArmRaisingAngle = calcAngle (leftElbow, leftShoulder, leftHip)
            rightArmRaisingAngle = calcAngle (rightElbow, rightShoulder, rightHip)
            leftLegAngle = calcAngle (leftHip, leftKnee, leftAnkle)
            rightLegAngle = calcAngle (rightHip, rightKnee, rightAnkle)
            leftHipAngle = calcAngle (leftShoulder, leftHip, leftKnee)
            rightHipAngle = calcAngle (rightShoulder, rightHip, rightKnee)

            # 글씨 뒤에 반투명한 직사각형 그리기
            overlay = image.copy ()
            opacity = 0.5
            cv2.rectangle (image, (0, 0), (330, 180), (0, 0, 0), -1)
            cv2.addWeighted (overlay, opacity, image, 1 - opacity, 0, image)
            overlay = image.copy ()
            cv2.rectangle (image, (0, 430), (200, 480), (0, 0, 0), -1)
            cv2.addWeighted (overlay, opacity, image, 1 - opacity, 0, image)

            # 각도 출력
            angles = {
                "leftArm" : leftArmAngle,
                "rightArm" : rightArmAngle,
                "leftArmRaising" : leftArmRaisingAngle,
                "rightArmRaising" : rightArmRaisingAngle,
                "leftLeg" : leftLegAngle,
                "rightLeg" : rightLegAngle,
                "leftHip" : leftHipAngle,
                "rightHip" : rightHipAngle
            }

            posY = 20 # 정보를 20px의 간격으로 출력
            for label, angle in angles.items ():
                cv2.putText (image, f"{label}: {angle}", [10, posY], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                posY += 20

            # 자세의 상태 체크 후 출력
            standOrSitAngle = [leftLegAngle, rightLegAngle, leftHipAngle, rightHipAngle]

            # 서 있는지 앉아 있는지, 아니면 둘 다 아닌지를 결정
            isStanding = ("STANDING" if all (150 <= angle <= 180 for angle in standOrSitAngle)
                            else "SITTING" if all (75 <= angle <= 135 for angle in standOrSitAngle)
                            else "ERROR")

            # 어느 손을 들고 있는지를 결정
            isRaising = ("BOTH" if 120 <= leftArmRaisingAngle <= 180 and 120 <= rightArmRaisingAngle <= 180
                        else "LEFT" if 120 <= leftArmRaisingAngle <= 180
                        else "RIGHT" if 120 <= rightArmRaisingAngle <= 180
                        else "NOT RAISING")

            cv2.putText (image, "isStanding: " + isStanding, [10, 450], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText (image, "isRaising: " + isRaising, [10, 470], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 좌표를 정점의 위치에 출력하는 방법입니다. 필요에 따라 저 형식의 코드를 사용해도 됩니다
            # cv2.putText (image, str (leftArmAngle),
            #             tuple (np.multiply (rightElbow, [640, 480]).astype (int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # cv2.putText (image, str (rightArmAngle),
            #             tuple (np.multiply (rightElbow, [640, 480]).astype (int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        except:
            pass
        
        # 스켈레톤 그리기
        mp_drawing.draw_landmarks (image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec (color = (245, 66, 230), thickness = 2, circle_radius = 2), # 정점, 필요에 따라 색을 바꿔도 됩니다
                                mp_drawing.DrawingSpec (color = (0, 255, 0), thickness = 2, circle_radius = 2)) # 간선, 필요에 따라 색을 바꿔도 됩니다
        
        cv2.imshow ("Camera", image)

        # q로 프로그램 종료
        if cv2.waitKey (10) & 0xFF == ord ('q'):
            break
    cap.release ()
    cv2.destroyAllWindows ()
