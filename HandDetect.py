import cv2
import mediapipe

mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands()
mp_draw = mediapipe.solutions.drawing_utils

img = cv2.imread("Test_landmark.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = hands.process(img_rgb)

if result.multi_hand_landmarks:
    for landmarks in result.multi_hand_landmarks:
        mp_draw.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
