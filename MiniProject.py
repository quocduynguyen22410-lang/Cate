import cv2
import numpy as np
import mediapipe as mp
import random

def gen_ball():
    return {
        "x": random.randint(35, 685),
        "y": 35,
        "color": tuple(map(lambda x: int(x), np.random.randint(0, 255, size = 3))),
        "speed": 4
    }

def collission(x1, y1, x2, y2, rad1, rad2):
    return (x1 - x2)**2 + (y1 - y2)**2 <= (rad1 + rad2)**2

def off(vid):
    vid.release()
    cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
width, height = 820, 680
radius = 35
balls = [gen_ball() for _ in range(3)]
score = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    fx, fy = 0, 0
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            for id, lms in enumerate(hand_lms.landmark):
                cx, cy = int(width*lms.x), int(height*lms.y)
                if id == 8:
                    cv2.circle(frame, (cx, cy), 25, (255, 0, 255), -1)
                    fx, fy = cx, cy
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    game_over = False
    for ball in balls:
        ball["y"] += ball["speed"]
        cv2.circle(frame, (ball["x"], ball["y"]), radius, ball["color"], -1)
        if collission(ball["x"], ball["y"], fx, fy, radius, 25):
            ball.update(gen_ball())
            score += 1
        
        if ball["y"] >= height:
            game_over = True

    if game_over:
        cv2.putText(frame, f"Total: {score}", (265, 300), cv2.FONT_ITALIC, 2, (0, 255, 255), 2)
        cv2.putText(frame, "Continue?", (245, 410), cv2.FONT_ITALIC, 2, (0, 255, 255), 2)
        cv2.imshow("Video", frame)
        if cv2.waitKey(0) == ord('y'):
            balls = [gen_ball() for _ in range(3)]
            score = 0
        else:
            break

    if not game_over:
        cv2.putText(frame, f"Score: {score}", (10, 70), cv2.FONT_ITALIC, 2, (0, 255, 0), 2)    
        if score % 6 == 0 and score != 0:
            for ball in balls:
                ball["speed"] += 1

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == ord('q'):
        break

off(cap)
