import cv2
import pandas as pd
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
EXPECTED_THUMP, EXPECTED_VECTOR_AND_LENGTH, EXPECTED_CORNER = 5, 20, 14
Algo_corner = [i for i in range(0, 5)] + [0] + [i for i in range(5, 17)] + [0] + [i for i in range(17, 21)]

vec = [f"{x}{i}" for i in range(4, 22, 4) for x in ['vector_x', 'vector_y', 'vector_z', 'length_vector']]
corner = [f"angle{i}" for i in range(1, 15)] 
base = [f"distance base ({i}, {i + 4})" for i in range(1, 14, 4)]
tip = [f"distance tip ({i, i + 4})" for i in range(4, 17, 4)]

column = ["LABEL"] + vec + base + tip + corner
test_df = pd.DataFrame(columns=column)
current_key = 'a'
activated, already_ocur = False, False
amount = {chr(i) : 0 for i in range(ord('a'), ord('z') + 1) if chr(i) not in ['j', 'z']}
change_time = {chr(i) : 0 for i in range(ord('a'), ord('z') + 1) if chr(i) not in ['j', 'z']}
finished, guide = False, False

def calculate_distance(lis1, lis2):
    return math.sqrt((lis1.x - lis2.x)**2 + (lis1.y - lis2.y)**2 + (lis1.z - lis2.z)**2)

def calculate_length_vector(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def calculate_vector(lis1, lis2):
    new_vector = [lis1.x - lis2.x, lis1.y - lis2.y, lis1.z - lis2.z]
    length_vector = calculate_length_vector(new_vector)
    if length_vector == 0:
        return [0, 0, 0]
    return [new_vector[0]/length_vector, new_vector[1]/length_vector, new_vector[2]/length_vector]

def calculate_corner(vec1, vec2):
    len1 = calculate_length_vector(vec1)
    len2 = calculate_length_vector(vec2)
    if len1*len2 == 0:
        return 0
    cos = (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2])/(len1*len2)
    return cos

def process(lms, scale):
    vector, basee, tipp, corner_landmarks, = [], [], [], [calculate_corner(calculate_vector(lms[0], lms[1]), calculate_vector(lms[0], lms[17]))]
    for id in range(4, 21, 4):
        raw_vec = [lms[id].x - lms[0].x, lms[id].y - lms[0].y, lms[id].z - lms[0].z]
        vector.extend(calculate_vector(lms[id], lms[0]))
        vector.append(calculate_length_vector(raw_vec) / scale)

    for id in range(1, 14, 4):
        basee.append(calculate_distance(lms[id], lms[id + 4]) / scale)

    for id in range(4, 17, 4):
        tipp.append(calculate_distance(lms[id], lms[id + 4]) / scale)

    for id in range(0, len(Algo_corner) - 2):
        point1, point2, point3 = Algo_corner[id], Algo_corner[id + 1], Algo_corner[id + 2]
        if point3 in [0, 9, 13] or point2 in [0, 9, 13]:
            continue
        this_corner = calculate_corner(calculate_vector(lms[point1], lms[point2]), calculate_vector(lms[point2], lms[point3]))
        corner_landmarks.append(this_corner)

    if [len(vector), len(tipp), len(basee), len(corner_landmarks)] == [EXPECTED_VECTOR_AND_LENGTH, EXPECTED_THUMP - 1, EXPECTED_THUMP - 1, EXPECTED_CORNER]:
        test_df.loc[(len(test_df))] = [current_key] + vector + basee + tipp + corner_landmarks

def get_next_key(this_key):
    if this_key == 'y':
        return 'End'
    else:
        if this_key == 'i':
            return 'k'
        return chr(ord(this_key) + 1)

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if current_key == 'End':
        break

    if not activated and not already_ocur and change_time[current_key] < 6:
        print("press S to continue")
        already_ocur = True
    
    elif change_time[current_key] == 6 and not already_ocur:
        next_key = get_next_key(current_key)
        print(f"Collected for {current_key} enough 600 times, press S to move to {next_key} or R to replay")
        finished = True
        already_ocur = True

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            lms = landmarks.landmark
            scale = (calculate_distance(lms[0], lms[5])
                        + calculate_distance(lms[0], lms[9]) + calculate_distance(lms[0], lms[13])
                        + calculate_distance(lms[0], lms[17])) / 4
            if activated and change_time[current_key] < 6 and (amount[current_key] % 100 != 0 or amount[current_key] == 0):
                process(lms, scale)
                amount[current_key] += 1
            elif activated and amount[current_key] % 100 == 0 and amount[current_key] != 0:
                change_time[current_key] += 1
                print(f"Collected {amount[current_key]*change_time[current_key]} times for {current_key}")
                activated, already_ocur = False, False
                amount[current_key] = 0

    cv2.imshow("Video", frame)
    img = cv2.imread(f'ASL_SIGN/{current_key}_ASL.png')
    img = cv2.resize(img, (300, 300))
    cv2.imshow("Tutorial", img)
    key_input = cv2.waitKey(1)
    if key_input == ord('1'):
        break
    if key_input == ord('S') and not activated:
        if finished:
            current_key = next_key
        activated = True
        already_ocur = False
        finished = False
        if current_key != 'End':
            print(f"Recording for {current_key}")
    if key_input == ord('R'):
        print(f"Replay recording for {current_key}")
        amount[current_key] = 0
        change_time[current_key] = 0
        already_ocur = False
        finished = False
        test_df = test_df.iloc[:-600]

test_df.to_csv("DATA_ASL.csv", index=False)

cap.release()
cv2.destroyAllWindows()