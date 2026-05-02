import cv2
import mediapipe as mp
import numpy as np
import os
import time


def get_second():
    return time.time()

def to_np(*args):
    return np.concatenate([np.atleast_1d(a).flatten() for a in args])

def safe_norm(v):
    return np.linalg.norm(v) + 1e-8

def calculate_vector(a, b):
    return np.array(b) - np.array(a)

def distance(a, b, scale=1):
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return np.linalg.norm(calculate_vector(a, b)) / (scale + 1e-8)

def angle(v1, v2):
    n1, n2 = safe_norm(v1), safe_norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos, -1, 1))

def safe_scale(hand):
    if np.all(hand == 0):
        return 1.0
    s = np.mean([distance(hand[i], hand[0], 1) for i in tip])
    return max(s, 1e-6)

def palm_orientation(hand, scale):
    if np.all(hand == 0):
        return np.zeros(3)

    wrist, index, pinky = hand[0], hand[5], hand[17]
    v1 = (index - wrist) / scale
    v2 = (pinky - wrist) / scale

    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return np.zeros(3)

    return n / norm

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

Width, Height = 900, 700

importance_pose = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST
]

importance_face_idx = [70, 105, 334, 300, 33, 133, 362, 263, 61, 291, 13, 14, 1]
tip = [4, 8, 12, 16, 20]
Angle_corner = [0, 1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

frame_count, activated, in_cool_down, prev_second, current_second = 0, False, False, 4, None
already_ocur = False

# Added state
waiting_confirm = False
replay_mode = False

DATA_PATH = os.path.join('Dataset')
alphabet = [chr(ch) for ch in range(ord('a'), ord('z') + 1)]
dynamic_actions = ['Hello', 'Goodbye', 'Thank you', 'Sorry', 'Please', 'How', 'Where', 'What',
                   'Who', 'You', 'I', 'Friend', 'Like', "Don't like", 'Good', 'Fine', 'Bad',
                   'Help', 'Need', 'Name', 'Work', 'Deaf', 'Hearing', 'Again', 'Mother', 'Father',
                   'Yes', 'No', 'Understand', 'Misunderstand']

actions = np.array(alphabet + dynamic_actions)
sequence_numbers, frame_in_sequence = 40, 30
current_action_index, current_frame, current_clip = 0, 0, 1

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if not activated and not already_ocur and not waiting_confirm:
        print(f"press s to collect for {actions[current_action_index]}")
        already_ocur = True

    if waiting_confirm and not already_ocur:
        print("Press S = accept | Press R = replay")
        already_ocur = True

    if activated:
        current_frame += 1

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (Width, Height))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    left = np.zeros((21, 3))
    right = np.zeros((21, 3))
    face_coor = np.zeros((13, 3))
    pose_coor = np.zeros((6, 3))

    has_left = has_right = False
    has_face = has_pose = False

    cv2.putText(frame, f"Collect for {actions[current_action_index]}", (50, 600), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 0, 255), 4, cv2.LINE_AA)

    res_hand = hands.process(rgb)
    res_face = face.process(rgb)
    res_pose = pose.process(rgb)

    if res_hand.multi_hand_landmarks:
        for lm, handness in zip(res_hand.multi_hand_landmarks, res_hand.multi_handedness):
            label = handness.classification[0].label
            coord = np.array([[p.x, p.y, p.z] for p in lm.landmark])

            if label == "Left":
                left = coord
                has_left = True
                color = (255, 0, 0)
            else:
                right = coord
                has_right = True
                color = (0, 0, 255)

            for p in lm.landmark:
                cv2.circle(frame, (int(p.x * Width), int(p.y * Height)), 3, color, -1)

    if res_face.multi_face_landmarks:
        has_face = True
        lm = res_face.multi_face_landmarks[0].landmark
        for i, idx in enumerate(importance_face_idx):
            p = lm[idx]
            face_coor[i] = [p.x, p.y, p.z]
            cv2.circle(frame, (int(p.x * Width), int(p.y * Height)), 3, (255, 255, 0), -1)

    if res_pose.pose_landmarks:
        has_pose = True
        lm = res_pose.pose_landmarks.landmark
        for i, idx in enumerate(importance_pose):
            p = lm[idx.value]
            pose_coor[i] = [p.x, p.y, p.z]
            cv2.circle(frame, (int(p.x * Width), int(p.y * Height)), 3, (0, 255, 255), -1)

    scale_left = safe_scale(left)
    scale_right = safe_scale(right)

    vec_left = to_np(*((left - left[0]) / scale_left))
    vec_right = to_np(*((right - right[0]) / scale_right))

    dis_tip_left = to_np([distance(left[i], left[0], scale_left) for i in tip])
    dis_tip_right = to_np([distance(right[i], right[0], scale_right) for i in tip])

    distance_tip_left = to_np([distance(left[tip[i]], left[tip[i - 1]], scale_left) for i in range(1, 4)])
    distance_tip_right = to_np([distance(right[tip[i]], right[tip[i - 1]], scale_right) for i in range(1, 4)])

    angle_left, angle_right = [], []
    for i in range(1, len(Angle_corner) - 1):
        a, b, c = Angle_corner[i - 1], Angle_corner[i], Angle_corner[i + 1]
        if b == 0 or b in tip:
            continue

        angle_left.append(angle(calculate_vector(left[a], left[b]), calculate_vector(left[b], left[c])))
        angle_right.append(angle(calculate_vector(right[a], right[b]), calculate_vector(right[b], right[c])))

    body_scale = max(distance(pose_coor[0], pose_coor[1]), 1)

    angle_left = to_np(angle_left)
    angle_right = to_np(angle_right)

    left_ori = palm_orientation(left, scale_left)
    right_ori = palm_orientation(right, scale_right)

    center_left = (2 * left[0] + left[1] + left[17]) / 2
    center_right = (2 * right[0] + right[1] + right[17]) / 2

    chest = (pose_coor[0] + pose_coor[1]) / 2 if has_pose else np.zeros(3)
    center_face = face_coor[3] if has_face else np.zeros(3)

    dis_face_left = distance(center_face, center_left) / body_scale
    dis_face_right = distance(center_face, center_right) / body_scale
    dis_chest_left = distance(chest, center_left) / body_scale
    dis_chest_right = distance(chest, center_right) / body_scale
    dis_hands = distance(center_left, center_right) / body_scale

    features = to_np(
        pose_coor, face_coor,
        vec_left, vec_right,
        dis_tip_left, dis_tip_right,
        distance_tip_left, distance_tip_right,
        angle_left, angle_right,
        left_ori, right_ori,
        dis_face_left, dis_face_right,
        dis_chest_left, dis_chest_right,
        dis_hands
    )

    features = np.nan_to_num(features)

    if activated:
        try:
            os.makedirs(os.path.join(DATA_PATH, actions[current_action_index], str(current_clip)))
        except:
            pass

        npy_path = os.path.join(DATA_PATH, actions[current_action_index], str(current_clip), str(current_frame))
        np.save(npy_path, features)

    if current_frame == frame_in_sequence:
        current_frame = 0
        activated = False
        waiting_confirm = True
        already_ocur = False

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q') or current_action_index >= len(actions):
        break

    elif key == ord('s') and not in_cool_down:
        if waiting_confirm:
            waiting_confirm = False
            replay_mode = False
            current_clip += 1
            already_ocur = False

            if current_clip > sequence_numbers:
                current_action_index += 1
                current_clip = 1
                waiting_confirm = False
                replay_mode = False
                already_ocur = False

        else:
            in_cool_down = True
            current_second = get_second()

    elif key == ord('r') and waiting_confirm and not in_cool_down:
        replay_mode = True
        waiting_confirm = False
        already_ocur = False
        current_frame = 0

        in_cool_down = True
        current_second = get_second()

        print(f"Replay clip {current_clip}")

    if in_cool_down:
        count = int(get_second() - current_second)

        if count >= 3:
            activated = True
            in_cool_down = False
            print(f"Collecting 30 frame for {actions[current_action_index]}")
            prev_second = 4
            continue

        remain = 3 - count

        if remain != prev_second:
            print(
                f"start collect for {actions[current_action_index]} in {remain} second left to clip {current_clip}"
            )
            prev_second = remain
            already_ocur = True

cap.release()
cv2.destroyAllWindows()
