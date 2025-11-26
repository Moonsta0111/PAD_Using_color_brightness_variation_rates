import cv2
import mediapipe as mp
import numpy as np
from collections import defaultdict

# -----------------------------
# 1) MediaPipe FaceMesh 초기화
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)

# -----------------------------
# 2) 카메라 캡처 객체 생성
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# 3) 섹션별 랜드마크 인덱스 정의
# -----------------------------
sections = {
    'left_eye': [
        113, 225, 224, 223, 222, 221, 189, 190, 226, 247, 130, 33, 161, 160, 159, 158, 157,
        173, 30, 29, 27, 28, 56, 7, 25, 31, 228, 110, 163, 229, 24, 144, 230, 23, 145, 153, 231,
        22, 232, 26, 154, 233, 112, 155, 244, 243, 133
    ],
    'right_eye': [
        413, 414, 441, 286, 442, 258, 443, 257, 444, 259, 445, 260, 342, 467, 446, 359, 261,
        255, 448, 339, 449, 254, 450, 253, 451, 452, 256, 453, 341, 464, 463, 362, 382, 381,
        380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ],
    'nose': [
        168, 188, 122, 6, 351, 412, 217, 174, 196, 197, 419, 399, 437, 209, 198, 236, 3, 195,
        248, 456, 420, 429, 129, 49, 131, 134, 51, 5, 281, 363, 360, 279, 102, 48, 115, 220,
        45, 4, 275, 440, 344, 358, 278, 64, 219, 235, 298, 240, 59, 166, 218, 75, 79, 60, 239,
        237, 99, 97, 238, 20, 44, 241, 242, 141, 125, 2, 94, 19, 1, 274, 354, 370, 462, 461,
        458, 250, 459, 457, 326, 328, 290, 309, 438, 305, 289, 392, 460, 455, 439, 327, 294
    ],
    'mouth': [
        57, 43, 182, 83, 313, 406, 335, 273, 287, 410, 393, 164, 167, 165, 92, 186, 14, 15, 16,
        17, 18, 317, 316, 315, 402, 403, 404, 405, 318, 319, 320, 321, 324, 325, 307, 308,
        299, 391, 407, 408, 409, 415, 322, 310, 76, 272, 270, 314, 271, 303, 269, 312, 268,
        302, 267, 0, 11, 12, 13, 82, 38, 72, 37, 39, 73, 41, 81, 40, 74, 42, 185, 184, 183, 191,
        61, 62, 78, 146, 77, 96, 106, 80, 88, 89, 90, 91, 181, 180, 179, 178, 84, 85, 86, 87
    ],
    'left_cheek': [
        266, 357, 350, 349, 348, 347, 346, 340, 372, 264, 343, 277, 329, 330, 280, 345,
        447, 352, 366, 401, 435, 376, 433, 367, 411, 416, 364, 427, 434, 394, 430, 422,
        432, 436, 426, 423, 425, 371
    ],
    'right_cheek': [
        128, 34, 227, 137, 177, 215, 138, 135, 169, 143, 116, 123, 147, 213, 192, 214, 210,
        202, 212, 187, 111, 117, 50, 207, 216, 206, 205, 118, 119, 101, 36, 203, 142, 100,
        120, 47, 121, 108, 114
    ]
}

# 각 섹션별 임계값 정의 (색상/명도)
section_thresholds = {
    'left_eye': {'color_diff': 12.25, 'brightness_diff': 6.5},
    'right_eye': {'color_diff': 12.5, 'brightness_diff': 6.6},
    'nose': {'color_diff': 11, 'brightness_diff': 5.8},
    'mouth': {'color_diff': 9, 'brightness_diff': 4.6},
    'left_cheek': {'color_diff': 6.85, 'brightness_diff': 3.35},
    'right_cheek': {'color_diff': 8.8, 'brightness_diff': 4.45}
}

# -------------------------------------------------
# 4) 명도/색상 변화율 계산에 필요한 변수 설정
# -------------------------------------------------
prev_colors = None

# ★ 변경 1) 프레임 단위 버퍼(문자열 저장 X),
#    섹션별 "색상 변동 비율", "밝기 변동 비율"을 쌓기 위한 구조
color_history = defaultdict(list)     # 예: color_history["left_eye"] = [0.3, 0.4, 0.2, ...]
brightness_history = defaultdict(list)

frame_count = 0                       # 현재까지 누적된 프레임 수 (0~100)
current_final_result = ""            # 100프레임 단위 결과 표시

# ---------------------------
# 5) 메인 루프 (while True)
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, c = frame.shape

    current_colors = []

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        for lm in face_landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            if 0 <= x < w and 0 <= y < h:
                current_colors.append(frame[y, x])
            else:
                current_colors.append(None)

        if prev_colors is not None and len(prev_colors) == len(current_colors):
            # 섹션별 color_changed, brightness_changed를 구해서
            # ratio(= changed / total_count)를 따로 기록한다.
            for section_name, indices in sections.items():
                thresholds = section_thresholds[section_name]
                color_diff_threshold = thresholds['color_diff']
                brightness_diff_threshold = thresholds['brightness_diff']

                color_changed = 0
                brightness_changed = 0
                total_pointer_count = 0

                for idx in indices:
                    if (idx < len(current_colors)
                            and current_colors[idx] is not None
                            and prev_colors[idx] is not None):
                        total_pointer_count += 1
                        curr_color = np.array(current_colors[idx], dtype=np.float32)
                        prev_color = np.array(prev_colors[idx], dtype=np.float32)

                        color_diff = np.linalg.norm(curr_color - prev_color)
                        curr_brightness = np.mean(curr_color)
                        prev_brightness = np.mean(prev_color)
                        brightness_diff = abs(curr_brightness - prev_brightness)

                        if color_diff >= color_diff_threshold:
                            color_changed += 1
                        if brightness_diff >= brightness_diff_threshold:
                            brightness_changed += 1

                # ratio 계산 (섹션별 현재 프레임)
                if total_pointer_count > 0:
                    color_ratio = color_changed / float(total_pointer_count)
                    brightness_ratio = brightness_changed / float(total_pointer_count)
                else:
                    # 랜드마크가 유효하지 않다면 0으로 처리
                    color_ratio = 0.0
                    brightness_ratio = 0.0

                # 이 프레임의 ratio를 저장
                color_history[section_name].append(color_ratio)
                brightness_history[section_name].append(brightness_ratio)

            # 한 프레임 처리 후 frame_count 증가
            frame_count += 1

            # ★ 변경 2) 100프레임이 쌓이면 섹션별 평균을 계산하여 최종 결과 판단
            if frame_count == 100:
                person_section_count = 0

                for section_name in sections.keys():
                    # 평균 계산
                    avg_color_ratio = sum(color_history[section_name]) / 100.0
                    avg_brightness_ratio = sum(brightness_history[section_name]) / 100.0

                    # 임계값 비교
                    thresholds = section_thresholds[section_name]
                    color_diff_threshold = thresholds['color_diff']
                    brightness_diff_threshold = thresholds['brightness_diff']

                    # "평균 변화율" vs "기존 threshold" 직접 비교
                    # 원하는 로직에 맞게 조건 구성
                    # 여기서는 "평균 color_ratio >= 0.5" 같은 방식이 아니라
                    # "평균 color_diff vs threshold" 직접 비교가 필요할 수도 있음.
                    # 다만, 문제에서 "임계값이랑 비교"라고 하셨으므로 일단 아래처럼 예시:
                    #   avg_color_ratio >= 0.5 -> "Person" 으로 볼 수도 있고,
                    #   혹은 threshold만큼 바뀐 도트가 많으면 된다라고 해석하면
                    #   threshold를 0.5 등으로 설정해야 함.

                    # (예시) 색상·밝기 변동 평균 비율이 둘 다 0.5 이상이면 Person
                    #        (실제로는 원하는 임계값 세팅이 필요)
                    # -------------------------------------------------
                    # 만약 “색상 diff 임계값”과 “평균 ratio”를 직접 매칭하려면
                    # 임계값(1.5, 2.0)은 절대 scale이고 ratio는 [0~1]이므로
                    # 별도의 로직이 필요할 수 있음.
                    # -------------------------------------------------
                    # 여기서는 간단히 "0.5" 고정 예시
                    if avg_color_ratio >= 0.16988 or avg_brightness_ratio >= 0.18:
                        person_section_count += 1

                # 섹션 중 3개 이상이 Person이면 최종 'Person'
                if person_section_count >= 3:
                    current_final_result = "Person"
                else:
                    current_final_result = "Picture"

                # 100프레임 분량 처리 후 모든 리스트 비우기
                for sec in sections.keys():
                    color_history[sec].clear()
                    brightness_history[sec].clear()

                frame_count = 0  # 다시 0으로 초기화

        prev_colors = current_colors

    else:
        cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 최종 결과 표시 (100프레임 단위로만 업데이트)
    if current_final_result:
        cv2.putText(frame, f"Final Result: {current_final_result}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 255), 3)

    cv2.imshow("Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()