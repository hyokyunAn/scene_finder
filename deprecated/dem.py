import cv2
import os
import json


def extract_images(video_filename):
    frame_interval = 150
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, prev_frame = cap.read()
    frame_num = 0
    count = 0
    time_mapping = {}
    saved_images = []

    while True:
        ret, frame = cap.read()
        frame_num += frame_interval
        if not ret:
            break

        # 연속된 두 프레임 간의 차이 계산
        diff = cv2.absdiff(prev_frame, frame)
        mean_diff = diff.mean()
        # print("mean_diff : ", mean_diff)
        # print("frame_num : ", frame_num)

        # 차이가 특정 임계값(예: 30)을 초과하면 해당 프레임 저장
        if mean_diff > 30:
            count += 1
            filename = os.path.join("C:\\Users\\hyoky\\AI_Projects\\scene_finder\\uploads\\shots\\", f"cut_{count}.png")
            cv2.imwrite(filename, frame)
            saved_images.append(filename)

            # 해당 이미지의 시간 정보 저장
            time_mapping[filename] = frame_to_time(frame_num, fps, frame_interval)

        prev_frame = frame

    cap.release()

    # 동영상 파일 이름을 기반으로 timestamps.json 파일 이름 지정
    # json_filename = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_filename) + "_timestamps.json")
    json_filename = "C:\\Users\\hyoky\\AI_Projects\\scene_finder\\uploads\\shots\\" + "archi" + "_timestamps.json"
    with open(json_filename, 'w') as f:
        json.dump(time_mapping, f, indent=4)

    return saved_images, json_filename


def frame_to_time(frame_num, fps, frame_interval):
    """프레임 번호를 분:초 형태의 문자열로 변환"""
    total_seconds = int(frame_num / fps / frame_interval)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"





extract_images("C:\\Users\\hyoky\\AI_Projects\\scene_finder\\uploads\\archi.mp4")