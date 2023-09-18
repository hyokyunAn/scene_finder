from flask import Flask, render_template, redirect, request, url_for, session
import random
import os
import transformers
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests
import threading
import secrets
from pytube import YouTube
import time
import cv2
import json
 



app = Flask(__name__, static_folder='uploads')
app.secret_key = secrets.token_hex(16) 
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# uploads 폴더가 없으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.route('/')
def main():
    return redirect('/scene_finder')

@app.route('/scene_finder')
def scene_finder_html():
    return render_template('./index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    # 텍스트 데이터 가져오기
    text_data = request.form['textData']
    
    # 동영상 파일 가져오기
    video_file = request.files['videoFile']
    youtube_url = request.form.get('youtubeURL')
    if youtube_url and video_file.filename:
        return "오류: 동영상 업로드와 유튜브 주소 입력 둘 중 하나만 선택해주세요."
    
    if youtube_url:  # 유튜브 주소가 입력된 경우
        print("유튜브 주소 : ", youtube_url)
        yt = YouTube(youtube_url)
        video_stream = yt.streams.filter(file_extension='mp4').first()
        video_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'youtube_{yt.title}.mp4')
        video_stream.download(output_path=app.config['UPLOAD_FOLDER'], filename=f'youtube_{yt.title}.mp4')
        # video_stream.download(output_path=app.config['UPLOAD_FOLDER'], filename=f'youtube_{yt.title}')
    else:  # 동영상 파일이 직접 업로드된 경우
        print("동영상 파일 이름 : ", request.files['videoFile'].filename)
        random_prefix = str(random.randint(1, 100000))
        video_filename = os.path.join(app.config['UPLOAD_FOLDER'], random_prefix + "_" + video_file.filename)
        video_file.save(video_filename)

    
    print("장면 설명 텍스트 : ", text_data)

    

    ### web_accessible_imgs = extract_images(video_filename)
    web_accessible_imgs, json_filename =  extract_images(video_filename)
    session['saved_images'] = web_accessible_imgs
    f_images, times = find_scenes(web_accessible_imgs, text_data, json_filename)
    
   
    image_text_pairs = zip(f_images, times)

    # 일정 시간 후 파일 삭제 스케줄링
    threading.Thread(target=remove_after_delay, args=(video_filename, 3600)).start()  # 1시간 후 삭제
    for image_path in f_images:
        threading.Thread(target=remove_after_delay, args=(image_path, 3600)).start()  # 1시간 후 삭제


    return render_template('result.html', image_text_pairs=image_text_pairs)


@app.route('/reprocess_data', methods=['POST'])
def reprocess_data():
    # 이전에 저장한 동영상 파일 경로
    video_filename = session.get('video_filename')
    
    # 새로운 텍스트 데이터 가져오기
    re_text_data = request.form['reTextData']

    # find_scenes 함수 호출
    web_accessible_imgs = session.get('saved_images')  # session에서 이미지 파일 경로들 가져오기

    if not web_accessible_imgs:
        return "오류: 이미지 정보를 찾을 수 없습니다. 먼저 process_data를 실행해주세요."
    f_images, times = find_scenes(web_accessible_imgs, re_text_data)
    web_accessible_imgs = [url_for('static', filename=os.path.basename(img)) for img in f_images]
    
    # session['video_filename'] = video_filename
    print("re_text_data : ", re_text_data)

    return render_template('result.html', image_text_pairs=zip(web_accessible_imgs, times))



def remove_after_delay(file_path, delay):
    """delay (초 단위) 후에 파일 삭제"""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)


def frame_to_time(frame_num, fps, frame_interval):
    """프레임 번호를 분:초 형태의 문자열로 변환"""
    total_seconds = int(frame_num / fps / frame_interval)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"



def extract_images(video_filename):
    # video_filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    # video_file.save(video_filename)
    
    frame_interval = 150
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, prev_frame = cap.read()
    frame_num = 0
    count = 0
    time_mapping = {}
    saved_images = []

    image_folder = video_filename.replace(".mp4", "")
    if os.path.exists(image_folder):
        pass
    else:
        os.makedirs(image_folder)

    while True:
        ret, frame = cap.read()
        frame_num += frame_interval
        if not ret:
            break

        diff = cv2.absdiff(prev_frame, frame)
        mean_diff = diff.mean()
        if mean_diff > 30:
            count += 1
            filename = os.path.join(image_folder, f"cut_{count}.png")
            cv2.imwrite(filename, frame)
            saved_images.append(filename)

            time_mapping[filename] = frame_to_time(frame_num, fps, frame_interval)

        prev_frame = frame

    cap.release()
    print("saved_images : ", saved_images)

    # 동영상 파일 이름을 기반으로 timestamps.json 파일 이름 지정
    # json_filename = os.path.join(image_folder, video_filename.replace(".mp4", "") + "_timestamps.json")
    print("image_folder ： ", image_folder)
    json_filename = os.path.join(image_folder, "timestamps.json")
    with open(json_filename, 'w') as f:
        json.dump(time_mapping, f, indent=4)

    web_accessible_imgs = [url_for('static', filename=os.path.basename(img)) for img in saved_images]
    print("web_accessible_imgs ： ", web_accessible_imgs)
    
    return web_accessible_imgs, json_filename


def save_cuts_from_video(video_filename):
    # 동영상 이름으로 디렉터리 생성
    video_name = os.path.basename(video_filename).split('.')[0]  # 확장자 제외
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, prev_frame = cap.read()
    frame_num = 0
    count = 0
    time_mapping = {}
    saved_images = []

    while True:
        ret, frame = cap.read()
        frame_num += 1
        if not ret:
            break

        diff = cv2.absdiff(prev_frame, frame)
        mean_diff = diff.mean()

        if mean_diff > 30:
            count += 1
            filename = os.path.join(dir_path, f"cut_{count}.png")
            cv2.imwrite(filename, frame)
            saved_images.append(filename)

            time_mapping[filename] = frame_to_time(frame_num, fps)

        prev_frame = frame

    cap.release()

    json_filename = os.path.join(dir_path, video_name + "_timestamps.json")
    with open(json_filename, 'w') as f:
        json.dump(time_mapping, f, indent=4)

    web_accessible_imgs = [url_for('static', filename=os.path.basename(img)) for img in saved_images]

    return web_accessible_imgs, json_filename
    # return saved_images, json_filename


def get_timestamps(video_filename):
    """동영상 파일 이름을 기반으로 해당 동영상의 timestamps.json 파일 내용을 반환합니다."""
    json_filename = video_filename + "_timestamps.json"
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            timestamps = json.load(f)
        return timestamps
    else:
        return None



def find_scenes(images, text, json_filename):
    # images_dir에서 모든 이미지 파일을 로드
    # images = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith(('.png', '.jpg'))]
    
    times = ["2", "3", "4:4"]
    print("images : ", images)
    # ... (나머지 find_scenes 함수 내용)
    return images[:3], times  # 예를 들어 processed_images는 처리된 이미지들의 리스트입니다.






if __name__ == "__main__":
    app.run(port=5001, debug=True)

