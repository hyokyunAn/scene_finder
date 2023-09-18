from flask import Flask, render_template, redirect, request, url_for, session
import random
import os
import secrets
import time
import cv2
import json
from pytube import YouTube
import threading
from pytube import YouTube
import xml.etree.ElementTree as ET
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
import time
from PIL import Image
import requests
import re
import copy
from google.cloud import storage
import openai
import ast
from gpt_models import get_response



num_displaying = 3
openai.api_key = "YOUR_OPENAI_KEY"
gcs_bucket_name = "YOUR_GCS_BUCKET_NAME"
sentence_bert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

gpt_sentence_seperator = "ft:gpt-3.5-turbo-0613:korea-university::7y2aGnTp"
gpt_description_differentiator = "ft:gpt-3.5-turbo-0613:korea-university::7y3qUnRe" # ft:gpt-3.5-turbo-0613:korea-university::7y3ej4d7

sentence_seperator_prompt = "Separate the transcript using periods(.) to distinguish between sentences. Transcript: "
description_differentiator_prompt = "Please differentiate the descriptions for the image and dialogue to suit the clip and sentence bert models, respectively. You can choose to use one or both of the descriptions. Mark any unused description as None. Description: "



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
sentence_bert.to(device)
clip.to(device)

print(device)

def generate_secret_key(length=256):
    return secrets.token_hex(length)

app = Flask(__name__, static_folder='uploads')
app.secret_key = generate_secret_key()

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# uploads 폴더가 없으면 생성
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/') 
def main():
    return redirect('/scene_finder')

@app.route('/scene_finder')
def scene_finder_html():
    return render_template('./index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    # 언어 설정 가져오기
    # lang = request.form['lang']

    # 텍스트 데이터 가져오기
    text_data = request.form['textData']
    print("text_data is : ", request.form['textData'])
    # 동영상 파일 가져오기
    video_file = request.files['videoFile']
    youtube_url = request.form.get('youtubeURL')
    if youtube_url and video_file.filename:
        return "오류: 동영상 업로드와 유튜브 주소 입력 둘 중 하나만 선택해주세요."
    
    if youtube_url:  # 유튜브 주소가 입력된 경우
        try:
            yt = YouTube(youtube_url)
        except:
            return "Error: Invalid youtube url."
        print("orig yt.title : ", yt.title)
        yt.title = remove_special_characters(yt.title)
        print("cleaned yt.title : ", yt.title)
        video_stream = yt.streams.filter(file_extension='mp4').first()
        video_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'youtube_{yt.title}.mp4')
        video_stream.download(output_path=app.config['UPLOAD_FOLDER'], filename=f'youtube_{yt.title}.mp4')
        # video_stream.download(output_path=app.config['UPLOAD_FOLDER'], filename=f'youtube_100.mp4')   
        title = yt.title
    else:  # 동영상 파일이 직접 업로드된 경우
        random_prefix = str(random.randint(1, 100000))
        # video_filename = os.path.join(app.config['UPLOAD_FOLDER'], random_prefix + "_" + video_file.filename)
        video_filename = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_filename)
        title = video_file.filename
    print("title : ", title)
    web_accessible_imgs, json_filename, images, image_times = extract_images(video_filename, title)

    # response 데이터 키 : transcript, timestampes
    response = extract_texts(youtube_url=youtube_url, video_file_path=video_filename)

    import ast
    from gpt_models import get_response
    # 사용자의 요구 변환
    prompt = get_response(gpt_description_differentiator, description_differentiator_prompt, text_data)
    prompt = ast.literal_eval(prompt)
    print("변환된 prompt : ", prompt)
    
    # transcript를 문장 단위로 쪼개기 
    transcript = get_response(gpt_sentence_seperator, sentence_seperator_prompt, response['transcript'])
    seperated_transcript = transcript.split('.')

    # 검색 1) transcript에서 유사한 문장 찾기
    from t2t_search import get_similar_sentences
    if prompt['sentence_bert'] != None and prompt['sentence_bert'] != "None":
        sentence_ranks = get_similar_sentences(sentence_bert=sentence_bert, transcript=transcript, target_sentence=prompt['sentence_bert'])
        '''
        sentence_ranks :  [[5, " I think we're going to need another feather of a professor"], [3, ' Wingardium Leviosa'], [0, 'Wingardium Leviosa stop stop stop'], [1, ' To Canes, take some with you'], [6, ''], [2, " Saying it wrong, it's Leviosa, not Leviosa"], [4, ' Oh wow']]
        '''
        print("len(seperated_transcript) == len(sentence_ranks) ? ", len(seperated_transcript) == len(sentence_ranks), len(seperated_transcript), len(sentence_ranks))
        print("top1 sentence : ", sentence_ranks[0][1])
        print("top2 sentence : ", sentence_ranks[1][1])
        print("top3 sentence : ", sentence_ranks[2][1])
    else:
        print("prompt['sentence_bert'] None")
        sentence_ranks = None

    # 검색 2) 설명과 유사한 이미지 찾기
    from i2t_search import get_IT_embeds
    from t2t_search import get_most_similar_indices
    if prompt['clip'] != None and prompt['clip'] != "None":
        image_embeds, text_embeds = get_IT_embeds(clip, clip_processor, images, prompt['clip'])
        image_ranks = get_most_similar_indices(target_embedding=text_embeds, embeddings=image_embeds)
        print("image_ranks : ", image_ranks)
    else:
        image_ranks = None

    import copy
    # 검색 결과 종합
    if image_ranks != None and sentence_ranks != None:
        from t2t_search import get_summed_rank
        summed_ranks = get_summed_rank(copy.deepcopy(image_ranks), 
                                       image_times, 
                                       copy.deepcopy(web_accessible_imgs), 
                                       copy.deepcopy(sentence_ranks), 
                                       seperated_transcript, response, num_displaying)
    else:
        if image_ranks != None:
            summed_ranks = image_ranks
        elif sentence_ranks != None:
            summed_ranks = sentence_ranks

    # 최종 이미지, 시간대 반환
    # f_images, times = find_scenes(session['saved_images'], text_data, session['json_filename'])
    f_images, times = find_scenes(copy.deepcopy(web_accessible_imgs), image_times, summed_ranks)
    

    image_embeds = image_embeds.tolist()
    blob_name_image_embeds = f"{video_filename}_image_embeds"

    session['saved_images'] = web_accessible_imgs
    session['json_filename'] = json_filename
    session['image_times'] = image_times
    session['transcript'] = transcript
    session['blob_name_image_embeds'] = blob_name_image_embeds

    

    # 리스트 저장
    from speech2text import save_list_to_gcs
    save_list_to_gcs(gcs_bucket_name, blob_name_image_embeds, image_embeds)



    # 일정 시간 후 파일 삭제 스케줄링
    threading.Thread(target=remove_after_delay, args=(video_filename, 3600)).start()
    for image_path in f_images:
        threading.Thread(target=remove_after_delay, args=(image_path, 3600)).start()

    return render_template('result.html', image_text_pairs=zip(f_images, times))

@app.route('/reprocess_data', methods=['POST'])
def reprocess_data():
    text_data = request.form['reTextData']

    web_accessible_imgs = session.get('saved_images')
    json_filename = session.get('json_filename')
    image_times = session.get('image_times')
    # image_embeds = session.get('image_embeds')
    transcript = session.get('transcript')
    blob_name_image_embeds = session.get('blob_name_image_embeds')

    from speech2text import load_list_from_gcs
    import numpy as np
    image_embeds = load_list_from_gcs(gcs_bucket_name, blob_name_image_embeds)
    image_embeds = np.array(image_embeds)

    print("re text_data : ", text_data)
    print("web_accessible_imgs : ", web_accessible_imgs)
    print("json_filename : ", json_filename)
    print("image_times : ", image_times)

    if not web_accessible_imgs:
        return "Error: Image information not found. Please reload the initial page."

    ######################3

    # 사용자의 요구 변환
    prompt = get_response(gpt_description_differentiator, description_differentiator_prompt, text_data)
    prompt = ast.literal_eval(prompt)
    print("변환된 prompt : ", prompt)
    
    # transcript를 문장 단위로 쪼개기 
    ### transcript = get_response(gpt_sentence_seperator, sentence_seperator_prompt, response['transcript'])
    seperated_transcript = transcript.split('.')

    # 검색 1) transcript에서 유사한 문장 찾기
    from t2t_search import get_similar_sentences
    if prompt['sentence_bert'] != None and prompt['sentence_bert'] != "None":
        sentence_ranks = get_similar_sentences(sentence_bert=sentence_bert, transcript=transcript, target_sentence=prompt['sentence_bert'])
        '''
        sentence_ranks :  [[5, " I think we're going to need another feather of a professor"], [3, ' Wingardium Leviosa'], [0, 'Wingardium Leviosa stop stop stop'], [1, ' To Canes, take some with you'], [6, ''], [2, " Saying it wrong, it's Leviosa, not Leviosa"], [4, ' Oh wow']]
        '''
        print("len(seperated_transcript) == len(sentence_ranks) ? ", len(seperated_transcript) == len(sentence_ranks), len(seperated_transcript), len(sentence_ranks))
        print("top1 sentence : ", sentence_ranks[0][1])
        print("top2 sentence : ", sentence_ranks[1][1])
        print("top3 sentence : ", sentence_ranks[2][1])
    else:
        print("prompt['sentence_bert'] None")
        sentence_ranks = None

    # 검색 2) 설명과 유사한 이미지 찾기
    ### from i2t_search import get_IT_embeds
    from t2t_search import get_most_similar_indices
    if prompt['clip'] != None and prompt['clip'] != "None":
        ### image_embeds, text_embeds = get_IT_embeds(clip, clip_processor, images, prompt['clip'])
        from i2t_search import get_text_embeds
        text_embeds = get_text_embeds(clip, clip_processor, text_data)
        image_ranks = get_most_similar_indices(target_embedding=text_embeds, embeddings=image_embeds)
        print("image_ranks : ", image_ranks)
    else:
        image_ranks = None

    # 검색 결과 종합
    if image_ranks != None and sentence_ranks != None:
        from t2t_search import get_summed_rank
        import copy
        summed_ranks = get_summed_rank(copy.deepcopy(image_ranks), 
                                       image_times, 
                                       copy.deepcopy(web_accessible_imgs), 
                                       copy.deepcopy(sentence_ranks), 
                                       seperated_transcript, response, num_displaying)
    else:
        if image_ranks != None:
            summed_ranks = image_ranks
        elif sentence_ranks != None:
            summed_ranks = sentence_ranks

    f_images, times = find_scenes(web_accessible_imgs, image_times, summed_ranks)


    return render_template('result.html', image_text_pairs=zip(f_images, times))

def remove_after_delay(file_path, delay):
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)

def frame_to_time(frame_num, fps, frame_interval):
    total_seconds = int(frame_num / fps / frame_interval)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return total_seconds
    # return f"{minutes:02}:{seconds:02}"

def extract_images(video_filename, title):
    frame_interval = 1
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, prev_frame = cap.read()
    frame_num = 0
    count = 0
    time_mapping = {}
    saved_images = []

    # image_folder = os.path.join(UPLOAD_FOLDER, os.path.splitext(os.path.basename(video_filename))[0])
    image_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    print("video_filename : ", video_filename)
    print("os.path.basename(video_filename) : ", os.path.basename(video_filename))
    images = []
    while True:
        ret, frame = cap.read()
        frame_num += frame_interval
        if not ret:
            break

        diff = cv2.absdiff(prev_frame, frame)
        mean_diff = diff.mean()
        
        if mean_diff > 20:  # 조건을 완화
            count += 1
            
            # filename = os.path.join(image_folder, f"cut_{count}.png")
            filename = os.path.join(image_folder, f"{title}_{count}.png")
            ### print("filename : ", filename)
            cv2.imwrite(filename, frame)
            saved_images.append(filename)
            images.append(frame)

            
            filename_ = copy.deepcopy(filename)
            filename_ = filename_.replace('\\', '/')
            
            # print('filename_ : ', filename_)
            time_mapping[filename_] = frame_to_time(frame_num, fps, frame_interval)

        prev_frame = frame
    cap.release()

    json_filename = os.path.join(image_folder, "timestamps.json")
    with open(json_filename, 'w') as f:
        json.dump(time_mapping, f, indent=4)

    web_accessible_imgs = [url_for('static', filename=os.path.relpath(img, UPLOAD_FOLDER).replace('\\', '/')) for img in saved_images]
    return web_accessible_imgs, json_filename, images, time_mapping


def extract_youtube_captions(youtube_url, language_code='en'):
    yt = YouTube(youtube_url)
    caption = yt.captions.get_by_language_code(language_code)
    
    # XML 형식의 자막 데이터 가져오기
    xml_captions = caption.xml_captions
    
    # XML 파싱
    root = ET.fromstring(xml_captions)
    
    # 자막 데이터 추출
    captions_data = []
    for child in root:
        start_time = child.attrib['start']  # 시작 시간
        duration = child.attrib['dur']  # 지속 시간
        text = child.text  # 자막 텍스트
        
        captions_data.append({
            "start_time": start_time,
            "duration": duration,
            "text": text
        })
    
    return captions_data


def extract_texts(youtube_url=None, video_file_path=None, pytube_captions=False, gcs_bucket_url="gs://scene_finder-video_bucket/"):
    if pytube_captions:
        captions_data = extract_youtube_captions(youtube_url)
        for data in captions_data:
            print(f"시작 시간: {data['start_time']}, 지속 시간: {data['duration']}, 텍스트: {data['text']}")
    else:
        import pickle
        from speech2text import extract_audio_from_video, transcribe_gcs
        audio_output_path = copy.deepcopy(video_file_path)
        audio_output_path = audio_output_path.replace(".mp4", ".mp3")

        extract_audio_from_video(video_file_path, 
                                audio_output_path,
                                bucket_name=gcs_bucket_name)
        response = transcribe_gcs(gcs_bucket_url + audio_output_path)

        # with open("./response", "wb") as f:
        #     pickle.dump(response, f)
        return  response

def find_scenes(image_names, timestamps, ranks):
    # print("find_scenes image_names : ", image_names)
    selected_images = [image_names[i] for i in ranks[:num_displaying]]
    try:
        times = [timestamps[img] for img in selected_images[:num_displaying]]
    except:
        times = [timestamps[img[1:]] for img in selected_images[:num_displaying]]
    
    min_and_seconds = []
    for time in times:
        minutes = time // 60
        seconds = time % 60
        min_and_seconds.append(f"{minutes}:{seconds:02}")
    return selected_images, min_and_seconds


def remove_special_characters(title):
    # 특수 문자 제거
    cleaned_title = re.sub(r'[^a-zA-Z0-9\s\-]', '', title)
    # 공백을 _로 치환
    return cleaned_title.replace(' ', '_')




if __name__ == "__main__":
    app.run(port=5001, debug=True)
