from flask import Flask, render_template, redirect, request
import random
import transformers
from transformers import CLIPModel, CLIPProcessor


app = Flask(__name__)
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.route('/')
def index():
    return 'hello'

@app.route('/tokenize/<sentence>')
def tokenize(sentence):
    ids = processor(sentence)['input_ids']
    
    return {sentence : ids} 


topics = [
    {'id': 1, 'title': 'html', 'body': 'html is ...'},
    {'id': 2, 'title': 'css', 'body': 'css is ...'},
    {'id': 3, 'title': 'javascript', 'body': 'javascript is ...'}
]


@app.route('/sample')
def index_sample():
    liTags = ''
    for topic in topics:
        liTags = liTags + f'<li><a href="/sample/{topic["id"]}/">{topic["title"]}</a></li>'
    return

@app.route('/scene_finder')
def scene_finder_html():
    return render_template('./index.html')


@app.route('/process_data', methods=['POST'])
def process_data():
    # 텍스트 데이터 가져오기
    text_data = request.form['textData']
    
    # 동영상 파일 가져오기
    video_file = request.files['videoFile']
    if video_file:
        video_file.save(f'uploads/{video_file.filename}')
    
    # 이후 원하는 파이썬 함수를 텍스트나 동영상에 적용할 수 있습니다.
    # 예: processed_data = some_function(text_data)
    print("입력된 텍스트 : ", text_data)

    return "데이터 처리 완료!" # 이 부분은 처리 결과나 다른 페이지로 리다이렉트 하는 등의 로직으로 변경될 수 있습니다.

if __name__ == "__main__":
    app.run(port=5001, debug=True)