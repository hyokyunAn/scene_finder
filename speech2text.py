from google.cloud import speech
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from moviepy.editor import *
import pickle


def speech_to_text(
    config: speech.RecognitionConfig,
    audio: speech.RecognitionAudio,
) -> speech.RecognizeResponse:
    client = speech.SpeechClient()

    # Synchronous speech recognition request
    response = client.recognize(config=config, audio=audio)

    return response


def print_response(response: speech.RecognizeResponse):
    for result in response.results:
        print_result(result)


def print_result(result: speech.SpeechRecognitionResult):
    best_alternative = result.alternatives[0]
    print("-" * 80)
    print(f"language_code: {result.language_code}")
    print(f"transcript:    {best_alternative.transcript}")
    print(f"confidence:    {best_alternative.confidence:.0%}")


def print_result(result: speech.SpeechRecognitionResult):
    best_alternative = result.alternatives[0]
    print("-" * 80)
    print(f"language_code: {result.language_code}")
    print(f"transcript:    {best_alternative.transcript}")
    print(f"confidence:    {best_alternative.confidence:.0%}")
    print("-" * 80)
    for word in best_alternative.words:
        start_s = word.start_time.total_seconds()
        end_s = word.end_time.total_seconds()
        print(f"{start_s:>7.3f} | {end_s:>7.3f} | {word.word}")
        
        
def transcribe_local_file(file_path):
    client = speech.SpeechClient()

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        # encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        sample_rate_hertz=16000,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        language_code="en",
    )

    response = speech_to_text(config, audio)
    print_response(response)
    return response


def transcribe_gcs_file(file_path):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=file_path)
    config = speech.RecognitionConfig(
        # encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
        language_code="en",
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result()

    response = speech_to_text(config, audio)
    print_response(response)
    return response


def transcribe_gcs(gcs_uri: str) -> dict:
    """Asynchronously transcribes the audio file specified by the gcs_uri.

    Args:
        gcs_uri: The Google Cloud Storage path to an audio file.

    Returns:
        A dictionary containing the transcript, confidence levels, and timestamps.
    """
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        sample_rate_hertz=44100,
        language_code="en-US",
        enable_word_time_offsets=True  # 이 옵션을 추가하여 단어의 시작 및 종료 타임스탬프를 활성화합니다.
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    transcript_builder = {
        'transcript': "",
        'confidence': [],
        'timestamps': []  
    }

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        # transcript_builder['transcript'].append(result.alternatives[0].transcript)
        # transcript_builder['confidence'].append(result.alternatives[0].confidence)
        transcript_builder['transcript'] += result.alternatives[0].transcript

        # word_times = []
        for word_info in result.alternatives[0].words:
            start_time = word_info.start_time
            end_time = word_info.end_time
            # word_times.append((word_info.word, start_time.total_seconds(), end_time.total_seconds()))
            transcript_builder['timestamps'].append((word_info.word, start_time.total_seconds(), end_time.total_seconds()))
            # word_times.append((word_info.word, start_time.seconds, end_time.seconds))

        # transcript_builder['timestamps'].append(word_times)

    return transcript_builder


def find_sentence_time(transcript_data: dict, sentence: str) -> tuple:
    """
    Args:
        transcript_data: transcribe_gcs에서 반환한 사전.
        sentence: 찾고자 하는 문장.
    Returns:
        (시작 시간, 종료 시간)의 튜플. 문장을 찾지 못하면 (None, None) 반환.
    """
    
    # 각 transcript 항목에서 원하는 문장을 검색
    words = sentence.split()
    if len(words) == 0:
        print("!! Pass length 0 target sentence : ", words)
        return (None, None)
    for idx, ts in enumerate(transcript_data['timestamps']):
        word = ts[0]
        start_time = ts[1]

        if word == words[0]:
            correct = True
            for idx_, w in enumerate(words[:-1]):
                if transcript_data['timestamps'][idx+idx_][0] != words[idx_]:
                    correct = False
                    break
                end_time = transcript_data['timestamps'][idx+idx_][2]
            if correct == True:
                return (start_time, end_time)
            
    return (None, None)  # 문장을 찾지 못했을 경우





def extract_audio_from_video(video_file, audio_file, bucket_name="scene_finder-video_bucket"):
    # 동영상 파일 로드
    clip = VideoFileClip(video_file)
    
    # 오디오 파일로 추출
    clip.audio.write_audiofile(audio_file)
    
    destination_blob_name = audio_file
    print("destination_blob_name : ", destination_blob_name)
    upload_to_bucket(bucket_name, audio_file, destination_blob_name)
    audio_path = bucket_name + destination_blob_name
    return audio_path


def upload_to_bucket(bucket_name, source_file_name, destination_blob_name):
    """GCS 버킷으로 파일을 업로드하는 함수."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"{source_file_name} 파일을 {destination_blob_name}로 GCS에 업로드하였습니다.")



import pickle
from google.cloud import storage

def save_list_to_gcs(bucket_name, blob_name, data_list):
    # GCS 클라이언트 초기화
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # 데이터 리스트를 직렬화
    serialized_data = pickle.dumps(data_list)
    
    # 직렬화된 데이터를 GCS에 저장
    blob.upload_from_string(serialized_data)

def load_list_from_gcs(bucket_name, blob_name):
    # GCS 클라이언트 초기화
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # 데이터 다운로드
    serialized_data = blob.download_as_bytes()

    # 데이터 역직렬화
    data_list = pickle.loads(serialized_data)

    return data_list


