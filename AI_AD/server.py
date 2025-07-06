from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import webbrowser
import threading
import time
import pandas as pd
import copy
import os
import json
from collections import deque

from main import process_watch_log
from AdRecommender import AdRecommender

app = Flask(__name__, static_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 상태 변수
watch_log = []           # 최근 광고 시청 로그
user_log_df = []         # 전체 누적 로그
new_ad_df = []           # 사용자 알고리즘 추천 결과 (15개 시청마다 갱신)
next_ad_df = deque()     # emit 대기 광고 큐 (FIFO)
ads_buffer = []          # 현재 클라이언트에 재생 중인 광고 개수
mid_scores = []
played_in_batch = 0
current_batch_size = 0
LOG_PATH = './recs/user_watch_log.csv' 

# user_config.json으로부터 사용자 정보 불러오기
def get_user_segment(config_path='user_config.json'):
    try:
        with open(config_path, encoding='utf-8') as f:
            user = json.load(f)
        gender = user.get('gender')
        age = user.get('age')

        # 클러스터링 결과에 맞춘 세그먼트 분류
        if age <= 9:
            return '어린이_4-9세'
        elif 10 <= age <= 39:
            return '청장년_10-30대'
        elif 40 <= age <= 49 and gender == 1:
            return '중장년_40-50대_남성'  # 40~49세 여성
        elif 40 <= age <= 59 and gender == 0:
            return '중장년_40-50대_남성'  # 40~59세 남성
        elif 50 <= age <= 59 and gender == 1:
            return '시니어_50-60대'  # 50대 여성도 cluster 1
        elif age >= 60:
            return '시니어_50-60대'
        else:
            return '청장년_10-30대'  # 기본값
        
    except Exception as e:
        print(f"⚠️ user_config.json 불러오기 실패 → 기본 세그먼트 사용 / 이유: {e}")
        return '청장년_10-30대'

def initial_recommend_ads(user_segment='청장년_10-30대', num_of_queue=10):
    data_path = './data/AISAC 광고소재명별 광고 정보.csv'
    ad_data = pd.read_csv(data_path)
    engine = AdRecommender(ad_data)
    engine.fit()
    props = engine.infer_major_props(user_segment, epsilon=0.2)
    initial_ads_df = engine.recommend(
        segment=user_segment,
        top_k=num_of_queue,
        candidate_size=len(ad_data),
        w_title=0.2,
        w_cat=0.8,
        major_props=props,
        max_per_cat=3,
        top_n_per_cat=5,
        temperature=1.5
    )
    return initial_ads_df


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/broadcast-program/<path:filename>')
def serve_broadcast_program(filename):
    return send_from_directory('broadcast-program', filename)

@socketio.on('connect')
def on_connect():
    print("✅ 클라이언트 연결됨")
    # 연결 시에는 광고 emit하지 않음

@socketio.on('check_user_config')
def check_user_config():
    exists = os.path.exists('user_config.json')
    socketio.emit('user_config_exists', {'exists': exists})

@socketio.on('user_info')
def handle_user_info(data):
    gender = data.get('gender')
    age = data.get('age')
    print(f"[user_info] gender: {gender}, age: {age}")
    # 필요하다면 파일 저장 또는 전역 변수에 저장 등 추가 처리
    # 예시: user_config.json에 저장
    try:
        gender_num = 0 if gender == 'male' else 1  # 남성:0, 여성:1
        user_info = {
            'user_id': 'user_001',
            'gender': gender_num, 
            'age': int(age)
            }
        with open('user_config.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(user_info, f, ensure_ascii=False)
        print("✅ user_config.json 저장 완료")
    except Exception as e:
        print(f"❌ user_info 저장 실패: {e}")

@socketio.on('request_ads')
def handle_request_ads():
    """1. user_watch_log.csv 있으면 행동기반 추천
       2. user_config.json만 있으면 세그먼트별 초기 추천
       3. 둘 다 없으면 개인정보 수집 UI 요청"""
    global ads_buffer, current_batch_size
    try:
        # 1. 행동기반 추천
        if os.path.exists(LOG_PATH):
            print("📂 user_watch_log.csv 감지됨 → 사용자 알고리즘 추천 수행")
            log_df = pd.read_csv(LOG_PATH)
            watch_log_data = log_df.to_dict(orient='records')
            result = process_watch_log(watch_log_data, top_k=10)

            if result and "ads" in result:
                ads_df = result["ads"]
                mid_score_data = result["mid_scores"]
                ads_buffer = ads_df.to_dict('records')
                current_batch_size = len(ads_buffer)
                socketio.emit('ads_urls', {
                    'ads': ads_buffer,
                    'mid_scores': mid_score_data.to_dict('records')
                })
                print("🎯 user_watch_log 기반 광고 emit 완료")
                return  # 추천 성공 시 종료

        # 2. user_config.json 기반 초기 추천
        elif os.path.exists('user_config.json'):
            print("🗂 user_config.json 감지됨 → 세그먼트별 초기 추천 수행")
            user_segment = get_user_segment()
            try:
                initial_ads = initial_recommend_ads(user_segment=user_segment, num_of_queue=10)
            except Exception as e_inner:
                print(f"⚠️ 초기 추천 실패 → 기본 세그먼트 사용 / 이유: {e_inner}")
                initial_ads = initial_recommend_ads(user_segment='청장년_10-30대', num_of_queue=10)

            ads_buffer = initial_ads[['행 번호', 'AISAC_URL']].dropna().to_dict('records')
            current_batch_size = len(ads_buffer)
            socketio.emit('ads_urls', {'ads': ads_buffer})
            print(f"🚀 [{user_segment}] user_config 기반 초기 맞춤 광고 emit 완료")
            return

        # 3. 개인정보 수집 UI 요청
        else:
            print("⚠️ user_config.json 없음 → 개인정보 수집 UI 요청")
            socketio.emit('need_user_info')
            # 프론트엔드에서 user_info 이벤트로 개인정보를 받아 저장 후 다시 광고 요청

    except Exception as e:
        print(f"❌ 전체 추천 emit 실패: {e}")



@socketio.on('watch_time')
def on_watch_time(data):
    global played_in_batch, watch_log, new_ad_df, next_ad_df, ads_buffer, user_log_df, mid_scores, current_batch_size

    row = data.get("row")
    sec = data.get("sec")
    watch_log.append({"row": row, "sec": sec})
    user_log_df.append({"row": row, "sec": sec})
    played_in_batch += 1

    print(f'🕒 [행 번호 {row}] 시청 시간(sec): {sec}')

    # user_log_df → CSV 저장
    try:
        df = pd.DataFrame([{"row": row, "sec": sec}])
        if not os.path.exists(LOG_PATH):
            df.to_csv(LOG_PATH, index=False)
            print("📁 로그 파일 새로 생성 완료")
        else:
            df.to_csv(LOG_PATH, mode='a', header=False, index=False)
    except Exception as e:
        print(f"❌ 로그 저장 실패: {e}")

    # 5개 시청 시 사용자 알고리즘 호출
    if played_in_batch % 5 == 0:
        print("✅ 5개 시청됨 → 사용자 알고리즘 추천 생성")
        log_copy = copy.deepcopy(user_log_df)
        result = process_watch_log(log_copy)
        if result and "ads" in result:
            new_ad_df = result["ads"].to_dict('records')
            mid_scores = result["mid_scores"].to_dict('records')
            print("📦 사용자 알고리즘 추천 저장 완료 (new_ad_df)")
            # 큐에 추가 (선입선출)
            next_ad_df.append(new_ad_df)

    # 현재 광고 배치 시청 완료 시점 → next_ad_df에서 하나 꺼내서 emit
    if played_in_batch >= current_batch_size:
        print("\U0001f504 배치 종료 → 다음 광고 emit")
        watch_log.clear()
        played_in_batch = 0

        if next_ad_df:
            ads_buffer = next_ad_df.popleft()
            current_batch_size = len(ads_buffer)
            socketio.emit('ads_urls', {
                'ads': ads_buffer,
                'mid_scores': mid_scores
            })
        else:
            print("⚠️ next_ad_df 비어 있음 – fallback 로직 필요")

# AD_URLS에 계속해서 광고 URL이 쌓이게끔 자동화? 성공!
# print(initial_ads)
# AD_URLS = initial_ads['AISAC_URL'].dropna().tolist()
# # 최초 초기 광고 생성
# initial_ads = initial_recommend_ads()
# ads_buffer = initial_ads[['행 번호', 'AISAC_URL']].dropna().to_dict('records')
# current_batch_size = len(ads_buffer)

if __name__ == '__main__':
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=open_browser).start()
    socketio.run(app, port=5000, debug=True, use_reloader=False)
