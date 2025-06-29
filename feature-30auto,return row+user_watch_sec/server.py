from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import webbrowser
import threading
import time
import pandas as pd

current_batch_size = 0   # 이번에 클라이언트에 내려간 큐 길이
played_in_batch    = 0   # 이번 배치 안에서 몇 개 봤는지 카운트
pending_ads        = []  # 다음 라운드에 내려줄 광고 리스트 (dict 목록)

from main import process_watch_log
from AdRecommender import AdRecommender

def initial_recommend_ads(user_segment='중장년_40-50대_남성', num_of_queue=10):
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

app = Flask(__name__, static_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@socketio.on('connect')
def on_connect():
    ...
    if initial_ads is not None:
        ads_info = initial_ads[['행 번호', 'AISAC_URL']].dropna().to_dict('records')
        socketio.emit('ads_urls', {'ads': ads_info})
        print("📤 [서버] 최초 추천 emit 완료")

        # ←★ 추가
        global current_batch_size
        current_batch_size = len(ads_info)   # 보통 10

watch_log = []

import copy  # 깊은 복사
import threading

@socketio.on('watch_time')
def on_watch_time(data):
    global played_in_batch, pending_ads, current_batch_size

    # ── 기존 로깅 그대로 ───────────────────────────
    user_watch_sec = data.get("sec")
    ads_row        = data.get("row")
    watch_log.append({"row": ads_row, "sec": user_watch_sec})
    played_in_batch += 1
    print(f'🕒 [행 번호 {ads_row}] 시청 시간(sec): {user_watch_sec}')

    # ① 7개째 → 다음 추천을 "계산만" 해 둔다
    if played_in_batch == 7 and not pending_ads:
        print("✅ 광고 7개 시청 → 다음 추천 미리 계산")

        # 필요 로그만 복사해서 추천 계산
        log_copy = copy.deepcopy(watch_log)
        result_df = process_watch_log(log_copy)

        if result_df is not None and not result_df.empty:
            pending_ads = result_df.to_dict('records')
            print(f"📦 다음 추천 {len(pending_ads)}개 준비 완료!")

    # ② 배치 전부 소진 → 준비해 둔 리스트를 emit
    if played_in_batch >= current_batch_size:
        print("🔄 배치 종료 → 새 추천 emit")
        watch_log.clear()           # 새 라운드를 위해 초기화
        played_in_batch = 0

        if pending_ads:             # 미리 계산돼 있으면 바로 사용
            socketio.emit('ads_urls', {'ads': pending_ads})
            current_batch_size = len(pending_ads)  # 다음 배치 크기
            pending_ads = []        # 큐 비우기
            print("📤 새 추천 광고 emit 완료!")
        else:
            print("⚠️ pending_ads 비어 있음 – fallback 로직 필요")

initial_ads = initial_recommend_ads()

# AD_URLS에 계속해서 광고 URL이 쌓이게끔 자동화? 성공!
print(initial_ads)
AD_URLS = initial_ads['AISAC_URL'].dropna().tolist()

if __name__ == '__main__':
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=open_browser).start()

    print("🚀 서버 실행 중: http://localhost:5000")
    socketio.run(app, port=5000)
