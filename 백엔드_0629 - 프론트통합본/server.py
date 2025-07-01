from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import webbrowser
import threading
import time
import pandas as pd
import copy
from collections import deque

from main import process_watch_log
from AdRecommender import AdRecommender

app = Flask(__name__, static_folder='.')
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 상태 변수
watch_log = []           # 최근 10개 광고 시청 로그
user_log_df = []         # 전체 누적 로그
new_ad_df = []           # 사용자 알고리즘 추천 결과 (5개 시청마다 갱신)
next_ad_df = deque()     # emit 대기 광고 큐 (FIFO)
ads_buffer = []          # 현재 클라이언트에 재생 중인 광고 10개
mid_scores = []
played_in_batch = 0
current_batch_size = 0


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


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


@socketio.on('connect')
def on_connect():
    global ads_buffer, current_batch_size
    if ads_buffer:
        socketio.emit('ads_urls', {'ads': ads_buffer})
        print("\U0001f4e4 [서버] 최초 추천 emit 완료")
        current_batch_size = len(ads_buffer)


@socketio.on('watch_time')
def on_watch_time(data):
    global played_in_batch, watch_log, new_ad_df, next_ad_df, ads_buffer, user_log_df, mid_scores, current_batch_size

    row = data.get("row")
    sec = data.get("sec")
    watch_log.append({"row": row, "sec": sec})
    user_log_df.append({"row": row, "sec": sec})
    played_in_batch += 1

    print(f'🕒 [행 번호 {row}] 시청 시간(sec): {sec}')

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
            print("🕓 next_ad_df에 추천 광고 누적 완료 (큐 사이즈: {}개)".format(len(next_ad_df)))

    # 현재 광고 10개 시청 완료 시점 → next_ad_df에서 하나 꺼내서 emit
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


# 최초 초기 광고 생성
initial_ads = initial_recommend_ads()
ads_buffer = initial_ads[['행 번호', 'AISAC_URL']].dropna().to_dict('records')
current_batch_size = len(ads_buffer)

print("🚀 서버 실행 중: http://localhost:5000")
if __name__ == '__main__':
    def open_browser():
        time.sleep(1)
        webbrowser.open("http://localhost:5000")
    threading.Thread(target=open_browser).start()
    socketio.run(app, port=5000)
