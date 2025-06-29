from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import webbrowser
import threading
import time
import pandas as pd
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
    print('클라이언트 접속!')
    if initial_ads is not None:
        ads_info = initial_ads[['행 번호', 'AISAC_URL']].dropna().to_dict(orient='records')
        socketio.emit('ads_urls', {'ads': ads_info})
        print("📤 [서버] 최초 추천 emit 완료")


watch_log = []

import copy  # 깊은 복사
import threading

@socketio.on('watch_time')
def on_watch_time(data):
    user_watch_sec = data.get("sec", None)
    ads_row = data.get("row", None)
    watch_log.append({"row": ads_row, "sec": user_watch_sec})
    print(f'🕒 [행 번호 {ads_row}] 시청 시간(sec): {user_watch_sec}')

    if len(watch_log) >= 7:
        print(f"✅ 광고 7개 시청 완료! main.py로 전달")

        def delayed_call():
            log_copy = copy.deepcopy(watch_log)
            result_df = process_watch_log(log_copy)
            watch_log.clear()
            print("🧹 watch_log 초기화 완료")

            if result_df is not None:
                ads_info = result_df.to_dict(orient='records')
                socketio.emit('ads_urls', {'ads': ads_info})
                print("📤 [서버] 새로운 추천 광고 emit 완료!")

        threading.Timer(0.1, delayed_call).start()
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
