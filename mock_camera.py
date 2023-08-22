import asyncio
import io
import logging
import time

import cv2
import redis
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

image_cache_client = redis.Redis(host='127.0.0.1', port=6379, db=0)
IMAGE_KEY = "image_key"


async def write_to_cache(frame):
    image = Image.fromarray(frame)
    png = io.BytesIO()  # 空のio.BytesIOオブジェクトを用意
    image.save(png, format='png')  # 空のio.BytesIOオブジェクトにpngファイルとして書き込み
    b_frame = png.getvalue()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, image_cache_client.set, IMAGE_KEY, b_frame)


async def main():
    # WebcamからキャプチャするためのVideoCaptureオブジェクトを作成 (仮のサンプルビデオファイルをゲーム実況をやる場合は各人のvideogame capture deviceにする必要あり)
    # NOTE: [注意] このサンプルビデオファイルだけMITライセンス/BSD-3ラインセンスではなく以下の任天堂ライセンスに従う
    # [ネットワークサービスにおける任天堂の著作物の利用に関するガイドライン｜任天堂](https://www.nintendo.co.jp/networkservice_guideline/ja/index.html)
    cap = cv2.VideoCapture('mock_sample.mp4')
    if not cap.isOpened():
        print("Error opening video stream or file")

    # fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)

    # インターバルの設定 (1秒 = 1000ミリ秒)
    interval_ms = 1500
    # 最後に画像を保存した時刻
    last_save_time = 0
    task = None

    while True:
        # Webcamから1フレーム取得
        ret, org_frame = cap.read()

        if not ret:
            # loopする
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # キャプチャが成功したら処理を続行
        current_time = time.time()
        # cv2の画像はBGRなのでRGBに変換
        # frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB)
        frame = org_frame

        # x秒ごとに画像を保存
        if current_time - last_save_time >= interval_ms / 1000:
            if task:
                # loop = asyncio.get_event_loop()
                # loop.run_until_complete(main())
                await task
            task = asyncio.create_task(write_to_cache(frame))
            last_save_time = current_time
        else:
            # webcamの場合はsleep(0)で良いと思うが、サンプルビデオファイルの場合はsleep(1/fps)で調整する
            await asyncio.sleep(1 / fps)

        # 画面に表示
        cv2.imshow('Webcam', frame)

        # "q"を押すとループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
