from openai import OpenAI
import base64
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

class VideoProcessor:
    def __init__(self, interval_sec=0.5):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.interval_sec = interval_sec
        self.cap = None
        self.fps = None
        self.frame_interval = None

    def _init_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Can't open video.")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = int(self.fps * self.interval_sec)

    def _encode_image(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    def predict(self, frame):
        base64_image = self._encode_image(frame)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "這張圖片中是否發生交通事故？請用簡潔文字說明。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        print(response.choices[0].message.content)
        return True

    def process(self, video_path):
        self._init_video(video_path)
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % self.frame_interval == 0:
                if self.predict(frame): 
                    pass
            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = VideoProcessor(interval_sec=0.5)
    processor.process("/data2/enginekevin/IIV/accident_video/6.mp4")