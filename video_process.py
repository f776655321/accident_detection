from openai import OpenAI
import base64
import cv2
import numpy as np
import os
from PIL import Image
from cls_model.model import Resnet50
from dotenv import load_dotenv
import random
import torch

load_dotenv(dotenv_path='.env')

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using CUDA

set_seed(42)

class VideoProcessor:
    def __init__(self, interval_sec=0.5, model=None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.interval_sec = interval_sec
        self.cap = None
        self.fps = None
        self.frame_interval = None
        self.model = model

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
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        return self.model.predict(image_pil) == 0 # 0 for Accident, 1 for Non Accident

    def llm(self, frames):
        content = [
            {
                "type": "text",
                "text": (
                    "以下是從交通監視器中連續擷取的 3 張事故相關畫面，已經確認有事故發生，請用簡潔中文描述畫面內容。"
                    "再模仿交通廣播（如警廣 FM）即時播報的語氣，用簡潔、專業的中文結合畫面內容，進行播報。"
                    "時間：7:30 AM、地點：台北市忠孝東路三段。"
                    "將播報內容敘述在「[播報內容]：」後。"
                )
            }
        ]

        for idx, frame in enumerate(frames):
            base64_image = self._encode_image(frame)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        print("[GPT 回覆]:\n", response.choices[0].message.content)
        return

    def process(self, video_path):
        self._init_video(video_path)
        frame_count = 0
        accident_frames = []  # 保存連續事故 frame

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_count % self.frame_interval == 0:
                if self.predict(frame):
                    
                    accident_frames.append(frame)  # 保存 frame（copy 避免後續被覆蓋）
                    if len(accident_frames) == 3:
                        print(f"🚨 連續 3 個事故 frame，執行 LLM 說明")
                        selected = [accident_frames[0], accident_frames[1], accident_frames[2]]
                        # save_dir = "frames"
                        # os.makedirs(save_dir, exist_ok=True)

                        # cv2.imwrite(os.path.join(save_dir, "accident_0.jpg"), accident_frames[0])
                        # cv2.imwrite(os.path.join(save_dir, "accident_1.jpg"), accident_frames[1])
                        # cv2.imwrite(os.path.join(save_dir, "accident_2.jpg"), accident_frames[2])
                        self.llm(selected)
                        accident_frames.clear()  # 重置
                        break
                else:
                    accident_frames.clear()  # 中斷連續 → 清空
            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Establish Video Processor...")
    processor = VideoProcessor(interval_sec=0.2, model=Resnet50('Resnet50.pth'))
    input("Press Enter to start processing...")
    processor.process("video/5.mp4")