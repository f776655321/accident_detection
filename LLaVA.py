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
import json
import time
# import websockets
import asyncio
from transformers import AutoProcessor, LlavaForConditionalGeneration

load_dotenv(dotenv_path='.env')

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if using CUDA

# async def send_data(text):
#     uri = "ws://localhost:8080"
#     async with websockets.connect(uri) as websocket:
#         data = {"type": "text-input", "payload": text}
#         await websocket.send(json.dumps(data))
#         response = await websocket.recv()
#         print("Received:", response)

set_seed(42)

class VideoProcessor:
    def __init__(self, processer, LM, interval_sec=0.5, model=None):
        self.processer = processer
        self.LM = LM
        self.interval_sec = interval_sec
        self.cap = None
        self.fps = None
        self.frame_interval = None
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "以下是從交通監視器中連續擷取的 3 張事故相關畫面，已經確認有事故發生，請用簡潔中文描述畫面內容。"
                            "再模仿交通廣播（如警廣 FM）即時播報的語氣，用簡潔、專業的中文結合畫面內容，進行播報。"
                            "時間：7:30 AM、地點：台北市忠孝東路三段。"
                            "將播報內容敘述在「[播報內容]：」後。"
                        )
                    },
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "image"},
                ]
            }
        ]
        prompt = self.processer.apply_chat_template(conversation, add_generation_prompt=True)

        pil_images = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(rgb))

        # 4. 把影像跟 prompt 一起 feed 進 processor，得到模型的輸入 tensors
        #    這裡 return_tensors="pt" 會回傳一個 dict 裡面有 input_ids, attention_mask, pixel_values 等 tensor
        #    再把它們移到同一個 device（例如 GPU），並且指定以 float16 精度（若你的 GPU 支援）
        inputs = self.processer(
            images=pil_images,
            text=prompt,
            return_tensors="pt"
        ).to(self.device, torch.float16)

        # 5. 呼叫 model.generate 取得 output token 序列
        #    這邊 max_new_tokens 可以自己調整，看要讓模型最多生成多少字
        output_ids = self.LM.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False  # 若要抽樣可以改成 True 並且加上 temperature/top_p/top_k
        )

        # 6. 用 processor.decode 把 output_ids 轉成中文文字
        #    output_ids[0] 代表第一條回應。如果要跳過前面 special tokens（例如 BOS/EOS），可以像範例一樣把前兩個 token 跳過： output_ids[0][2:]
        decoded = self.processer.decode(output_ids[0][2:], skip_special_tokens=True)

        # 7. 擷取「[播報內容]：」之後的部分
        if "[播報內容]：" in decoded:
            return decoded.split("[播報內容]：")[-1].strip()
        else:
            # 如果沒有標記，就直接回傳全部
            return decoded.strip()

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
                        print(f"🚨 連續 3 個事故 frame，執行 GPT 說明")
                        selected = [accident_frames[0], accident_frames[1], accident_frames[2]]
                        save_dir = "frames"
                        os.makedirs(save_dir, exist_ok=True)

                        # cv2.imwrite(os.path.join(save_dir, "accident_0.jpg"), accident_frames[0])
                        # cv2.imwrite(os.path.join(save_dir, "accident_1.jpg"), accident_frames[1])
                        # cv2.imwrite(os.path.join(save_dir, "accident_2.jpg"), accident_frames[2])
                        start = time.time()
                        llm_text = self.llm(selected)
                        print(llm_text)
                        end = time.time()
                        print(f"LLM 處理時間: {end - start:.2f} 秒")
                        # asyncio.run(send_data(llm_text))
                        accident_frames.clear()  # 重置
                        break
                else:
                    accident_frames.clear()  # 中斷連續 → 清空
            frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Establish Video Processor...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    processer = AutoProcessor.from_pretrained(model_id)
    v = VideoProcessor(processer=processer, LM=model, interval_sec=0.2, model=Resnet50('Resnet50.pth'))
    input("Press Enter to start processing...")
    v.process("video/5.mp4")