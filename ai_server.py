import uvicorn
from fastapi import FastAPI, UploadFile, File
import openvino as ov
import numpy as np
import cv2
import sys
import torch

# [!!!] --- v6.0: YOLOv5의 "공식" 유틸리티 수입 --- [!!!]
yolov5_path = "C:/dev/GNU_REAL_PBL/yolov5"
sys.path.append(yolov5_path)
from utils.general import non_max_suppression
# [!!!] -------------------------------------------------- [!!!]


# --- (1) NPU로 AI 모델 로드하기 ---
print("AI 서버: NPU용 모델을 로드하는 중...")
core = ov.Core()

# [!!] 경로 확인 필수! (exp2, exp3 등) [!!]
onnx_model_path = "C:/dev/GNU_REAL_PBL/yolov5/runs/train/exp3/weights/best.onnx" 
model = core.read_model(model=onnx_model_path)
compiled_model = core.compile_model(model=model, device_name="NPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
input_shape = input_layer.shape 
input_height, input_width = input_shape[2], input_shape[3]
print(f"AI 모델 로드 완료. 입력 크기: ({input_height}, {input_width})")
# ------------------------------------

# --- (2) data.yaml과 100% 일치하는 정답지 ---
class_names = ['can', 'general_waste', 'glass', 'paperpack', 'plastic', 'vinyl'] 
num_classes = len(class_names)
# ------------------------------------

app = FastAPI()

# [!!!] --- (3) "진짜" 전처리 함수 (v6.0 - Letterbox) --- [!!!]
def preprocess_image_letterbox(image_data, new_shape=(640, 640), color=(114, 114, 114)):
    # 1. 바이너리 데이터를 이미지로 디코딩 (BGR)
    nparr = np.frombuffer(image_data, np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. 원본 이미지 크기
    shape = im.shape[:2]  # (height, width)
    
    # 3. 새 크기(640x640)에 맞게 비율 계산
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 여백
    dw /= 2  # 양쪽 여백
    dh /= 2
    
    # 4. '찌그러뜨리지 않고' 비율에 맞게 리사이즈
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 5. 검은색(114) 여백(Letterbox) 추가
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # 6. BGR -> RGB로 색상 순서 바로잡기
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # 7. 이미지 차원 변경 (HWC -> CHW) 및 정규화
    im_transposed = im_rgb.transpose(2, 0, 1) # [H, W, C] -> [C, H, W]
    im_normalized = im_transposed.astype(np.float32) / 255.0
    
    # 8. 배치 차원 추가
    input_tensor = np.expand_dims(im_normalized, 0)
    return input_tensor, shape
# [!!!] ---------------------------------------------------- [!!!]


@app.get("/")
def read_root():
    return {"Hello": "Welcome to Re:Cycle AI Server (v1.5+ NPU Ready)"}


# --- (4) 핵심 API: AI 이미지 분석 (v6.0) ---
@app.post("/api/ai/classify")
async def classify_image(image: UploadFile = File(...)):
    
    # 1. 이미지 읽기
    image_data = await image.read()
    
    # 2. 이미지 전처리 (v6.0 Letterbox 함수 호출!)
    input_tensor, original_shape = preprocess_image_letterbox(image_data, (input_height, input_width))
    
    # 3. [NPU 실행!] AI 모델로 예측 실행
    results = compiled_model([input_tensor])[output_layer] # Shape [1, 25200, 9]

    # [!!!] --- v5.0: "공식" 뇌 해석기(NMS) 사용 --- [!!!]
    pred = torch.tensor(results)
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
    
    if len(pred[0]):
        # 7. 찾은 물체들 중, confidence(확신도)가 가장 높은 놈을 찾음
        best_prediction = max(pred[0], key=lambda x: x[4]) # 4번 인덱스가 확신도
        
        # 8. 그 놈의 클래스 인덱스(5번)를 정답으로 확정
        best_class_index = int(best_prediction[5])
        prediction_name = class_names[best_class_index]
    else:
        # 9. NMS가 "아무것도 못 찾겠다"고 하면
        prediction_name = "unknown" # (또는 'background')
    # [!!!] --------------------------------------------- [!!!]
    
    print(f"이미지 분석 완료: {prediction_name}")
    
    # 7. 명세서대로 '이름'을 JSON으로 반환
    return {"prediction": prediction_name}


if __name__ == "__main__":
    print("AI 서버를 http://localhost:8001 에서 시작합니다.")
    uvicorn.run(app, host="0.0.0.0", port=8001)