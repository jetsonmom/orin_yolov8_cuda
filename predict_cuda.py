from ultralytics import YOLO
from PIL import Image
import cv2
import torch

# CUDA 상세 정보 출력
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')

# 기존 코드
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device: '+ device)

# 모델 로드 전 메모리 사용량
if device == 'cuda':
    print(f'GPU memory before model load: {torch.cuda.memory_allocated()/1024**2:.2f} MB')

# 모델 로드
model = YOLO('yolov8n.pt').to(device)

# 모델 로드 후 메모리 사용량
if device == 'cuda':
    print(f'GPU memory after model load: {torch.cuda.memory_allocated()/1024**2:.2f} MB')

source = "0"  # camera
results = model.predict(source, show=True, save=True)

# 추론 후 메모리 사용량
if device == 'cuda':
    print(f'GPU memory after inference: {torch.cuda.memory_allocated()/1024**2:.2f} MB')
