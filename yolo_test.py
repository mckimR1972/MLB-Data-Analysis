import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 및 비디오 로드
print("YOLOv8 모델을 로드합니다...")
model = YOLO('yolov8n.pt') 

video_path = 'test.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 비디오 저장 설정
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_yolo_tracking.mp4', fourcc, fps, (width, height))

points = []
print("YOLO 기반 궤적 추적을 시작합니다. (ROI 및 물리 필터 적용)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2. 타이트한 관심 영역(ROI) 설정 
    # 페랄타(투수)의 몸통과 저지(타자)의 움직임을 배제하기 위한 경계선
    y_min, y_max = int(height * 0.25), int(height * 0.65) # 스코어보드 아래 ~ 잔디밭 위
    x_min, x_max = int(width * 0.40), int(width * 0.65)  # 투수 우측 ~ 포수/타자 앞

    # 디버깅용: 설정된 ROI 영역을 파란색 사각형으로 화면에 표시 (필요시 주석 해제)
    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # 3. YOLO 추론 (핵심 파라미터 튜닝)
    # imgsz=1280: 화면을 확대 분석하여 작고 뭉개진 공을 더 잘 찾게 함
    # conf=0.05: 확신도가 5%만 되어도 일단 탐지 (모션 블러 극복용)
    results = model(frame, conf=0.05, imgsz=1280, classes=32, verbose=False)
    
    candidates = []
    
    # 4. 탐지된 객체 필터링
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        w, h = x2 - x1, y2 - y1
        
        # 조건 A: 공의 중심 좌표가 우리가 설정한 ROI(투수와 포수 사이) 안에 있는가?
        if x_min < cx < x_max and y_min < cy < y_max:
            # 조건 B: 바운딩 박스의 크기가 너무 크지 않은가? (유니폼 등을 오인하는 것 방지)
            if w < 40 and h < 40: 
                candidates.append((cx, cy))

    # 5. 궤적 추적 및 텔레포트 방지
    best_pt = None
    if candidates:
        if not points:
            # 릴리스 포인트 찾기: 이 앵글에서는 보통 x값이 가장 작은 것(가장 왼쪽)이 투수 손에 가깝습니다.
            best_pt = min(candidates, key=lambda p: p[0])
        else:
            last_pt = points[-1]
            valid_candidates = []
            
            for pt in candidates:
                dx = pt[0] - last_pt[0]
                dy = pt[1] - last_pt[1]
                dist_sq = dx**2 + dy**2
                
                # 물리적 제약: 1프레임만에 100픽셀(제곱 10000) 이상 순간이동 할 수 없음
                if dist_sq < 10000:
                    valid_candidates.append((dist_sq, pt))
            
            if valid_candidates:
                # 조건을 통과한 것 중 이전 위치와 가장 가까운 점을 다음 공으로 선택
                valid_candidates.sort(key=lambda x: x[0])
                best_pt = valid_candidates[0][1]

    if best_pt:
        points.append(best_pt)
        cv2.circle(frame, best_pt, 4, (0, 0, 255), -1) # 현재 공 위치 빨간 점

    # 6. 궤적 선 그리기
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 3)

    out.write(frame)
    cv2.imshow("YOLOv8 Pitch Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("분석 완료! 'output_yolo_tracking.mp4'를 확인하세요.")