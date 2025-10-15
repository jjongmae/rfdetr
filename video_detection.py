"""
RF-DETR Video Detection Script
영상을 실시간으로 검지하고 결과를 화면에 표시하며 저장하는 스크립트
"""

import cv2
import numpy as np
from datetime import datetime
from rfdetr import RFDETRBase, RFDETRSmall, RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# ========== 설정 상수 ==========
# 비디오 소스 (웹캠: 0, 1, 2... / 파일: 파일 경로)
VIDEO_SOURCE = f"D:\data\자료요청_20251014\양지_20250220021010_20250220021240.avi"

# 결과 저장 경로 (None이면 저장 안 함)
OUTPUT_PATH = f"output.mp4"

# 모델 크기 ('nano', 'small', 'base')
MODEL_SIZE = 'base'

# 검지 신뢰도 임계값 (0.0 ~ 1.0)
CONFIDENCE_THRESHOLD = 0.5

# 화면 표시 여부
DISPLAY = True
# ==============================


class VideoDetector:
    """RF-DETR을 사용한 영상 검지 클래스"""

    def __init__(self, model_size='base', confidence_threshold=0.5):
        """
        Args:
            model_size: 모델 크기 ('nano', 'small', 'base')
            confidence_threshold: 검지 신뢰도 임계값
        """
        print(f"RF-DETR {model_size} 모델 로딩 중...")

        # 모델 선택
        if model_size.lower() == 'nano':
            self.model = RFDETRNano()
        elif model_size.lower() == 'small':
            self.model = RFDETRSmall()
        else:
            self.model = RFDETRBase()

        # 추론 최적화 (에러 발생 시 스킵)
        try:
            self.model.optimize_for_inference()
            print("모델 최적화 완료")
        except Exception as e:
            print(f"모델 최적화 스킵 (에러: {e})")

        self.confidence_threshold = confidence_threshold

        print("모델 로딩 완료!")

    def process_video(self, video_source, output_path=None, display=True):
        """
        비디오를 처리하여 객체 검지 수행

        Args:
            video_source: 비디오 파일 경로 또는 웹캠 번호 (0, 1, ...)
            output_path: 결과 영상 저장 경로 (None이면 저장하지 않음)
            display: 실시간 화면 표시 여부
        """
        # 비디오 캡처 객체 생성
        if isinstance(video_source, int) or video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
            print(f"웹캠 {video_source} 사용")
        else:
            cap = cv2.VideoCapture(video_source)
            print(f"비디오 파일: {video_source}")

        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_source}")

        # 비디오 속성 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"비디오 정보: {width}x{height} @ {fps}fps, 총 {total_frames} 프레임")

        # 비디오 저장 설정
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"결과 저장 경로: {output_path}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # BGR을 RGB로 변환 (RF-DETR은 RGB 입력 필요)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 객체 검지 수행
                detections = self.model.predict(frame_rgb, threshold=self.confidence_threshold)

                # 디버깅: 검지된 클래스 출력 (처음 5프레임만)
                if frame_count <= 5 and len(detections) > 0:
                    class_ids = detections.class_id if hasattr(detections, 'class_id') else None
                    confidences = detections.confidence if hasattr(detections, 'confidence') else None
                    if class_ids is not None:
                        print(f"\n[프레임 {frame_count}] 검지된 클래스:")
                        for i, cid in enumerate(class_ids):
                            conf = confidences[i] if confidences is not None else 0
                            class_name = COCO_CLASSES.get(int(cid), f"Class{int(cid)}")
                            print(f"  - ID: {int(cid)}, 이름: {class_name}, 신뢰도: {conf:.3f}")

                # 검지 결과를 프레임에 그리기
                annotated_frame = self.draw_detections(frame.copy(), detections)

                # 검지 개수 (로그용)
                num_detections = len(detections) if hasattr(detections, '__len__') else 0

                # 화면 표시
                if display:
                    cv2.imshow('RF-DETR Detection', annotated_frame)

                    # 'q' 키를 누르면 종료, 스페이스바로 일시정지
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n사용자에 의해 종료됨")
                        break
                    elif key == ord(' '):
                        print("일시정지 (아무 키나 눌러 계속)")
                        cv2.waitKey(0)

                # 결과 저장
                if writer:
                    writer.write(annotated_frame)

                # 진행 상황 출력
                if frame_count % 30 == 0:
                    print(f"처리 중: {frame_count} 프레임, 검지 개수: {num_detections}")

        finally:
            # 리소스 정리
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            print(f"\n처리 완료: 총 {frame_count} 프레임")

    def draw_detections(self, frame, detections):
        """
        검지 결과를 프레임에 그리기

        Args:
            frame: 원본 프레임 (BGR)
            detections: supervision.Detections 객체
        """
        # supervision.Detections 객체 처리
        if not hasattr(detections, 'xyxy') or len(detections) == 0:
            return frame

        # supervision.Detections 객체의 속성 접근
        boxes = detections.xyxy  # [N, 4] - (x1, y1, x2, y2)
        class_ids = detections.class_id if hasattr(detections, 'class_id') else None
        scores = detections.confidence if hasattr(detections, 'confidence') else None

        # NumPy 배열로 변환
        if hasattr(boxes, 'cpu'):  # PyTorch tensor인 경우
            boxes = boxes.cpu().numpy()
        if class_ids is not None and hasattr(class_ids, 'cpu'):
            class_ids = class_ids.cpu().numpy()
        if scores is not None and hasattr(scores, 'cpu'):
            scores = scores.cpu().numpy()

        # 각 검지 결과 그리기
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(class_ids[i]) if class_ids is not None and i < len(class_ids) else 0
            score = float(scores[i]) if scores is not None and i < len(scores) else 0.0

            # 바운딩 박스 좌표
            x1, y1, x2, y2 = map(int, box)

            # 랜덤 색상 (클래스별 고정)
            color = self.get_color(class_id)

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 클래스 이름 가져오기 (COCO_CLASSES는 딕셔너리)
            class_name = COCO_CLASSES.get(class_id, f"Class{class_id}")

            # 레이블 텍스트
            label = f"{class_name}: {score:.2f}"

            # 레이블 배경
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - label_height - 10),
                         (x1 + label_width, y1), color, -1)

            # 레이블 텍스트
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    @staticmethod
    def get_color(class_id):
        """클래스 ID에 따른 고정 색상 반환"""
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color


def main():
    # 출력 경로 자동 생성 (웹캠 사용 시)
    output_path = OUTPUT_PATH
    if output_path is None and isinstance(VIDEO_SOURCE, int):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"detection_webcam_{timestamp}.mp4"
        print(f"출력 경로 자동 생성: {output_path}")

    # 검지기 생성 및 실행
    detector = VideoDetector(model_size=MODEL_SIZE,
                            confidence_threshold=CONFIDENCE_THRESHOLD)

    detector.process_video(
        video_source=VIDEO_SOURCE,
        output_path=output_path,
        display=DISPLAY
    )


if __name__ == '__main__':
    main()
