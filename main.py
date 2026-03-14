# encoding: utf-8
import time
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 导入人脸网格的连接信息，用于手动绘图
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS, FACEMESH_TESSELATION

class FaceMeshDetector:
    def __init__(self, model_path='face_landmarker.task', maxFaces=2):
        """
        使用 Task API 和本地模型文件来初始化人脸网格检测器。
        :param model_path: .task 模型文件的路径。
        :param maxFaces: 要检测的最大人脸数。
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=maxFaces
        )
        # 创建人脸地标检测器
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def find_face_mesh(self, frame, draw=True):
        """
        在一帧图像中寻找人脸网格，并手动绘制它们。
        :param frame: 输入的视频帧 (BGR 格式)。
        :param draw: 是否在图像上绘制网格。
        :return: 处理后的图像, 只有骨架的黑色图像, 以及关键点列表。
        """
        img_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_RGB)
        detection_result = self.landmarker.detect(mp_image)

        processed_frame = frame.copy()
        skeleton_img = np.zeros(frame.shape, np.uint8)

        if detection_result.face_landmarks and draw:
            # 遍历检测到的每一张脸
            for face_landmarks in detection_result.face_landmarks:
                
                # 1. 绘制脸部轮廓线 (与原脚本行为一致)
                # 如果想绘制完整的网格，可以使用 FACEMESH_TESSELATION
                for connection in FACEMESH_CONTOURS:
                    start_idx = connection[0]
                    end_idx = connection[1]

                    if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                        start_landmark = face_landmarks[start_idx]
                        end_landmark = face_landmarks[end_idx]
                        
                        h, w, _ = frame.shape
                        start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                        end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                        
                        # 在两个图像上都画线
                        cv.line(processed_frame, start_point, end_point, (0, 255, 0), 1)
                        cv.line(skeleton_img, start_point, end_point, (0, 255, 0), 1)

                # 2. 绘制轮廓上的关键点
                for landmark in face_landmarks:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv.circle(processed_frame, (cx, cy), 1, (0, 0, 255), -1)
                    cv.circle(skeleton_img, (cx, cy), 1, (0, 0, 255), -1)
                
                # --- 手动绘图结束 ---

        return processed_frame, skeleton_img, detection_result.face_landmarks

    def frame_combine(self, frame, src):
        """水平拼接两个图像。"""
        if len(frame.shape) == 3:
            frameH, frameW = frame.shape[:2]
            srcH, srcW = src.shape[:2]
            dst = np.zeros((max(frameH, srcH), frameW + srcW, 3), np.uint8)
            dst[:, :frameW] = frame[:, :]
            dst[:, frameW:] = src[:, :]
        else:
            src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            frameH, frameW = frame.shape[:2]
            imgH, imgW = src.shape[:2]
            dst = np.zeros((frameH, frameW + imgW), np.uint8)
            dst[:, :frameW] = frame[:, :]
            dst[:, frameW:] = src[:, :]
        return dst


if __name__ == '__main__':
    def detect_image(image_path, model_path='face_landmarker.task'):
        """
        加载图片并进行人脸网格检测，展示结果。
        """
        img = cv.imread(image_path)
        if img is None:
            print(f"无法加载图片: {image_path}")
            return
        detector = FaceMeshDetector(model_path=model_path)
        processed_frame, skeleton_img, _ = detector.find_face_mesh(img, draw=True)
        dst = detector.frame_combine(processed_frame, skeleton_img)
        cv.imshow('Face Mesh Detection - Image', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def detect_video(video_path, model_path='face_landmarker.task'):
        """
        加载视频并进行人脸网格检测，逐帧展示结果。
        """
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        detector = FaceMeshDetector(model_path=model_path)
        pTime = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, skeleton_img, _ = detector.find_face_mesh(frame, draw=True)
            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
            pTime = cTime
            text = "FPS : " + str(int(fps))
            cv.putText(processed_frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            dst = detector.frame_combine(processed_frame, skeleton_img)
            cv.imshow('Face Mesh Detection - Video', dst)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()

    # 示例用法（可注释掉，供后续GUI调用）
    # detect_image('test.jpg')
    # detect_video('test.mp4')