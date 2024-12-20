import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import solutions, YOLO
from ultralytics.solutions import AIGym
from ultralytics.utils.plotting import Annotator
from utils import *

class CustomAIGym(AIGym):
    def __init__(self, tracking_id=1, max_angle=140, min_angle=80, workout_type="press", **kwargs):
        super().__init__(**kwargs)  # AIGym 클래스 상속
        
        self.tracking_id = tracking_id  # 추적 대상 객체 ID
        
        self.count = [0]  # 반복 횟수 초기화
        self.stage = [None]  # 운동 상태 초기화 (예: 수축 또는 이완)
        self.prev_angle = (max_angle + min_angle)/2
        
        self.angle_history = []  # 관절 각도 변화를 저장할 리스트 - 노이즈는 제거되지 않음
        self.angle_hist_plot = [] # 관절 각도 변화를 저장할 리스트 - 노이즈는 제거됨, 플롯용
        
        self.count_history = []  # 반복 횟수 변화 기록 리스트
        
        self.speed_data = []  # 각 운동 반복의 속도(시간 차이)를 저장할 리스트
        self.angle_change_hist = [] # 각도의 변화량을 기록할 리스트
        ## TODO
        
        self.current_idx = 0
        self.time_diff = [0]
        
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.workout_type = workout_type #? 이제는 필요없다. 
        
        self.threshold = 1
        self.goal = 5
        # 가동범위가 너무 적은 경우
        # 일정 시간 이상 낮은 속도를 돌파하지 못할 경우
        # 추가...?
        # 만약 threshold가 5이상일 경우 실패.
        # 만약 목표한 개수만큼 채웠다면 성공, last 틀기.
        
    def detect_slow_movement(self, window_size=20, mindiff=1.0):
        """
        최근 window_size 프레임 동안의 평균 각도 변화량을 검사하여,
        운동 속도가 매우 느린 상태(거의 정지 상태)에 있는지 감지하는 메서드.
        
        :param window_size: 검사할 프레임 수
        :param threshold: 평균 각도 변화량 임계값
        """
        if len(self.angle_change_hist) >= window_size:
            recent_changes = self.angle_change_hist[-window_size:]
            avg_change = sum(abs(c) for c in recent_changes) / window_size
            if avg_change < mindiff:
                # 임계값 이하로 각도 변화량이 작으면 운동 속도가 매우 느린 상태로 간주
                #! 실패한 곳
                self.threshold += 1
                print(f"주의: 최근 {window_size} 프레임 동안 각도 변화량이 평균 {avg_change:.2f}° 이하입니다. 운동 속도가 매우 느립니다.")


    def update_stage(self, current_angle):
        """
        관절 각도 변화에 따라 운동의 현재 상태를 업데이트합니다.
        :param current_angle: 현재 관절 각도
        :return: 운동의 업데이트된 상태 (예: 수축 또는 이완)
        """
        if self.prev_angle is not None : 
            angle_diff = current_angle - self.prev_angle  # 현재 각도와 이전 각도의 차이 계산
        else :
            angle_diff = 0
        
        self.angle_change_hist.append(angle_diff) # 각도 변화량 기록하기. 
        
        self.detect_slow_movement(window_size=30, mindiff=1.0)
    
        # 운동 종류에 따른 변수 설정.
        if self.workout_type == "press" :
            up = "contraction"
            down = "relaxation"
            self.kpts = [6, 8, 10]
        elif self.workout_type == "pull" :
            up = "relaxation"
            down = "contraction"
            self.kpts = [6, 8, 10]
        elif self.workout_type == "legs" :
            up = "contraction"
            down = "relaxation"
            self.kpts = [12, 14, 16]            

        # 각도 변화에 따라 현재 상태 결정
        if angle_diff < -3:  # 각도가 충분히 감소한 경우 이완 단계로 설정
            current_stage = down
        elif angle_diff > 3:  # 각도가 충분히 증가한 경우 수축 단계으로 설정
            current_stage = up
        else:  # 변화가 미미한 경우 현재 상태 유지
            current_stage = "-"

        # 수축에서 이완으로 전환될 때 반복 횟수 증가
        if self.stage[0] == up and current_stage == "-" :
            # 일단 인덱스 기록. 수축/이완이 바뀔 때마다 일단 기록한다.
            # 시작 인덱스는 이전 극값의 프레임
            start_idx = self.current_idx
            # 끝 인덱스는 현재 극값의 프레임.
            end_idx = len(self.angle_history)
            # 가동 범위 평가: 이전 반복 구간의 시작 인덱스부터 현재까지
            print("index is : ",start_idx, end_idx)
            is_full = evaluate_range_of_motion(start_idx, end_idx, angle_history=self.angle_history, max_threshold=self.max_angle, min_threshold=self.min_angle)
            # 올바른 최대 가동범위일 경우
            if is_full == True : 
                self.count[0] += 1  # 반복 횟수 증가
                # 반복 횟수 증가 시 속도 계산하고, 속도 기록 리스트에 추가.
                self.time_diff[0] = calculate_repetition_speed(current_frame=end_idx, previous_frame=start_idx)  # 반복 횟수 증가 시 운동 속도 계산
                print(f"Repetition Count Incremented: {self.count[0]}")
                # 그리고 현재 인덱스를 업데이트해준다...
                self.current_idx = end_idx
            else : 
                print("Not counted.")
                # self.threshold += 1
                # 다른 여러 문제로 인해 못하겠다..
                
        # 현재 상태와 이전 각도 업데이트
        self.stage[0] = current_stage
        self.prev_angle = current_angle
        
        # 해당 프레임 시점에 반복 횟수를 리스트로 추가. 
        # 잘 된다는 사실 확인. 
        self.count_history.append(self.count[0])  # 반복 횟수 기록 추가
        self.speed_data.append(round(self.time_diff[0], 3))
        return current_stage

    def annotate_image(self, im0, keypoints, current_angle, current_stage):
        """
        이미지에 키포인트, 현재 각도, 반복 횟수 및 운동 상태를 주석으로 표시합니다.
        :param im0: 주석을 추가할 입력 이미지
        :param keypoints: 이미지에서 감지된 키포인트
        :param current_angle: 현재 관절 각도
        :param current_stage: 운동의 현재 상태
        :return: 주석이 추가된 이미지
        """
        # 이미지에 특정 키포인트 그리기
        im0 = self.annotator.draw_specific_points(keypoints, self.kpts, radius=self.line_width)
        # 이미지에 각도, 반복 횟수 및 운동 상태 정보 표시
        self.annotator.plot_angle_and_count_and_stage(
            angle_text=current_angle,
            count_text=self.count[0],
            stage_text=f"{current_stage}, {self.angle_change_hist[-1]:.2f}",
            center_kpt=keypoints[1],  # 팔꿈치 키포인트를 중심으로 텍스트 표시
        )
        return im0

    def monitor(self, im0):
        """
        추적 대상 감지, 키포인트 처리 및 이미지 주석을 수행하는 메인 모니터링 함수입니다.
        :param im0: 입력 이미지
        :return: 주석이 추가된 이미지
        """
        
        tracks = track_and_detect(im0=im0, model = self.model, CFG=self.CFG)  # 이미지에서 객체 추적 및 감지 수행
        self.annotator = Annotator(im0, line_width=self.line_width)  # Annotator 인스턴스 생성

        current_angle = None  # 현재 각도를 초기화

        if tracks.boxes.id is not None:
            # 감지된 객체와 그 키포인트들을 반복 처리
            for obj_id, keypoints in zip(tracks.boxes.id.cpu().numpy(), tracks.keypoints.data):
                if obj_id == self.tracking_id:  # 지정된 ID를 가진 추적 대상만 처리
                    print(f"Processing target ID: {self.tracking_id}")
                    kpts = process_keypoints(kpts = self.kpts, keypoints = keypoints)  # 대상 객체의 키포인트 추출
                    current_angle = calculate_angle(kpts=kpts, prev_angle = self.prev_angle, angle_history = self.angle_history, annotator=self.annotator)  # 현재 관절 각도 계산
                    self.angle_hist_plot.append(current_angle)
                    current_stage = self.update_stage(current_angle)  # 운동 상태 업데이트
                    im0 = self.annotate_image(im0, kpts, current_angle, current_stage)  # 이미지에 주석 추가

        self.display_output(im0)  # 주석이 추가된 이미지 출력
        
        video_index = get_video_index_by_threshold(self.threshold)
        
        #영상 재생이 아닐 경우 - video_index에 해당하는 영상 재생. 
        
        if self.count[0] >= self.goal :
            video_index = 4
        
        return im0, video_index

    def plot_training_data(self):
        """
        관절 각도 변화, 반복 횟수, 운동 속도 데이터를 통합적으로 시각화합니다.
        """
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))

        # Plot angle history
        axs[0].plot(self.angle_hist_plot, label='Angle History', color='b')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Joint Angle (degrees)')
        axs[0].set_title('Joint Angle History Over Time')
        axs[0].legend()
        axs[0].grid(True)

        # Plot count history
        axs[1].plot(self.count_history, label='Repetition Count', color='g', marker='o')
        axs[1].set_xlabel('Repetitions')
        axs[1].set_ylabel('Count')
        axs[1].set_title('Repetition Count Over Time')
        axs[1].legend()
        axs[1].grid(True)

        # Plot speed data
        axs[2].plot(self.speed_data, label='Repetition Speed', color='r', marker='o')
        axs[2].set_xlabel('Repetitions')
        axs[2].set_ylabel('Speed (seconds)')
        axs[2].set_title('Repetition Speed Over Time')
        axs[2].legend()
        axs[2].grid(True)
        
        axs[3].plot(self.angle_change_hist, label='Angle diff', color='b', marker='o')
        axs[3].set_xlabel('Repetitions')
        axs[3].set_ylabel('Aangle diff')
        axs[3].set_title('Angle Diff Over Time')
        axs[3].legend()
        axs[3].grid(True)

        plt.tight_layout()
        plt.show()