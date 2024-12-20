import numpy as np

def track_and_detect(im0, model, CFG):
    """
    입력 이미지에서 객체 추적 및 감지를 수행합니다.
    :param im0: 입력 이미지
    :return: 감지된 객체 및 그 정보를 포함한 트랙
    """
    # im0 = im0.cpu().numpy()
    # print(im0)
    
    tracks = model.track(source=im0, persist=True, classes = CFG["classes"])[0]
    print(tracks.boxes.id.cpu().numpy() if tracks.boxes.id is not None else "No IDs detected")
    return tracks

def smooth_angles(angle_list, base_window_size=15):
    """
    적응형 이동 평균 필터를 사용하여 노이즈를 완화하되 극값을 유지합니다.

    :param angle_list: 각도 기록 리스트
    :param base_window_size: 기본 윈도우 크기 (기본값 10)
    :return: 부드럽게 처리된 각도 값
    """
    if len(angle_list) < base_window_size:
        return angle_list[-1]

    # 지수적 증가 가중치 적용: 최근 데이터에 훨씬 더 높은 가중치를 부여
    weights = np.exp(np.linspace(0, 1, base_window_size))  # 0에서 2까지 선형 증가 후 이를 지수로 변환하여 가중치 생성
    weights /= np.sum(weights)  # 가중치 합이 1이 되도록 정규화
    smoothed_angle = np.sum(np.array(angle_list[-base_window_size:]) * weights)
    return smoothed_angle

def calculate_angle(kpts, annotator, angle_history, prev_angle):
    """
    주어진 키포인트를 사용하여 관절 각도를 계산하고 부드럽게 처리합니다.
    :param kpts: 관절 각도를 계산하기 위한 키포인트
    :return: 부드럽게 처리된 관절 각도
    """
    # 키포인트의 수가 충분하지 않은 경우 초기 프레임에서 발생하는 노이즈 방지
    if len(kpts) < 3 or any(kpt is None for kpt in kpts):
        print("Insufficient keypoints detected, skipping angle calculation.")
        return prev_angle  # 이전 각도를 유지하여 노이즈 최소화

    current_angle = annotator.estimate_pose_angle(*kpts)  # 키포인트를 사용하여 각도 추정
    angle_history.append(current_angle)  # 현재 각도를 각도 기록에 추가

    # 충분한 데이터가 쌓이기 전까지는 노이즈 감소를 위해 이전 각도 사용
    if len(angle_history) < 15:
        return prev_angle
    # 만약 충분한 데이터가 있을 경우.
    else :
        return smooth_angles(angle_list = angle_history)

def process_keypoints(kpts, keypoints):
    """
    사전 정의된 인덱스를 기반으로 특정 신체 부위의 키포인트를 추출합니다.
    :param keypoints: 이미지에서 감지된 키포인트
    :return: 목표 신체 부위의 키포인트 리스트
    """
    kpts = [keypoints[int(kpts[i])].cpu() for i in range(3)]  # 세 개의 관련된 포인트에 대한 키포인트 추출
    return kpts

def evaluate_range_of_motion(start_idx, end_idx, angle_history, max_threshold, min_threshold):
    """
    주어진 반복 구간에서 각도의 극대값과 극소값을 평가합니다.
    :param start_idx: 반복 구간의 시작 인덱스
    :param end_idx: 반복 구간의 종료 인덱스
    """
    angle_segment = angle_history[start_idx:end_idx]

    # 극대값과 극소값을 찾습니다.
    max_value = max(angle_segment)
    min_value = min(angle_segment)
    
    current_value = angle_segment[-1] # '미는 운동'에 한해서 사용한다. 당기는 운동일 경우 거꾸로.
    if current_value - min_value < 30 : 
        print('너무 조금 들어올렸거나, 올바르지 못한 탐지입니다.')
        return False

    # 최대값 및 최소값 평가
    if max_value < max_threshold:
        print(f"주의: 최대 가동 범위가 충분하지 않습니다. 최대 각도: {max_value:.2f}° (임계값: {max_threshold}°)")
        return False
    else : 
        print(f"올바른 최대 가동 범위입니다. 최대 각도: {max_value:.2f}°")

    if min_value > min_threshold:
        print(f"주의: 최소 가동 범위가 충분하지 않습니다. 최소 각도: {min_value:.2f}° (임계값: {min_threshold}°)")
        return False
    else : 
        print(f"올바른 최소 가동 범위입니다. 최소 각도: {min_value:.2f}°")
    
    if max_value >= max_threshold and min_value <= min_threshold :
        print("올바른 가동범위를 수행했습니다.")
        return True

def calculate_repetition_speed(current_frame, previous_frame):
    """
    수축과 이완을 통해 한 번의 반복이 완료되었을 때 운동 속도를 계산합니다.
    
    :param previous_time: 이전 반복의 타임스탬프
    :param speed_data: 시간 차이를 저장하는 리스트
    :return: 업데이트된 이전 타임스탬프
    """
    frame_diff = current_frame - previous_frame
    time_diff = frame_diff / 30 # 30FPS이므로 30으로 나눈다. 

    print(f"이번 반복의 속도: {time_diff:.2f}초")
    return time_diff

def get_video_index_by_threshold(threshold):
    if threshold < 10:
        return 0
    elif threshold < 20:
        return 1
    elif threshold < 30:
        return 2
    else:
        return 3