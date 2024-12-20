# CCP_project

# 운동 분석 및 피드백 시스템

이 프로젝트는 웹캠을 사용하여 운동을 분석하고, 피드백 영상을 제공하는 시스템입니다.

## 파일 설명

### gym.py
`CustomAIGym` 클래스를 정의합니다. 이 클래스는 운동 분석을 수행하고, 각도 계산, 운동 상태 업데이트, 이미지 주석 추가 등의 기능을 포함합니다.

### main.py
메인 실행 파일로, 웹캠 설정 및 `CustomAIGym` 인스턴스를 생성하고 모니터링을 수행합니다.

### utils.py
유틸리티 함수들을 포함합니다. 객체 추적 및 감지, 각도 계산, 키포인트 처리, 가동 범위 평가, 반복 속도 계산 등의 기능을 제공합니다.

## 설치 및 실행 방법

1. 필요한 라이브러리 설치:
    ```
    pip install opencv-python matplotlib torch ultralytics
    ```

2. 프로젝트 파일 다운로드 및 디렉토리 구조 설정:
    ```
    프로젝트 디렉토리/
    ├── gym.py
    ├── main.py
    ├── utils.py
    ├── video/
    │   ├── 1.mp4
    │   ├── 2.mp4
    │   ├── 3.mp4
    │   ├── 4.mp4
    │   └── 5.mp4
    └── incline bench press/
        └── DB_press_1.mp4
    ```

3. `main.py` 파일 실행:
    ```bash
    python main.py
    ```

## 주요 기능

### CustomAIGym 클래스 (gym.py)
- **__init__**: 초기화 메서드로, 추적 대상 객체 ID, 반복 횟수, 운동 상태, 각도 기록 등을 초기화합니다.
- **detect_slow_movement**: 최근 프레임 동안의 평균 각도 변화량을 검사하여 운동 속도가 매우 느린 상태인지 감지합니다.
- **update_stage**: 관절 각도 변화에 따라 운동의 현재 상태를 업데이트합니다.
- **annotate_image**: 이미지에 키포인트, 현재 각도, 반복 횟수 및 운동 상태를 주석으로 표시합니다.
- **monitor**: 추적 대상 감지, 키포인트 처리 및 이미지 주석을 수행하는 메인 모니터링 함수입니다.
- **plot_training_data**: 관절 각도 변화, 반복 횟수, 운동 속도 데이터를 통합적으로 시각화합니다.

### 유틸리티 함수 (utils.py)
- **track_and_detect**: 입력 이미지에서 객체 추적 및 감지를 수행합니다.
- **smooth_angles**: 적응형 이동 평균 필터를 사용하여 각도 기록 리스트의 노이즈를 완화합니다.
- **calculate_angle**: 주어진 키포인트를 사용하여 관절 각도를 계산하고 부드럽게 처리합니다.
- **process_keypoints**: 특정 신체 부위의 키포인트를 추출합니다.
- **evaluate_range_of_motion**: 주어진 반복 구간에서 각도의 극대값과 극소값을 평가합니다.
- **calculate_repetition_speed**: 수축과 이완을 통해 한 번의 반복이 완료되었을 때 운동 속도를 계산합니다.
- **get_video_index_by_threshold**: 주어진 임계값에 따라 비디오 인덱스를 반환합니다.

## 참고 사항
- `main.py`를 실행하기 전에 `gym.py`와 `utils.py`가 동일한 디렉토리에 있는지 확인하세요.
- 웹캠이 제대로 연결되어 있는지 확인하세요.
