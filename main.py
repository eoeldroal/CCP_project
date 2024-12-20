import cv2
from gym import CustomAIGym
from utils import get_video_index_by_threshold

video_paths = [
    "video/1.mp4",
    "video/2.mp4",
    "video/3.mp4",
    "video/4.mp4",
    "video/5.mp4"
]

input_video = "incline bench press/DB_press_1.mp4"

output_path = input_video + "annotated.mp4"

gym = CustomAIGym(
    line_width=2,
    show=True,
    model="yolo11m-pose.pt",
    tracking_id=1,
    max_angle=120,
    min_angle=90,
    workout_type = "press",
)

# 웹캠 설정
webcam_index = 0
cap = cv2.VideoCapture(webcam_index)
assert cap.isOpened(), "웹캠을 열 수 없습니다."

# 출력 파일 경로
output_path = "output_video.mp4"

# 웹캠 영상 속성 가져오기
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)
if original_fps == 0 or original_fps is None:
    original_fps = 15.0  # FPS 정보를 제대로 얻지 못하면 임의로 15FPS 사용

fps = 15  # 원하는 FPS로 처리 가능

# 비디오 작성기 설정 (분석 결과를 저장)
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w // 2, h // 2))

# 초기 video_index 설정
current_video_index = get_video_index_by_threshold(gym.threshold)
cap2 = cv2.VideoCapture(video_paths[current_video_index])
if not cap2.isOpened():
    print(f"Error opening secondary video: {video_paths[current_video_index]}")
    
# cv2.namedWindow("Feedback Video", cv2.WINDOW_NORMAL)  # 윈도우 크기 변경 가능
# cv2.setWindowProperty("Feedback Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

n_tries = 0
while True:
    # 웹캠에서 프레임 읽기
    success, im0 = cap.read()
    if not success:
        print("웹캠 프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # 프레임 크기 조정 (분석 용도)
    im0 = cv2.resize(im0, (w, h))

    # 운동 분석 (monitor 함수 호출)
    im0, new_video_index = gym.monitor(im0)

    # video_index가 바뀌었는지 확인하여 피드백 영상 전환
    if new_video_index != current_video_index:
        # 이전 피드백 영상 캡쳐 객체 해제
        if cap2.isOpened():
            cap2.release()
        current_video_index = new_video_index
        new_video_path = video_paths[current_video_index]
        cap2 = cv2.VideoCapture(new_video_path)
        if not cap2.isOpened():
            print(f"Error: Could not open {new_video_path}")
        else:
            print(f"Switched to video: {new_video_path}")

    # 현재 video_index에 해당하는 피드백 영상에서 프레임 읽기
    ret2, frame2 = cap2.read()
    if not ret2:
        # 피드백 영상이 끝났다면 다시 처음부터 재생
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, frame2 = cap2.read()
        if not ret2:
            print("피드백 영상 재생 불가.")
            break

    # 피드백 영상 표시
    cv2.imshow("Feedback Video", frame2)

    # 분석 결과 영상(운동 분석 영상) 저장
    video_writer.write(im0)
    # cv2.imshow("Exercise Analysis", im0)

    n_tries += 1

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap2.release()
video_writer.release()
cv2.destroyAllWindows()

print("비디오 저장이 완료되었습니다.")

gym.plot_training_data()