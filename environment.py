import pygame
import numpy as np
import math
import cv2
import gymnasium as gym
from gymnasium import spaces


class LineTracingCameraEnv(gym.Env):

    def __init__(self, render_mode=False):
        super().__init__()
        pygame.init()
        self.low_speed_count = 0
        self.prev_steering = 0.0
                # =========================
        
        # =========================
        # 화면 설정
        # =========================
        self.width = 900
        self.height = 600
        self.render_mode = render_mode

        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Line Tracing Camera Environment")
        else:
            self.screen = pygame.Surface((self.width, self.height))

        self.clock = pygame.time.Clock()

        # =========================
        # 색상 설정
        # =========================
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (50, 120, 255)
        self.RED = (255, 60, 60)
        self.GREEN = (0, 200, 0)
        self.GRAY = (180, 180, 180)
        self.YELLOW = (255, 220, 0)

        # =========================
        # 트랙 / 자동차 시작점 설정
        # =========================
        self.closed_track = True

        self.available_tracks = [0, 1, 2, 3]
        self.num_tracks = len(self.available_tracks)

        self.track_id = np.random.choice(self.available_tracks)

        self.track_points = self.build_bezier_track(self.track_id)

        all_start_points = self.generate_start_points_by_interval(interval=20)

        # track별로 제외할 start 지정
        bad_start_ids_by_track = {
            1: {5},   # Track 1의 Start 5 제외
        }

        bad_start_ids = bad_start_ids_by_track.get(self.track_id, set())

        self.start_points = [
            point for i, point in enumerate(all_start_points)
            if i not in bad_start_ids
        ]

        self.start_original_ids = [
            i for i, point in enumerate(all_start_points)
            if i not in bad_start_ids
        ]

        # 현재 속도
        self.current_speed = 0.0

        # =========================
    # 시간 / 제어 주기 설정
    # =========================
        self.dt = 0.1  # 10Hz 기준, 한 step = 0.1초

    # =========================
    # action 관련 설정
    # =========================
    # steering: -1.0 ~ 1.0
    # throttle: -1.0 ~ 1.0

    # 초당 최대 회전 각도
        self.max_turn_rate = 60.0  # degree per second

    # 초당 최대 이동 속도
        self.max_speed = 60.0  # pixel per second
        # =========================
        # 카메라 설정
        # =========================
        # 실제 라즈베리파이 카메라가 본다고 가정하는 영역 크기
        self.camera_width = 160
        self.camera_height = 120

        # 자동차 앞쪽 몇 px 떨어진 지점을 카메라 중심으로 볼 것인지
        self.camera_distance = 60

        # 카메라 관찰값을 몇 칸으로 나눌 것인지
        # 예: 16칸이면 화면 가로를 16등분해서 각 칸에 선이 있는지 확인
        self.observation_bins = 16
        self.observation_rows = 3

        # Gymnasium space 설정
        # =========================
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_bins * self.observation_rows + self.observation_rows + 2,),
            dtype=np.float32
        )

        # action[0] = steering, action[1] = throttle
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # =========================
        # 에피소드 설정
        # =========================
        self.step_count = 0
        self.max_steps = 2000
        self.target_steps = 500

        # 처음 화면 준비
        self.reset()

    def cubic_bezier(self, p0, p1, p2, p3, num_points=30):
        """
        4개의 control point로 cubic Bezier 곡선 점들을 생성
        """
        points = []

        for i in range(num_points):
            t = i / (num_points - 1)

            x = (
                (1 - t) ** 3 * p0[0]
                + 3 * (1 - t) ** 2 * t * p1[0]
                + 3 * (1 - t) * t ** 2 * p2[0]
                + t ** 3 * p3[0]
            )

            y = (
                (1 - t) ** 3 * p0[1]
                + 3 * (1 - t) ** 2 * t * p1[1]
                + 3 * (1 - t) * t ** 2 * p2[1]
                + t ** 3 * p3[1]
            )

            points.append((int(x), int(y)))

        return points


    def build_bezier_track(self, track_id=0):
        """
        track_id에 따라 서로 다른 베지어 루프 트랙 생성
        """

        track_segments = [
            # =========================
            # Track 0: 현재 성공한 기본 루프
            # =========================
            [
                ((240, 300), (270, 230), (360, 220), (430, 235)),
                ((430, 235), (520, 250), (610, 270), (620, 330)),
                ((620, 330), (610, 390), (500, 390), (430, 370)),
                ((430, 370), (350, 350), (300, 390), (250, 360)),
                ((250, 360), (190, 330), (200, 290), (240, 300)),
            ],

            # =========================
            # Track 1: 조금 더 넓은 루프
            # =========================
            [
                ((230, 310), (260, 230), (370, 210), (450, 235)),
                ((450, 235), (550, 260), (650, 280), (650, 340)),
                ((650, 340), (620, 410), (500, 400), (420, 370)),
                ((420, 370), (330, 340), (280, 400), (230, 360)),
                ((230, 360), (170, 320), (190, 270), (230, 310)),
            ],

            # =========================
            # Track 2: 아래쪽 굴곡이 조금 다른 루프
            # =========================
            [
                ((250, 290), (300, 220), (390, 230), (460, 250)),
                ((460, 250), (560, 280), (620, 250), (640, 320)),
                ((640, 320), (650, 390), (520, 390), (440, 360)),
                ((440, 360), (360, 330), (300, 370), (240, 350)),
                ((240, 350), (180, 320), (200, 270), (250, 290)),
            ],

            # =========================
            # Track 3: 좌우 변화가 조금 더 있는 루프
            # =========================
            [
                ((240, 310), (280, 240), (360, 220), (430, 250)),
                ((430, 250), (500, 290), (590, 230), (630, 310)),
                ((630, 310), (670, 390), (520, 400), (440, 370)),
                ((440, 370), (350, 340), (310, 410), (250, 360)),
                ((250, 360), (190, 320), (200, 280), (240, 310)),
            ],
        ]

        segments = track_segments[track_id]

        track_points = []

        for idx, segment in enumerate(segments):
            bezier_points = self.cubic_bezier(*segment, num_points=35)

            if idx > 0:
                bezier_points = bezier_points[1:]

            track_points.extend(bezier_points)

        return track_points

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 매 에피소드마다 트랙 랜덤 선택
        self.track_id = np.random.choice(self.available_tracks)
        self.track_points = self.build_bezier_track(self.track_id)

        all_start_points = self.generate_start_points_by_interval(interval=20)

        bad_start_ids_by_track = {
            1: {5},   # Track 1의 원래 Start 5 제외
        }

        bad_start_ids = bad_start_ids_by_track.get(self.track_id, set())

        self.start_points = [
            point for i, point in enumerate(all_start_points)
            if i not in bad_start_ids
        ]

        self.start_original_ids = [
            i for i, point in enumerate(all_start_points)
            if i not in bad_start_ids
        ]    

        self.prev_steering = 0.0
        self.low_speed_count = 0
        self.current_speed = 0.0
        self.step_count = 0

        # 시작점 하나만 랜덤 선택
        self.start_id = np.random.randint(len(self.start_points))
        self.start_original_id = self.start_original_ids[self.start_id]

        start_x, start_y, start_angle = self.start_points[self.start_id]

        self.car_x = start_x
        self.car_y = start_y
        self.car_angle = start_angle

        # 시작 위치 / 이동거리 기록
        self.start_x = self.car_x
        self.start_y = self.car_y

        self.total_distance = 0.0
        self.prev_x = self.car_x
        self.prev_y = self.car_y

        self.screen.fill(self.WHITE)
        self.draw_track()

        observation = self.get_observation()
        info = {
            "start_id": self.start_id,
            "start_original_id": self.start_original_id,
            "track_id": self.track_id,
        }

        return observation, info

    def get_track_segments(self):
        """
        track_points를 이용해 선분 리스트를 만든다.
        closed_track=True면 마지막 점과 첫 점도 연결한다.
        """
        segments = []

        for i in range(len(self.track_points) - 1):
            p1 = self.track_points[i]
            p2 = self.track_points[i + 1]
            segments.append((p1, p2))

        if self.closed_track:
            segments.append((self.track_points[-1], self.track_points[0]))

        return segments
    
    def generate_start_points_by_interval(self, interval=20):
        """
        부드러운 track_points에서 일정 간격마다 시작점을 생성.
        angle은 다음 점 방향으로 자동 계산.
        """
        start_points = []

        n = len(self.track_points)

        for i in range(0, n, interval):
            p1 = self.track_points[i]
            p2 = self.track_points[(i + 1) % n]

            x1, y1 = p1
            x2, y2 = p2

            dx = x2 - x1
            dy = y2 - y1

            angle = math.degrees(math.atan2(dy, dx))

            start_points.append((float(x1), float(y1), float(angle)))

        return start_points


    def generate_start_points(self, samples_per_segment=1):
        start_points = []

        segments = self.get_track_segments()

        for p1, p2 in segments:
            x1, y1 = p1
            x2, y2 = p2

            dx = x2 - x1
            dy = y2 - y1

            angle = math.degrees(math.atan2(dy, dx))

            if samples_per_segment == 1:
                t_values = [0.35]
            else:
                t_values = [
                    (s + 1) / (samples_per_segment + 1)
                    for s in range(samples_per_segment)
                ]

            for t in t_values:
                x = x1 + dx * t
                y = y1 + dy * t

                start_points.append((float(x), float(y), float(angle)))

        return start_points

    def draw_track(self):
        pygame.draw.lines(
            self.screen,
            self.BLACK,
            self.closed_track,
            self.track_points,
            35
        )

    def get_camera_center(self):
        """
        자동차 앞쪽에 있는 가상의 카메라 중심 좌표를 계산
        """
        rad = math.radians(self.car_angle)

        camera_x = self.car_x + math.cos(rad) * self.camera_distance
        camera_y = self.car_y + math.sin(rad) * self.camera_distance

        return camera_x, camera_y

    def get_camera_image(self):
        """
        자동차 앞쪽 카메라가 보는 이미지를 가져옴.

        실제 라즈베리파이에서는 여기 부분이
        picamera2 또는 cv2.VideoCapture로 바뀜.

        지금은 pygame 화면에서 자동차 앞쪽 영역을 잘라서
        카메라 이미지처럼 사용함.
        """

        camera_x, camera_y = self.get_camera_center()

        left = int(camera_x - self.camera_width / 2)
        top = int(camera_y - self.camera_height / 2)

        # 화면 범위 밖으로 나가지 않도록 제한
        left = max(0, min(left, self.width - self.camera_width))
        top = max(0, min(top, self.height - self.camera_height))

        # pygame Surface에서 카메라 영역 잘라내기
        camera_rect = pygame.Rect(
            left,
            top,
            self.camera_width,
            self.camera_height
        )

        camera_surface = self.screen.subsurface(camera_rect).copy()

        # pygame Surface -> numpy array
        camera_array = pygame.surfarray.array3d(camera_surface)

        # pygame은 (width, height, color) 형태라서
        # OpenCV에서 쓰기 좋게 (height, width, color)로 바꿈
        camera_array = np.transpose(camera_array, (1, 0, 2))

        return camera_array

    def preprocess_camera_image(self, image):
        """
        카메라 이미지에서 검은 선만 추출함.

        실제 카메라에서도 비슷하게 처리함:
        1. RGB 이미지를 grayscale로 변환
        2. 검은색 라인만 threshold로 분리
        """

        # RGB -> Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 검은색 선 검출
        # gray 값이 80보다 작으면 검은 선이라고 판단
        _, binary = cv2.threshold(
            gray,
            80,
            255,
            cv2.THRESH_BINARY_INV
        )

        return binary

    def get_observation(self):
        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)

        h, w = binary.shape

        bin_width = w // self.observation_bins
        row_height = h // self.observation_rows

        observation = []
        row_centers = []

        for row in range(self.observation_rows):
            start_y = row * row_height

            if row == self.observation_rows - 1:
                end_y = h
            else:
                end_y = (row + 1) * row_height

            row_binary = binary[start_y:end_y, :]

            # =========================
            # 1. row별 선 중심 위치 추가
            # =========================
            ys, xs = np.where(row_binary > 0)

            if len(xs) == 0:
                # 해당 row에서 선을 못 찾으면 0으로 둠
                # 0은 중앙이라는 의미라 완벽하진 않지만,
                # 일단 학습 안정성을 위해 큰 이상값은 피함
                row_center = 0.0
            else:
                line_center_x = np.mean(xs)

                # -1에 가까움: 왼쪽
                #  0에 가까움: 중앙
                #  1에 가까움: 오른쪽
                row_center = (line_center_x - w / 2) / (w / 2)

            row_centers.append(row_center)

            # =========================
            # 2. 기존 bin별 검은 픽셀 비율
            # =========================
            for i in range(self.observation_bins):
                start_x = i * bin_width

                if i == self.observation_bins - 1:
                    end_x = w
                else:
                    end_x = (i + 1) * bin_width

                roi = binary[start_y:end_y, start_x:end_x]

                black_ratio = np.sum(roi > 0) / roi.size
                observation.append(black_ratio)

        # row별 중심 위치 3개 추가
        observation.extend(row_centers)

        angle_normalized = ((self.car_angle + 180) % 360 - 180) / 180.0
        speed_norm = self.current_speed / self.max_speed

        observation.append(angle_normalized)
        observation.append(speed_norm)

        return np.array(observation, dtype=np.float32)

    def calculate_line_position(self):
        """
        카메라 이미지 안에서 선이 어느 쪽에 있는지 계산.

        -1에 가까움: 왼쪽에 선이 있음
         0에 가까움: 중앙에 선이 있음
         1에 가까움: 오른쪽에 선이 있음

        이 값은 사람이 직접 제어에 쓰는 게 아니라,
        reward 계산용으로만 사용함.
        """

        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)

        h, w = binary.shape

        ys, xs = np.where(binary > 0)

        # 선을 못 찾은 경우
        if len(xs) == 0:
            return None

        line_center_x = np.mean(xs)

        # 화면 중앙 기준으로 정규화
        normalized_position = (line_center_x - w / 2) / (w / 2)

        return normalized_position

    def is_off_track(self):
        """
        카메라에서 검은 선이 거의 안 보이면 트랙을 놓쳤다고 판단
        """

        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)

        black_ratio = np.sum(binary > 0) / binary.size

        if black_ratio < 0.03:
            return True

        return False

    def step(self, action):
        """
        action을 받아서 자동차를 움직임.

        action[0] = steering       # -1.0 ~ 1.0
        action[1] = raw_throttle   # -1.0 ~ 1.0

        실제 throttle은 0.0 ~ 1.0으로 변환해서 사용
        """

        self.step_count += 1

        steering = float(action[0])
        raw_throttle = float(action[1])

        steering = np.clip(steering, -1.0, 1.0)
        raw_throttle = np.clip(raw_throttle, -1.0, 1.0)

        steering = 0.5 * self.prev_steering + 0.5 * steering

        # PPO는 -1~1로 출력하지만, 실제 throttle은 0~1로 변환
        throttle = (raw_throttle + 1.0) / 2.0

        if throttle < 0.1:
            self.low_speed_count += 1
        else:
            self.low_speed_count = 0

        # steering에 따라 자동차 각도 변경
        self.car_angle += steering * self.max_turn_rate * self.dt

        # throttle에 따라 현재 속도 결정
        self.current_speed = throttle * self.max_speed

        # 자동차 이동
        rad = math.radians(self.car_angle)

        self.car_x += math.cos(rad) * self.current_speed * self.dt
        self.car_y += math.sin(rad) * self.current_speed * self.dt

        step_distance = math.sqrt(
            (self.car_x - self.prev_x) ** 2 +
            (self.car_y - self.prev_y) ** 2
        )

        self.total_distance += step_distance
        self.prev_x = self.car_x
        self.prev_y = self.car_y

        # 화면 다시 그림
        self.screen.fill(self.WHITE)
        self.draw_track()

        # observation 계산
        observation = self.get_observation()

        # reward 계산
        reward = self.calculate_reward(steering, throttle)

        # 다음 reward 계산을 위해 저장
        self.prev_steering = steering

        # 종료 조건
        terminated = False
        truncated = False
        done_reason = "none"

        finish_line_x = 830.0

        # 1. finish는 off_track보다 먼저 검사
        #if self.car_x >= finish_line_x:
        #    terminated = True
        #    reward += 20.0
        #    done_reason = "finish"
        
        # 2. 저속 종료
        if self.low_speed_count >= 30:
            terminated = True
            reward -= 5.0
            done_reason = "low_speed"

        # 3. 선 이탈 종료
        elif self.is_off_track():
            terminated = True

            progress_ratio = min(self.step_count / self.target_steps, 1.0)
            early_fail_penalty = 3.0 * (1.0 - progress_ratio)

            reward -= 6.0
            reward -= early_fail_penalty
            done_reason = "off_track"

        # 4. 화면 밖 종료
        elif self.car_x < 0 or self.car_x > self.width:
            terminated = True
            reward -= 6.0
            done_reason = "x_out"

        elif self.car_y < 0 or self.car_y > self.height:
            terminated = True
            reward -= 6.0
            done_reason = "y_out"

        # 5. 최대 step 도달
        if self.step_count >= self.max_steps:
            truncated = True
            done_reason = "max_steps"
        
        distance_from_start = math.sqrt(
            (self.car_x - self.start_x) ** 2 + 
            (self.car_y - self.start_y) ** 2
        )

        info = {
            "step_count": self.step_count,
            "steering": steering,
            "throttle": throttle,
            "car_x": self.car_x,
            "car_y": self.car_y,
            "car_angle": self.car_angle,
            "start_id": self.start_id,
            "done_reason": done_reason,
            "distance_from_start": distance_from_start,
            "total_distance": self.total_distance,
            "start_original_id": self.start_original_id,
            "track_id": self.track_id,
        }

        return observation, reward, terminated, truncated, info

    def calculate_reward(self, steering, throttle):
        line_position = self.calculate_line_position()

        if line_position is None:
            return -2.0

        center_error = abs(line_position)
        center_reward = 1.0 - center_error

        reward = 0.0

         # 선이 중앙에서 많이 벗어난 상태에서 너무 빠르면 벌점
        if center_error > 0.35 and throttle > 0.6:
            reward -= (throttle - 0.6) * 1.5

        # 선이 화면 끝쪽이면 더 강하게 속도 벌점
        if center_error > 0.6 and throttle > 0.5:
            reward -= (throttle - 0.5) * 2.0

        # 많이 꺾는 중인데 속도가 높으면 벌점
        if abs(steering) > 0.3 and throttle > 0.65:
            reward -= (throttle - 0.65) * abs(steering) * 2.0

        # 1. 핵심: 선 중앙 + 전진
        target_throttle = 0.55 + 0.25 * center_reward
        speed_score = 1.0 - abs(throttle - target_throttle)

        reward += center_reward * max(speed_score, 0.0) * 1.2

        # 2. 너무 느린 행동 방지
        if throttle < 0.1:
            reward -= 0.8
        elif throttle < 0.25:
            reward -= 0.2

        # 3. 선이 너무 가장자리면 명확한 벌점
        if center_error > 0.6:
            reward -= 0.8

        # 4. 조향 벌점은 약하게
        reward -= abs(steering) * 0.04

        steering_change = abs(steering - self.prev_steering)
        reward -= steering_change * 0.02

        # 5. 생존 보상은 작게, throttle과 묶기
        #reward += 0.03 * throttle

        return reward

    def render(self):
        """
        화면에 자동차, 방향, 가상 카메라 영역을 그림
        """

        # 화면은 step에서 이미 한 번 그렸지만,
        # render를 호출할 때 시각화용 요소를 추가로 그림

        # 자동차 그리기
        pygame.draw.circle(
            self.screen,
            self.BLUE,
            (int(self.car_x), int(self.car_y)),
            13
        )

        # 자동차 방향 표시
        rad = math.radians(self.car_angle)
        front_x = self.car_x + math.cos(rad) * 25
        front_y = self.car_y + math.sin(rad) * 25

        pygame.draw.line(
            self.screen,
            self.RED,
            (int(self.car_x), int(self.car_y)),
            (int(front_x), int(front_y)),
            4
        )

        # 가상 카메라 영역 표시
        camera_x, camera_y = self.get_camera_center()

        left = int(camera_x - self.camera_width / 2)
        top = int(camera_y - self.camera_height / 2)

        left = max(0, min(left, self.width - self.camera_width))
        top = max(0, min(top, self.height - self.camera_height))

        camera_rect = pygame.Rect(
            left,
            top,
            self.camera_width,
            self.camera_height
        )

        pygame.draw.rect(
            self.screen,
            self.GREEN,
            camera_rect,
            2
        )

        # 카메라 중심점 표시
        pygame.draw.circle(
            self.screen,
            self.YELLOW,
            (int(camera_x), int(camera_y)),
            5
        )
        if(self.render_mode):
            pygame.display.update()

    def close(self):
        pygame.quit()