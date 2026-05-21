import pygame
import numpy as np
import math
import gymnasium as gym
import cv2
from gymnasium import spaces


class LineTracingCameraEnv(gym.Env):

    def __init__(self, render_mode=False):
        super().__init__()
        pygame.init()
        self.low_speed_count = 0
        self.prev_steering = 0.0
        self.road_width = 90
        self.lane_line_width = 5
        self.lane_lost_count = 0
        
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

        self.available_tracks = [0]
        self.num_tracks = len(self.available_tracks)

        self.track_id = np.random.choice(self.available_tracks)

        self.track_points = self.build_bezier_track(self.track_id)

        all_start_points = self.generate_start_points_by_interval(interval=20)

        # track별로 제외할 start 지정
        bad_start_ids_by_track = {
            0:{1,2,3,7},
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

        self.track_points = self.build_bezier_track(self.track_id)
        self.left_lane_points, self.right_lane_points = self.build_lane_lines()

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
        self.max_turn_rate = 40.0  # degree per second

    # 초당 최대 이동 속도
        self.max_speed = 50.0  # pixel per second
        # =========================
        # 카메라 설정
        # =========================
        # 실제 라즈베리파이 카메라가 본다고 가정하는 영역 크기
        self.camera_width = 160
        self.camera_height = 120

        # 자동차 앞쪽 몇 px 떨어진 지점을 카메라 중심으로 볼 것인지
        self.camera_distance = 90

        # 카메라 관찰값을 몇 칸으로 나눌 것인지
        # 예: 16칸이면 화면 가로를 16등분해서 각 칸에 선이 있는지 확인
        self.observation_bins = 12
        self.observation_rows = 3

        # Gymnasium space 설정
        # =========================

        self.lane_scan_rows = 16

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.lane_scan_rows * 2 + 2,),
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
        self.max_steps = 4000
        self.target_steps = 500

        # 처음 화면 준비
        self.reset()

    def get_camera_image_with_angle_sign(self, sign):
        camera_x, camera_y = self.get_camera_center()

        screen_array = pygame.surfarray.array3d(self.screen)
        screen_array = np.transpose(screen_array, (1, 0, 2))

        angle = sign * self.car_angle

        M = cv2.getRotationMatrix2D(
            (camera_x, camera_y),
            angle,
            1.0
        )

        M[0, 2] += self.camera_width / 2 - camera_x
        M[1, 2] += self.camera_height / 2 - camera_y

        camera_image = cv2.warpAffine(
            screen_array,
            M,
            (self.camera_width, self.camera_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.WHITE
        )

        return camera_image


    def debug_camera_sign_score(self):
        img_minus = self.get_camera_image_with_angle_sign(-1)
        obs_minus = self.get_lane_observation_from_camera(img_minus)
        valid_minus = int(np.sum(obs_minus[1::2] > 0.01))

        img_plus = self.get_camera_image_with_angle_sign(1)
        obs_plus = self.get_lane_observation_from_camera(img_plus)
        valid_plus = int(np.sum(obs_plus[1::2] > 0.01))

        print(
            f"[Camera Debug] "
            f"angle=-car_angle ValidRows={valid_minus} | "
            f"angle=+car_angle ValidRows={valid_plus}"
        )

    def build_lane_lines(self):
        """
        track_points를 중심 경로로 보고,
        좌우 차선 점들을 생성한다.
        """
        left_points = []
        right_points = []

        n = len(self.track_points)
        half_width = self.road_width / 2

        for i in range(n):
            prev_p = self.track_points[i - 1]
            next_p = self.track_points[(i + 1) % n]

            dx = next_p[0] - prev_p[0]
            dy = next_p[1] - prev_p[1]

            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                left_points.append(self.track_points[i])
                right_points.append(self.track_points[i])
                continue

            # 진행 방향의 단위 벡터
            tx = dx / length
            ty = dy / length

            # 진행 방향에 수직인 법선 벡터
            nx = -ty
            ny = tx

            cx, cy = self.track_points[i]

            left_x = cx + nx * half_width
            left_y = cy + ny * half_width

            right_x = cx - nx * half_width
            right_y = cy - ny * half_width

            left_points.append((int(left_x), int(left_y)))
            right_points.append((int(right_x), int(right_y)))

        return left_points, right_points

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
        self.lane_lost_count = 0
        self.last_step_distance = 0.0

        all_start_points = self.generate_start_points_by_interval(interval=20)

        bad_start_ids_by_track = {
            0: {0,2,3,7},   # Track 1의 원래 Start 5 제외
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
        if self.track_id == 0 and 8 in self.start_original_ids and np.random.rand() < 0.3:
            self.start_id = self.start_original_ids.index(8)
        else:
            self.start_id = np.random.randint(len(self.start_points))

        self.start_original_id = self.start_original_ids[self.start_id]
        self.start_original_id = self.start_original_ids[self.start_id]

        self.left_lane_points, self.right_lane_points = self.build_lane_lines()

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
        self.last_lane_observation = observation

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
            lookahead = 5
            p2 = self.track_points[(i + lookahead) % n]

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
    # 배경
        self.screen.fill(self.WHITE)

        # 도로 영역: 어두운 회색으로 넓게 그림
        pygame.draw.lines(
            self.screen,
            (60, 60, 60),
            self.closed_track,
            self.track_points,
            self.road_width
        )

        # 왼쪽 차선
        pygame.draw.lines(
            self.screen,
            (255, 255, 0),
            self.closed_track,
            self.left_lane_points,
            self.lane_line_width
        )

        # 오른쪽 차선
        pygame.draw.lines(
            self.screen,
            (255, 255, 0),
            self.closed_track,
            self.right_lane_points,
            self.lane_line_width
        )

    def save_debug_camera_image(self, filename="debug_camera.png"):
        camera_image = self.get_camera_image()

        import cv2
        camera_bgr = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, camera_bgr)

        obs = self.get_lane_observation_from_camera(camera_image)
        valid_rows = int(np.sum(obs[1::2] > 0.01))

        print(f"[Camera Debug] saved={filename}, valid_rows={valid_rows}")

    def get_camera_center(self):
        rad = math.radians(self.car_angle)

        camera_x = self.car_x + math.cos(rad) * self.camera_distance
        camera_y = self.car_y + math.sin(rad) * self.camera_distance

        return camera_x, camera_y

    def get_camera_image(self):
        """
        차량 기준 전방 카메라 이미지 생성.
        화면 전체를 회전 crop하지 않고,
        차량의 진행 방향 기준으로 앞쪽 영역만 샘플링한다.
        """

        # pygame 화면 -> numpy [H, W, C]
        screen_array = pygame.surfarray.array3d(self.screen)
        screen_array = np.transpose(screen_array, (1, 0, 2))

        h = self.camera_height
        w = self.camera_width

        rad = math.radians(self.car_angle)

        # 차량 진행 방향 벡터
        forward_x = math.cos(rad)
        forward_y = math.sin(rad)

        # 차량 오른쪽 방향 벡터
        right_x = -math.sin(rad)
        right_y = math.cos(rad)

        # 카메라가 볼 범위
        near_dist = 20.0
        far_dist = 190.0
        view_width = 180.0

        # 이미지 좌표 만들기
        xs = np.linspace(-view_width / 2, view_width / 2, w)

        # 이미지 위쪽이 먼 곳, 아래쪽이 가까운 곳
        ys = np.linspace(far_dist, near_dist, h)

        local_x, local_y = np.meshgrid(xs, ys)

        # 차량 기준 local 좌표 -> 월드 좌표
        map_x = (
            self.car_x
            + forward_x * local_y
            + right_x * local_x
        ).astype(np.float32)

        map_y = (
            self.car_y
            + forward_y * local_y
            + right_y * local_x
        ).astype(np.float32)

        camera_image = cv2.remap(
            screen_array,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.WHITE
        )

        return camera_image

    def get_lane_observation_from_camera(self, image):
        """
        카메라 numpy 이미지에서 왼쪽/오른쪽 차선을 감지하고,
        각 scan line마다 차선 중앙 오차를 observation으로 만든다.
        """

        img = image  # 이미 [H, W, C] numpy array

        h, w, _ = img.shape
        camera_center_x = w / 2

        # 노란색 차선 감지
        lane_mask = (
            (img[:, :, 0] > 160) &
            (img[:, :, 1] > 130) &
            (img[:, :, 2] < 180)
        )

        scan_ys = np.linspace(
            int(h * 0.15),
            int(h * 0.95),
            self.lane_scan_rows
        ).astype(int)

        observations = []

        for y in scan_ys:
            xs = np.where(lane_mask[y])[0]

            if len(xs) < 2:
                observations.append(0.0)  # center_error
                observations.append(0.0)  # confidence
                continue

            left_candidates = xs[xs < camera_center_x]
            right_candidates = xs[xs > camera_center_x]

            if len(left_candidates) == 0 or len(right_candidates) == 0:
                observations.append(0.0)
                observations.append(0.0)
                continue

            # 왼쪽 차선은 왼쪽 후보들의 평균보다 max가 더 안정적일 때가 많음
            left_x = np.mean(left_candidates)
            right_x = np.mean(right_candidates)

            lane_center_x = (left_x + right_x) / 2
            lane_width_px = right_x - left_x

            center_error = (lane_center_x - camera_center_x) / (w / 2)

            # confidence는 차선 폭이 정상적으로 보이면 1에 가깝게
            lane_width_norm = lane_width_px / w
            confidence = np.clip(lane_width_norm, 0.0, 1.0)

            observations.append(center_error)
            observations.append(confidence)

        return np.array(observations, dtype=np.float32)

    def get_observation(self):
        camera_image = self.get_camera_image()
        lane_obs = self.get_lane_observation_from_camera(camera_image)

        extra_obs = np.array([
            self.prev_steering,
            self.current_speed / max(self.max_speed, 1e-6)
        ], dtype=np.float32)

        observation = np.concatenate([lane_obs, extra_obs]).astype(np.float32)

        return observation

    def is_off_road_by_lane_detection(self):
        lane_obs_len = self.lane_scan_rows * 2
        obs = self.last_lane_observation[:lane_obs_len]

        center_errors = obs[0::2]
        confidences = obs[1::2]

        valid = confidences > 0.01

        if np.sum(valid) == 0:
            self.lane_lost_count += 1
        else:
            self.lane_lost_count = 0

        # 차선을 15 step 연속 못 볼 때만 종료
        lost_limit = 60

        if self.track_id == 0 and self.start_original_id == 8:
            lost_limit = 80

        if self.lane_lost_count >= lost_limit:
            return True
        
        if np.any(valid):
            mean_abs_error = np.mean(np.abs(center_errors[valid]))
            if mean_abs_error > 1.2:
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

        self.last_step_distance = step_distance

        self.total_distance += step_distance
        self.prev_x = self.car_x
        self.prev_y = self.car_y

        # 화면 다시 그림
        self.screen.fill(self.WHITE)
        self.draw_track()

        # observation 계산
        observation = self.get_observation()
        self.last_lane_observation = observation

        # reward 계산
        reward = self.calculate_reward(steering, throttle)

        #lane 관련
        lane_obs_len = self.lane_scan_rows * 2
        lane_obs = self.last_lane_observation[:lane_obs_len]
        confidences = lane_obs[1::2]
        valid_count = int(np.sum(confidences > 0.01))

        # 다음 reward 계산을 위해 저장
        self.prev_steering = steering

        # 종료 조건
        terminated = False
        truncated = False
        done_reason = "none"

        #finish_line_x = 830.0

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
        elif self.is_off_road_by_lane_detection():
            terminated = True
            reward -= 25.0
            done_reason = "off_road"

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
            "valid_lane_rows": valid_count,
        }

        return observation, reward, terminated, truncated, info

    def calculate_reward(self, steering, throttle):
        lane_obs_len = self.lane_scan_rows * 2
        obs = self.last_lane_observation[:lane_obs_len]

        center_errors = obs[0::2]
        confidences = obs[1::2]

        valid = confidences > 0.01
        valid_count = int(np.sum(valid))
        valid_ratio = valid_count / len(confidences)

        # 차선을 하나도 못 보면 손해
        if valid_count == 0:
            reward = -0.8
            reward -= throttle * 0.2
            reward -= min(self.lane_lost_count * 0.03, 0.8)
            return reward

        valid_errors = center_errors[valid]
        mean_abs_error = np.mean(np.abs(valid_errors))

        center_score = 1.0 - np.clip(mean_abs_error, 0.0, 1.0)

        reward = 0.0

        # 1. 차선을 많이 볼수록 보상
        reward += valid_ratio * 1.0

        # 2. 차선 중앙에 가까울수록 보상
        reward += center_score * 0.8

        # 3. 차선을 어느 정도 보고 있을 때만 속도 보상
        if valid_count >= 3 and mean_abs_error < 0.5:
            reward += throttle * 0.25
        else:
            reward -= throttle * 0.15

        # 4. 과도한 조향만 살짝 패널티
        reward -= abs(steering) * 0.08

        # 5. 조향 변화 패널티도 약하게
        steer_change = abs(steering - self.prev_steering)
        reward -= steer_change * 0.05

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

        pygame.draw.rect(self.screen, self.GREEN, camera_rect, 2)
        pygame.draw.circle(self.screen, self.YELLOW, (int(camera_x), int(camera_y)), 4)

        if self.render_mode:
            pygame.display.update()

    def close(self):
        pygame.quit()