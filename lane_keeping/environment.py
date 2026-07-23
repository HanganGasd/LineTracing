import pygame
import numpy as np
import math
import gymnasium as gym
import cv2
from gymnasium import spaces


class LaneKeepingEnv(gym.Env):

    def __init__(self, render_mode=True):
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
        self.max_turn_rate = 60.0  # degree per second

    # 초당 최대 이동 속도
        self.max_speed = 60.0  # pixel per second
        # =========================
        # 카메라 설정
        # =========================
        # 실제 라즈베리파이 카메라가 본다고 가정하는 영역 크기
        self.camera_width = 160
        self.camera_height = 120
        self.camera_near_dist = 20.0
        self.camera_far_dist = 190.0
        self.camera_view_width = 180.0
        camera_xs = np.linspace(
            -self.camera_view_width / 2.0,
            self.camera_view_width / 2.0,
            self.camera_width,
        )
        camera_ys = np.linspace(
            self.camera_far_dist,
            self.camera_near_dist,
            self.camera_height,
        )
        self._camera_local_x, self._camera_local_y = np.meshgrid(
            camera_xs,
            camera_ys,
        )

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

                # =========================
        # 진행도 / 완주 계산용
        # =========================
        self.track_cumulative_lengths = None
        self.track_length = 1.0
        self.prev_track_progress = 0.0
        self.start_progress = 0.0
        self.lap_progress = 0.0
        self.last_progress_delta = 0.0
        self.dist_to_center = 0.0

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
        학습 안정성을 위한 부드러운 폐곡선 트랙.
        기존 Bezier 조합 트랙은 급커브/자기교차 때문에 차선이 꼬이기 쉬움.
        """

        points = []

        cx, cy = 450, 310
        rx, ry = 260, 135

        num_points = 240

        for i in range(num_points):
            t = 2.0 * math.pi * i / num_points

            # 기본 타원 + 약한 굴곡
            x = cx + rx * math.cos(t)
            y = cy + ry * math.sin(t)

            # 너무 단순한 원형이 되지 않도록 약한 변형
            x += 35 * math.sin(2 * t)
            y += 20 * math.sin(3 * t)

            points.append((int(x), int(y)))

        return points

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

        # 진행도 계산 초기화
        self.build_track_progress_table()
        self.start_progress, self.dist_to_center = self.get_progress_on_track(
            self.car_x,
            self.car_y
        )
        self.prev_track_progress = self.start_progress
        self.lap_progress = 0.0
        self.last_progress_delta = 0.0

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
    
    def build_track_progress_table(self):
        """
        track_points를 따라 누적 거리 테이블을 만든다.
        닫힌 루프라서 마지막 점 -> 첫 점 구간도 포함한다.
        """
        points = self.track_points
        cumulative = [0.0]
        total = 0.0

        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_len = math.sqrt(dx * dx + dy * dy)

            total += seg_len
            cumulative.append(total)

            if not self.closed_track and i == len(points) - 2:
                break

        self.track_cumulative_lengths = cumulative
        self.track_length = max(total, 1e-6)

    def get_progress_on_track(self, x, y):
        """
        현재 자동차 위치를 track 중심선에 투영해서
        1) track을 따라간 진행 거리 progress
        2) 중심선으로부터 거리 dist
        를 반환한다.
        """
        if self.track_cumulative_lengths is None:
            self.build_track_progress_table()

        best_dist_sq = float("inf")
        best_progress = 0.0

        points = self.track_points
        n = len(points)

        segment_count = n if self.closed_track else n - 1

        for i in range(segment_count):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]

            vx = x2 - x1
            vy = y2 - y1

            wx = x - x1
            wy = y - y1

            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq < 1e-9:
                continue

            t = (wx * vx + wy * vy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)

            proj_x = x1 + t * vx
            proj_y = y1 + t * vy

            dx = x - proj_x
            dy = y - proj_y
            dist_sq = dx * dx + dy * dy

            if dist_sq < best_dist_sq:
                seg_len = math.sqrt(seg_len_sq)
                best_dist_sq = dist_sq
                best_progress = self.track_cumulative_lengths[i] + t * seg_len

        return best_progress, math.sqrt(best_dist_sq)

    def get_progress_delta(self, new_progress):
        """
        닫힌 트랙에서 진행도 차이를 계산한다.
        마지막 지점에서 시작 지점으로 넘어가는 wrap-around도 처리한다.
        """
        delta = new_progress - self.prev_track_progress

        # 한 바퀴 경계 넘어간 경우 보정
        if delta < -0.5 * self.track_length:
            delta += self.track_length
        elif delta > 0.5 * self.track_length:
            delta -= self.track_length

        # 이상치 방지
        delta = float(np.clip(delta, -10.0, 10.0))

        return delta

    def draw_track(self):
        # 배경
        self.screen.fill(self.WHITE)

        # 도로 본체
        pygame.draw.lines(
            self.screen,
            (55, 55, 55),
            self.closed_track,
            self.track_points,
            self.road_width
        )

        # 도로 중앙선 보조선: 디버깅용
        # 필요 없으면 이 부분 주석 처리해도 됨
        pygame.draw.lines(
            self.screen,
            (80, 80, 80),
            self.closed_track,
            self.track_points,
            2
        )

        # 왼쪽 차선
        pygame.draw.lines(
            self.screen,
            self.YELLOW,
            self.closed_track,
            self.left_lane_points,
            self.lane_line_width
        )

        # 오른쪽 차선
        pygame.draw.lines(
            self.screen,
            self.YELLOW,
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

        rad = math.radians(self.car_angle)

        # 차량 진행 방향 벡터
        forward_x = math.cos(rad)
        forward_y = math.sin(rad)

        # 차량 오른쪽 방향 벡터
        right_x = -math.sin(rad)
        right_y = math.cos(rad)

        # 차량 기준 local 좌표 -> 월드 좌표
        map_x = (
            self.car_x
            + forward_x * self._camera_local_y
            + right_x * self._camera_local_x
        ).astype(np.float32)

        map_y = (
            self.car_y
            + forward_y * self._camera_local_y
            + right_y * self._camera_local_x
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
        카메라 numpy 이미지에서 차선을 감지한다.

        개선점:
        1. 양쪽 차선이 다 보이면 안쪽 경계 기준으로 중앙 계산
        2. 한쪽 차선만 보여도 예상 차선폭으로 중앙 추정
        3. confidence를 너무 약하게 주지 않도록 보정
        """

        img = image

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

        # 카메라 view_width가 get_camera_image()에서 180.0이므로,
        # road_width 90px는 카메라 이미지 기준 대략 절반 정도
        expected_lane_width_px = w * (self.road_width / 180.0)
        expected_lane_width_px = np.clip(expected_lane_width_px, w * 0.25, w * 0.85)

        for y in scan_ys:
            xs = np.where(lane_mask[y])[0]

            if len(xs) == 0:
                observations.append(0.0)
                observations.append(0.0)
                continue

            left_candidates = xs[xs < camera_center_x]
            right_candidates = xs[xs > camera_center_x]

            has_left = len(left_candidates) > 0
            has_right = len(right_candidates) > 0

            if has_left and has_right:
                # 왼쪽 차선의 안쪽 경계, 오른쪽 차선의 안쪽 경계
                left_x = np.max(left_candidates)
                right_x = np.min(right_candidates)
                confidence = 1.0

            elif has_left:
                # 왼쪽 차선만 보이면 오른쪽 차선을 예상 폭으로 추정
                left_x = np.max(left_candidates)
                right_x = left_x + expected_lane_width_px
                confidence = 0.45

            elif has_right:
                # 오른쪽 차선만 보이면 왼쪽 차선을 예상 폭으로 추정
                right_x = np.min(right_candidates)
                left_x = right_x - expected_lane_width_px
                confidence = 0.45

            else:
                observations.append(0.0)
                observations.append(0.0)
                continue

            lane_center_x = (left_x + right_x) / 2.0
            lane_width_px = right_x - left_x

            center_error = (lane_center_x - camera_center_x) / (w / 2.0)
            center_error = float(np.clip(center_error, -1.0, 1.0))

            # 차선 폭이 너무 이상하면 confidence 낮춤
            width_ratio = lane_width_px / max(expected_lane_width_px, 1e-6)
            width_score = 1.0 - abs(width_ratio - 1.0)
            width_score = float(np.clip(width_score, 0.2, 1.0))

            confidence = float(np.clip(confidence * width_score, 0.0, 1.0))

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

        valid = confidences > 0.05

        if np.sum(valid) == 0:
            self.lane_lost_count += 1
        else:
            self.lane_lost_count = 0

        # 10Hz 기준 2.5초 정도 차선이 안 보이면 종료
        lost_limit = 25

        if self.lane_lost_count >= lost_limit:
            return True

        if np.any(valid):
            mean_abs_error = np.mean(np.abs(center_errors[valid]))

            # 0.95는 너무 빨리 죽을 수 있어서 1.05로 완화
            if mean_abs_error > 1.05:
                return True

        return False
    

    def step(self, action):
        """
        action[0] = steering       # -1.0 ~ 1.0
        action[1] = raw_throttle   # -1.0 ~ 1.0

        raw_throttle은 내부에서 0.0 ~ 1.0으로 변환한다.
        """

        self.step_count += 1

        steering = float(action[0])
        raw_throttle = float(action[1])

        steering = np.clip(steering, -1.0, 1.0)
        raw_throttle = np.clip(raw_throttle, -1.0, 1.0)

        # 조향 smoothing
        steering = 0.5 * self.prev_steering + 0.5 * steering

        # PPO 출력 -1~1을 실제 throttle 0~1로 변환
        throttle = (raw_throttle + 1.0) / 2.0
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # 저속 판정 강화
        if throttle < 0.25:
            self.low_speed_count += 1
        else:
            self.low_speed_count = 0

        # 자동차 각도 변경
        self.car_angle += steering * self.max_turn_rate * self.dt

        # 속도 계산
        self.current_speed = throttle * self.max_speed

        # 이동 전 위치
        old_x = self.car_x
        old_y = self.car_y

        # 자동차 이동
        rad = math.radians(self.car_angle)

        self.car_x += math.cos(rad) * self.current_speed * self.dt
        self.car_y += math.sin(rad) * self.current_speed * self.dt

        step_distance = math.sqrt(
            (self.car_x - old_x) ** 2 +
            (self.car_y - old_y) ** 2
        )

        self.last_step_distance = step_distance
        self.total_distance += step_distance

        self.prev_x = self.car_x
        self.prev_y = self.car_y

        # 트랙 진행도 계산
        current_progress, self.dist_to_center = self.get_progress_on_track(
            self.car_x,
            self.car_y
        )

        self.last_progress_delta = self.get_progress_delta(current_progress)

        # 앞으로 간 경우만 누적 진행도에 더함
        if self.last_progress_delta > 0:
            self.lap_progress += self.last_progress_delta

        self.prev_track_progress = current_progress

        # 화면 다시 그림
        self.screen.fill(self.WHITE)
        self.draw_track()

        # observation 계산
        observation = self.get_observation()
        self.last_lane_observation = observation

        # reward 계산
        reward = self.calculate_reward(steering, throttle)

        # lane 관련 info
        lane_obs_len = self.lane_scan_rows * 2
        lane_obs = self.last_lane_observation[:lane_obs_len]
        confidences = lane_obs[1::2]
        valid_count = int(np.sum(confidences > 0.05))

        terminated = False
        truncated = False
        done_reason = "none"

        # =========================
        # 종료 조건
        # =========================

        # 1. 완주 종료
        #if self.lap_progress >= self.track_length * 0.95 and self.step_count > 50:
        #    terminated = True
        #    reward += 300.0
        #   done_reason = "lap_complete"

        # 2. 저속 종료
        if self.low_speed_count >= 35:
            terminated = True
            reward -= 80.0
            done_reason = "low_speed"

        # 3. 실제 트랙 중심선 기준 이탈
        # 기존 0.65는 살짝 빡셈. 일단 0.80으로 완화.

        elif self.dist_to_center > self.road_width * 0.80:
            terminated = True

            # 그냥 -35는 너무 약함.
            # 실패 에피소드가 200점 받는 걸 막기 위해 강한 패널티.
            remaining_ratio = 1.0 - np.clip(
                self.lap_progress / max(self.track_length, 1e-6),
                0.0,
                1.0
            )

            reward -= 180.0
            reward -= remaining_ratio * 80.0

            done_reason = "off_road_geometry"

        # 4. 카메라 차선 감지 기준 이탈
        elif self.is_off_road_by_lane_detection():
            terminated = True

            remaining_ratio = 1.0 - np.clip(
                self.lap_progress / max(self.track_length, 1e-6),
                0.0,
                1.0
            )

            reward -= 150.0
            reward -= remaining_ratio * 60.0

            done_reason = "off_road_lane"

        # 5. 화면 밖 종료
        elif self.car_x < 0 or self.car_x > self.width:
            terminated = True
            reward -= 180.0
            done_reason = "x_out"

        elif self.car_y < 0 or self.car_y > self.height:
            terminated = True
            reward -= 180.0
            done_reason = "y_out"

        # 6. 최대 step 도달
        if self.step_count >= self.max_steps:
            truncated = True
            done_reason = "max_steps"

        # 다음 step용 steering 저장
        self.prev_steering = steering

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
            "track_progress": current_progress,
            "last_progress_delta": self.last_progress_delta,
            "lap_progress": self.lap_progress,
            "track_length": self.track_length,
            "dist_to_center": self.dist_to_center,
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

        valid = confidences > 0.05
        valid_count = int(np.sum(valid))
        valid_ratio = valid_count / len(confidences)

        reward = 0.0

        # =========================
        # 1. 진행도 보상
        # 너무 크게 주면 박아도 고득점이 되므로 낮춤
        # =========================
        progress = self.last_progress_delta

        if progress > 0:
            reward += progress * 0.10
        else:
            # 뒤로 가는 것은 더 강하게 손해
            reward += progress * 0.40

        # =========================
        # 2. 도로 중심선 거리 패널티
        # 중심선에서 멀어질수록 계속 손해
        # =========================
        center_dist_ratio = self.dist_to_center / max(self.road_width / 2.0, 1e-6)

        # 도로 중앙 근처: 거의 패널티 없음
        # 차선 근처/바깥쪽: 빠르게 패널티 증가
        if center_dist_ratio > 0.4:
            reward -= ((center_dist_ratio - 0.4) ** 2) * 1.2

        # 도로 바깥에 가까우면 매 step 강한 경고
        if center_dist_ratio > 0.9:
            reward -= 1.0

        # =========================
        # 3. 차선을 아예 못 보면 강한 패널티
        # =========================
        if valid_count == 0:
            reward -= 1.5
            reward -= throttle * 0.8
            reward -= min(self.lane_lost_count * 0.08, 1.5)
            return float(reward)

        valid_errors = center_errors[valid]
        mean_abs_error = float(np.mean(np.abs(valid_errors)))

        center_score = 1.0 - np.clip(mean_abs_error, 0.0, 1.0)

        # =========================
        # 4. 중앙 유지 보상
        # 기존보다 약하게.
        # 주행 보조용이지 메인 보상이 아님.
        # =========================
        reward += center_score * valid_ratio * 0.35

        # =========================
        # 5. 차선이 많이 보이면 약간 보상
        # =========================
        reward += valid_ratio * 0.20

        # =========================
        # 6. 속도 보상
        # 중앙에 있을 때만 속도 보상.
        # 삐딱한 상태에서 밟으면 손해.
        # =========================
        if valid_count >= 5 and mean_abs_error < 0.35 and center_dist_ratio < 0.75:
            reward += throttle * 0.25
        else:
            reward -= throttle * 0.35

        # =========================
        # 7. 조향 패널티
        # 너무 크게 꺾는 것보다 부드러운 조향 유도
        # =========================
        reward -= abs(steering) * 0.05

        steer_change = abs(steering - self.prev_steering)
        reward -= steer_change * 0.04

        # =========================
        # 8. 너무 느린 행동 방지
        # =========================
        if throttle < 0.25:
            reward -= 0.25

        return float(reward)
    
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


# Backward-compatible alias for existing notebooks and experiments.
LineTracingCameraEnv = LaneKeepingEnv
