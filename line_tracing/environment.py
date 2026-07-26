import pygame
import numpy as np
import math
import cv2
import gymnasium as gym
from gymnasium import spaces

from driving.procedural import CameraDomainRandomizer, generate_closed_track


class LineTracingCameraEnv(gym.Env):

    def __init__(
        self,
        render_mode=False,
        *,
        procedural_tracks=True,
        domain_randomization=False,
    ):
        super().__init__()
        pygame.init()
        self.low_speed_count = 0
        self.prev_steering = 0.0
        
        # 라인트레이싱 선 설정
        self.line_width = 8
        self.off_line_margin = 4
        self.off_line_limit = 3

        self.min_black_ratio = 0.008
        self.line_lost_limit = 8
        self.line_lost_count = 0
        self.off_line_count = 0

        # =========================
        # 트랙 진행도 계산용
        # =========================
        self.track_cumulative_lengths = None
        self.track_length = 1.0
        self.prev_track_progress = 0.0
        self.last_progress_delta = 0.0
        self.lap_progress = 0.0
        
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
        self.procedural_tracks = procedural_tracks
        self.domain_randomizer = CameraDomainRandomizer(
            enabled=domain_randomization
        )
        self._track_sequence = 0

        self.available_tracks = [0,1,2,3,4,5,6,7]
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
        self.max_turn_rate = 55.0  # degree per second

    # 초당 최대 이동 속도
        self.max_speed = 45.0  # pixel per second
        # =========================
        # 카메라 설정
        # =========================
        # 실제 라즈베리파이 카메라가 본다고 가정하는 영역 크기
        self.camera_width = 160
        self.camera_height = 120
        self.camera_near_dist = 5.0
        self.camera_far_dist = 135.0
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
        self.camera_distance = 60

        # 카메라 관찰값을 몇 칸으로 나눌 것인지
        # 예: 16칸이면 화면 가로를 16등분해서 각 칸에 선이 있는지 확인
        self.observation_bins = 16
        self.observation_rows = 3
        self._observation_bin_width = self.camera_width // self.observation_bins
        self._observation_bin_starts = (
            np.arange(self.observation_bins) * self._observation_bin_width
        )
        self._observation_bin_widths = np.diff(
            np.append(self._observation_bin_starts, self.camera_width)
        )

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

        if self.procedural_tracks:
            rng = getattr(self, "np_random", np.random.default_rng())
            return generate_closed_track(
                rng,
                width=self.width,
                height=self.height,
                margin=75.0,
            )

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
            ((220, 300), (300, 180), (470, 190), (580, 280)),
            ((580, 280), (700, 380), (560, 470), (430, 390)),
            ((430, 390), (310, 310), (230, 470), (160, 360)),
            ((160, 360), (100, 270), (170, 230), (220, 300)),
            ],
            # =========================
            # Track 3: 좌우 변화가 조금 더 있는 루프
            # =========================
            [
                ((230, 260), (340, 120), (560, 190), (630, 310)),
                ((630, 310), (710, 470), (490, 500), (390, 390)),
                ((390, 390), (280, 280), (210, 430), (170, 340)),
                ((170, 340), (110, 250), (170, 210), (230, 260)),
            ],
                        # =========================
            # Track 4: 큰 완만한 외곽 루프
            # =========================
            [
                ((180, 300), (210, 180), (380, 145), (540, 200)),
                ((540, 200), (700, 255), (735, 385), (610, 455)),
                ((610, 455), (485, 525), (300, 475), (210, 395)),
                ((210, 395), (125, 320), (135, 230), (180, 300)),
            ],

            # =========================
            # Track 5: 콩 모양 루프
            # =========================
            [
                ((250, 260), (330, 165), (470, 185), (585, 275)),
                ((585, 275), (690, 360), (620, 455), (490, 425)),
                ((490, 425), (395, 405), (340, 320), (265, 390)),
                ((265, 390), (175, 470), (125, 345), (250, 260)),
            ],

            # =========================
            # Track 6: 위아래 폭이 큰 루프
            # =========================
            [
                ((230, 330), (250, 170), (430, 130), (560, 215)),
                ((560, 215), (700, 305), (655, 455), (510, 470)),
                ((510, 470), (390, 485), (335, 360), (235, 410)),
                ((235, 410), (115, 470), (130, 275), (230, 330)),
            ],

            # =========================
            # Track 7: S자 느낌이 있는 난이도 높은 루프
            # =========================
            [
                ((220, 310), (285, 190), (430, 180), (520, 255)),
                ((520, 255), (610, 330), (690, 235), (680, 360)),
                ((680, 360), (665, 480), (470, 505), (405, 395)),
                ((405, 395), (345, 295), (265, 485), (180, 375)),
                ((180, 375), (105, 285), (155, 225), (220, 310)),
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
        self.domain_randomizer.reset(self.np_random)
        self._track_sequence += 1

        # 매 에피소드마다 트랙 랜덤 선택
        self.track_id = self._track_sequence
        self.track_points = self.build_bezier_track(self.track_id)

        all_start_points = self.generate_start_points_by_interval(interval=20)

        bad_start_ids_by_track = {
            0: {3,7},
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
        self.off_line_count = 0
        self.line_lost_count = 0

        # 시작점 하나만 랜덤 선택
        self.start_id = int(self.np_random.integers(len(self.start_points)))
        self.start_original_id = self.start_original_ids[self.start_id]

        start_x, start_y, start_angle = self.start_points[self.start_id]

        self.car_x = start_x
        self.car_y = start_y
        self.car_angle = start_angle

        # 진행도 계산 초기화
        self.build_track_progress_table()
        self.prev_track_progress = self.get_progress_on_track(
            self.car_x,
            self.car_y
        )
        self.last_progress_delta = 0.0
        self.lap_progress = 0.0

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
        track_points에서 일정 간격마다 시작점을 생성.
        시작 각도는 바로 다음 점이 아니라 lookahead만큼 앞의 점을 보고 계산한다.
        이렇게 해야 시작 방향이 덜 튄다.
        """
        start_points = []

        n = len(self.track_points)
        lookahead = 5

        for i in range(0, n, interval):
            p1 = self.track_points[i]
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
        self.screen.fill(self.WHITE)

        pygame.draw.lines(
            self.screen,
            self.BLACK,
            self.closed_track,
            self.track_points,
            self.line_width
        )

    def get_distance_to_track_center(self, x, y):
        """
        자동차 위치와 track 중심선 사이의 최단거리 계산.
        라인트레이싱에서는 이 값이 커지면 선을 벗어난 것.
        """
        best_dist_sq = float("inf")

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
                best_dist_sq = dist_sq

        return math.sqrt(best_dist_sq)

    def build_track_progress_table(self):
        """
        track_points를 따라 누적 거리 테이블을 만든다.
        닫힌 트랙이면 마지막 점 -> 첫 점 구간도 포함한다.
        """
        points = self.track_points
        cumulative = [0.0]
        total = 0.0

        n = len(points)
        segment_count = n if self.closed_track else n - 1

        for i in range(segment_count):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]

            dx = x2 - x1
            dy = y2 - y1
            seg_len = math.sqrt(dx * dx + dy * dy)

            total += seg_len
            cumulative.append(total)

        self.track_cumulative_lengths = cumulative
        self.track_length = max(total, 1e-6)

    def get_progress_on_track(self, x, y):
        """
        현재 위치를 트랙 중심선에 투영해서
        트랙을 따라간 진행 거리 progress를 계산한다.
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

        return best_progress

    def get_progress_delta(self, current_progress):
        """
        현재 progress와 이전 progress의 차이를 계산한다.
        닫힌 트랙에서 끝 지점 -> 시작 지점으로 넘어가는 경우도 보정한다.
        """
        delta = current_progress - self.prev_track_progress

        # 한 바퀴 경계 보정
        if delta < -0.5 * self.track_length:
            delta += self.track_length
        elif delta > 0.5 * self.track_length:
            delta -= self.track_length

        # 이상치 방지
        delta = float(np.clip(delta, -10.0, 10.0))

        return delta

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
        차량 진행 방향 기준 전방 카메라 이미지 생성.
        기존 방식은 화면 좌표 기준 사각형 crop이라서
        차량이 회전해도 카메라 이미지가 같이 회전하지 않는 문제가 있었음.
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

        return self.domain_randomizer.apply(camera_image, self.np_random)

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

    def get_observation_from_binary(self, binary):
        """
        이미 계산된 binary 이미지로 observation 생성.
        get_camera_image/preprocess를 반복 호출하지 않기 위한 최적화 버전.
        """
        h, w = binary.shape

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

            ys, xs = np.where(row_binary > 0)

            if len(xs) == 0:
                row_center = 0.0
            else:
                line_center_x = np.mean(xs)
                row_center = (line_center_x - w / 2) / (w / 2)

            row_centers.append(row_center)

            column_counts = np.count_nonzero(row_binary, axis=0)
            counts = np.add.reduceat(
                column_counts,
                self._observation_bin_starts,
            )
            ratios = counts / (
                row_binary.shape[0] * self._observation_bin_widths
            )
            observation.extend(ratios.tolist())

        observation.extend(row_centers)

        angle_normalized = ((self.car_angle + 180) % 360 - 180) / 180.0
        speed_norm = self.current_speed / self.max_speed

        observation.append(angle_normalized)
        observation.append(speed_norm)

        return np.array(observation, dtype=np.float32)

    def get_observation(self):
        """
        기존 호환용 함수.
        step()에서는 가능하면 이 함수 대신 get_observation_from_binary()를 사용.
        """
        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)
        return self.get_observation_from_binary(binary)
    
    def calculate_line_position_from_binary(self, binary):
        """
        이미 계산된 binary 이미지에서 선 위치 계산.
        """
        h, w = binary.shape
        ys, xs = np.where(binary > 0)

        if len(xs) == 0:
            return None

        line_center_x = np.mean(xs)
        normalized_position = (line_center_x - w / 2) / (w / 2)

        return normalized_position

    def calculate_line_position(self):
        """
        기존 호환용 함수.
        """
        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)
        return self.calculate_line_position_from_binary(binary)

    def is_off_track_from_binary(self, binary):
        """
        이미 계산된 binary 이미지로 off_track 판단.
        """
        black_ratio = np.sum(binary > 0) / binary.size

        if black_ratio < self.min_black_ratio:
            self.line_lost_count += 1
        else:
            self.line_lost_count = 0

        return self.line_lost_count >= self.line_lost_limit

    def is_off_track(self):
        """
        기존 호환용 함수.
        step()에서는 가능하면 is_off_track_from_binary()를 사용.
        """
        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)
        return self.is_off_track_from_binary(binary)
    
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

        # 트랙 진행도 계산
        current_progress = self.get_progress_on_track(self.car_x, self.car_y)
        self.last_progress_delta = self.get_progress_delta(current_progress)

        if self.last_progress_delta > 0:
            self.lap_progress += self.last_progress_delta

        self.prev_track_progress = current_progress

        # 화면 다시 그림
        self.screen.fill(self.WHITE)
        self.draw_track()

        # 카메라 이미지는 step당 딱 1번만 계산
        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)

        # observation 계산
        observation = self.get_observation_from_binary(binary)

        # reward 계산
        reward = self.calculate_reward_from_binary(binary, steering, throttle)

        # 다음 reward 계산을 위해 저장
        self.prev_steering = steering

        dist_to_line = self.get_distance_to_track_center(self.car_x, self.car_y)
    
                # 종료 조건
        terminated = False
        truncated = False
        done_reason = "none"

               # 실제 위치 기준 선 이탈 정도
        off_line_threshold = self.line_width / 2.0 + self.off_line_margin

        if dist_to_line > off_line_threshold:
            self.off_line_count += 1
            reward -= 1.0
            reward -= min((dist_to_line - off_line_threshold) * 0.05, 2.0)
        else:
            self.off_line_count = 0

        # 1. 실제 위치 기준 선 이탈 종료
        if self.off_line_count >= self.off_line_limit:
            terminated = True
            reward -= 80.0
            done_reason = "off_line_geometry"

        # 2. 카메라 기준 선 놓침
        elif self.is_off_track_from_binary(binary):
            terminated = True

            progress_ratio = min(self.step_count / self.target_steps, 1.0)
            early_fail_penalty = 3.0 * (1.0 - progress_ratio)

            reward -= 6.0
            reward -= early_fail_penalty
            done_reason = "off_track"

        # 3. 저속 종료
        elif self.low_speed_count >= 30:
            terminated = True
            reward -= 5.0
            done_reason = "low_speed"

        # 4. 화면 밖 종료
        elif self.car_x < 0 or self.car_x > self.width:
            terminated = True
            reward -= 6.0
            done_reason = "x_out"

        elif self.car_y < 0 or self.car_y > self.height:
            terminated = True
            reward -= 6.0
            done_reason = "y_out"

        # 5. max_steps까지 버티면 완주 처리
        if (not terminated) and self.step_count >= self.max_steps:
            terminated = True
            reward += 300.0
            done_reason = "lap_complete"
        
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
            "dist_to_line": dist_to_line,
            "last_progress_delta": self.last_progress_delta,
            "lap_progress": self.lap_progress,
            "track_length": self.track_length,
        }

        return observation, reward, terminated, truncated, info

    def calculate_reward_from_binary(self, binary, steering, throttle):
        line_position = self.calculate_line_position_from_binary(binary)

        if line_position is None:
            return -2.0

        center_error = abs(line_position)
        center_reward = 1.0 - center_error

        reward = 0.0

        # 실제 진행도 보상
        progress = self.last_progress_delta

        if progress > 0:
            reward += progress * 0.08
        else:
            reward += progress * 0.25

        # 선이 중앙에서 많이 벗어난 상태에서 너무 빠르면 벌점
        if center_error > 0.25 and throttle > 0.55:
            reward -= (throttle - 0.55) * 2.0

        # 선이 화면 끝쪽이면 더 강하게 속도 벌점
        if center_error > 0.45 and throttle > 0.45:
            reward -= (throttle - 0.45) * 3.0

        # 많이 꺾는 중인데 속도가 높으면 벌점
        if abs(steering) > 0.3 and throttle > 0.65:
            reward -= (throttle - 0.65) * abs(steering) * 2.0

        # 선 중앙 + 적절한 속도
        target_throttle = 0.55 + 0.25 * center_reward
        speed_score = 1.0 - abs(throttle - target_throttle)

        reward += center_reward * max(speed_score, 0.0) * 1.2

        # 너무 느린 행동 방지
        if throttle < 0.1:
            reward -= 0.8
        elif throttle < 0.25:
            reward -= 0.2

        # 선이 너무 가장자리면 벌점
        if center_error > 0.45:
            reward -= 1.2

        # 거의 중앙에 있을 때 추가 보상
        if center_error < 0.18:
            reward += 0.25

        # 조향 벌점
        reward -= abs(steering) * 0.04

        steering_change = abs(steering - self.prev_steering)
        reward -= steering_change * 0.02

        return float(reward)

    def calculate_reward(self, steering, throttle):
        """
        기존 호환용 함수.
        step()에서는 calculate_reward_from_binary()를 사용하는 게 빠름.
        """
        camera_image = self.get_camera_image()
        binary = self.preprocess_camera_image(camera_image)
        return self.calculate_reward_from_binary(binary, steering, throttle)

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
