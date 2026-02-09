import math
import os
import random
import pygame
import sys
from array import array
from dataclasses import dataclass


def _get_chinese_font(size: int):
    """优先使用支持中文的字体，避免规则说明等出现乱码。"""
    for name in ("microsoftyahei", "Microsoft YaHei", "simhei", "SimHei", "msyh"):
        try:
            f = pygame.font.SysFont(name, size)
            if f:
                return f
        except Exception:
            pass
    win_font = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc")
    if os.path.isfile(win_font):
        try:
            return pygame.font.Font(win_font, size)
        except Exception:
            pass
    return pygame.font.Font(None, size)


# =======================
# 全局常量配置（方便调整）
# =======================

# 屏幕尺寸
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 640

# 颜色（复古高对比风格）
COLOR_BG = (10, 10, 30)        # 深蓝偏黑背景
COLOR_PLATFORM = (200, 200, 180)
COLOR_LADDER = (180, 180, 255)
COLOR_SLIDE = (255, 200, 120)
COLOR_PRISONER = (230, 230, 230)
COLOR_PRISONER_STRIPE = (40, 40, 40)
COLOR_POLICE = (20, 60, 160)
COLOR_TEXT = (230, 230, 230)
COLOR_BULLET = (220, 40, 40)  # 红色子弹（警察）
# 楼层设置
FLOOR_COUNT = 5
FLOOR_MARGIN_TOP = 80
FLOOR_MARGIN_BOTTOM = 80
# 从下到上等距分布
FLOOR_Y_SPACING = (SCREEN_HEIGHT - FLOOR_MARGIN_TOP - FLOOR_MARGIN_BOTTOM) // (FLOOR_COUNT - 1)

# 角色尺寸（逃犯略小）
PRISONER_WIDTH = 22
PRISONER_HEIGHT = 30
POLICE_WIDTH = 26
POLICE_HEIGHT = 38

# 速度设定：数值 × SPEED_UNIT = 像素/秒
SPEED_UNIT = 80.0
# 警察：巡逻（低速）、追捕（中速）、爬梯（上下一致）
POLICE_PATROL_SPEED = 1.0 * SPEED_UNIT  # 巡逻模式：低速
POLICE_CHASE_SPEED = 2.0 * SPEED_UNIT   # 追捕模式：中速（仍低于逃犯）
POLICE_LADDER_SPEED = 1.0 * SPEED_UNIT
# 逃犯：正常 3，爬梯 3（上下一致），滑梯 5
PRISONER_SPEED = 3.0 * SPEED_UNIT
PRISONER_LADDER_SPEED = 3.0 * SPEED_UNIT
SLIDE_SPEED = 5.0 * SPEED_UNIT

# 爬梯宽度
LADDER_WIDTH = 18

# 滑梯宽度
SLIDE_WIDTH = 20

# 跳跃：仅平地，高度为逃犯身高的两倍
JUMP_HEIGHT = PRISONER_HEIGHT * 2
GRAVITY = 400.0  # 像素/秒²
JUMP_INITIAL_VY = (2 * GRAVITY * JUMP_HEIGHT) ** 0.5

# 警察射击：冷却时间、子弹速度与尺寸
POLICE_SHOOT_COOLDOWN = 10.0
BULLET_SPEED = PRISONER_SPEED  # 子弹速度等于逃犯奔跑速度
BULLET_WIDTH = 12
BULLET_HEIGHT = 6
POLICE_SHOOT_MIN_DISTANCE = 10  # 射击最小距离（像素）
INTEL_EXPIRE_TIME = 5.0  # 情报失效时间（秒）：5秒内无发现则失效

# 帧率
FPS = 60

# 胜负时间（秒）
WIN_TIME_SECONDS = 60

# 音效：采样率（与 mixer 一致）
SAMPLE_RATE = 22050
LADDER_BEEP_DURATION = 0.12
LADDER_BEEP_FREQ = 420
SLIDE_WHOOSH_FREQ = 180  # 滑梯音效时长由落地时间决定，见 _make_slide_sound
LADDER_SOUND_INTERVAL = 0.14  # 爬梯音效间隔（秒）


def _make_shoot_sound() -> pygame.mixer.Sound | None:
    """射击音效。"""
    try:
        n = int(SAMPLE_RATE * 0.08)
        buf = array("h", [0] * n)
        for i in range(n):
            t = i / SAMPLE_RATE
            buf[i] = int(6000 * math.sin(2 * math.pi * 600 * t) * (1 - i / n))
        return pygame.mixer.Sound(buffer=buf)
    except Exception:
        return None


def _make_explosion_sound() -> pygame.mixer.Sound | None:
    """击中逃犯时的爆破音效。"""
    try:
        n = int(SAMPLE_RATE * 0.2)
        buf = array("h", [0] * n)
        for i in range(n):
            t = i / SAMPLE_RATE
            f = 150 * (1 + 2 * i / n)
            buf[i] = int(8000 * math.sin(2 * math.pi * f * t) * (1 - i / n))
        return pygame.mixer.Sound(buffer=buf)
    except Exception:
        return None


def _make_ladder_sound() -> pygame.mixer.Sound | None:
    """生成爬梯时的短促哔哔声。"""
    try:
        n = int(SAMPLE_RATE * LADDER_BEEP_DURATION)
        buf = array("h", [0] * n)
        for i in range(n):
            t = i / SAMPLE_RATE
            buf[i] = int(8000 * math.sin(2 * math.pi * LADDER_BEEP_FREQ * t) * (1 - i / n))
        return pygame.mixer.Sound(buffer=buf)
    except Exception:
        return None


def _make_slide_sound(duration_seconds: float) -> pygame.mixer.Sound | None:
    """生成下滑梯时的滑行声，持续 duration_seconds 秒（通常为从顶到底的时长）。"""
    try:
        n = int(SAMPLE_RATE * duration_seconds)
        if n <= 0:
            n = 1
        buf = array("h", [0] * n)
        for i in range(n):
            t = i / SAMPLE_RATE
            f = SLIDE_WHOOSH_FREQ * (1 - 0.25 * i / n)
            buf[i] = int(5000 * math.sin(2 * math.pi * f * t) * (1 - 0.7 * i / n))
        return pygame.mixer.Sound(buffer=buf)
    except Exception:
        return None


def floor_index_to_y(idx: int) -> int:
    """
    根据楼层索引(0-4, 0为1F最底层)返回平台y坐标（角色脚底坐标）。
    """
    return SCREEN_HEIGHT - FLOOR_MARGIN_BOTTOM - idx * FLOOR_Y_SPACING


# ================
# 爬梯与滑梯布局
# ================
# 爬梯：4 短（连 1 层）+ 3 中（连 2 层）+ 2 长（连 3 层）= 9 个，游戏开始时随机 X，之后不变
# 滑梯：始终 5F-1F，每 5 秒上下顶点改变位置

LADDERS: list[dict] = []
LADDER_CONNECTIONS: list[dict] = []


# 楼梯 X 方向最小间距，避免重叠
LADDER_MIN_SPACING = 50


def init_ladders() -> None:
    """游戏开始或重新开始时调用：9 个楼梯随机分配 X 位置，且互不重叠。"""
    global LADDERS, LADDER_CONNECTIONS
    left, right = 100, SCREEN_WIDTH - 100
    span = right - left
    slot_w = max(LADDER_MIN_SPACING + LADDER_WIDTH, span // 9)
    xs = []
    for i in range(9):
        low = left + i * slot_w
        high = min(left + (i + 1) * slot_w - LADDER_WIDTH, right - LADDER_WIDTH)
        if high > low:
            xs.append(random.randint(low, high))
        else:
            xs.append(low)
    random.shuffle(xs)
    # 4 短（连 1 层）、3 中（连 2 层）、2 长（连 3 层）
    LADDERS = [
        {"x": xs[0], "from_floor": 1, "to_floor": 2},
        {"x": xs[1], "from_floor": 2, "to_floor": 3},
        {"x": xs[2], "from_floor": 3, "to_floor": 4},
        {"x": xs[3], "from_floor": 4, "to_floor": 5},
        {"x": xs[4], "from_floor": 1, "to_floor": 3},
        {"x": xs[5], "from_floor": 2, "to_floor": 4},
        {"x": xs[6], "from_floor": 3, "to_floor": 5},
        {"x": xs[7], "from_floor": 1, "to_floor": 4},
        {"x": xs[8], "from_floor": 2, "to_floor": 5},
    ]
    LADDER_CONNECTIONS = []
    for l in LADDERS:
        a, b, x = l["from_floor"], l["to_floor"], l["x"]
        LADDER_CONNECTIONS.append({"x": x, "from_floor": a, "to_floor": b})
        LADDER_CONNECTIONS.append({"x": x, "from_floor": b, "to_floor": a})


# 滑梯：5F-1F，顶点每 5 秒变化（仅改 X，Y 由楼层定）
SLIDE_START_FLOOR = 5
SLIDE_END_FLOOR = 1
SLIDE_CHANGE_INTERVAL = 30.0
_slide_start_x = 100
_slide_end_x = 100


def get_slide_start_x() -> int:
    return _slide_start_x


def get_slide_end_x() -> int:
    return _slide_end_x


def set_slide_vertices(start_x: int, end_x: int) -> None:
    global _slide_start_x, _slide_end_x
    _slide_start_x = start_x
    _slide_end_x = end_x


def slide_position(t: float) -> tuple[int, int]:
    """t=0 在顶部(5F)，t=1 在底部(1F)。"""
    start_y = floor_index_to_y(SLIDE_START_FLOOR - 1)
    end_y = floor_index_to_y(SLIDE_END_FLOOR - 1)
    x = _slide_start_x + (_slide_end_x - _slide_start_x) * t
    y = start_y + (end_y - start_y) * t
    return int(x), int(y)


def is_on_floor(rect: pygame.Rect, floor_index: int) -> bool:
    y = floor_index_to_y(floor_index)
    return abs(rect.bottom - y) <= 2


def clamp(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))


@dataclass
class Bullet:
    """子弹：仅警察使用。"""
    rect: pygame.Rect
    vx: float
    owner: str = "police"


@dataclass
class Actor:
    rect: pygame.Rect
    floor: int  # 0-4
    vx: float = 0.0
    vy: float = 0.0
    on_ladder: bool = False
    climbing_ladder: dict | None = None  # 当前使用的爬梯描述
    sliding: bool = False
    slide_t: float = 0.0
    last_ladder_sound_t: float = 0.0
    jump_vy: float = 0.0  # 跳跃竖直速度（>0 向上），0 表示在地面
    last_shot_time: float = -999.0  # 上次射击时间（仅警察用）
    # 警察模式与状态（仅警察使用）
    mode: str = "patrol"  # "patrol"（巡逻）、"chase"（追捕）
    # 警察巡逻/搜索用（仅警察使用）
    patrol_dir: int = 1
    c_search_target: int = 1  # C 当前目标楼层 0-based
    c_search_going_up: bool = True


def create_prisoner() -> Actor:
    floor = 4  # 5F 右上方
    y = floor_index_to_y(floor)
    rect = pygame.Rect(0, 0, PRISONER_WIDTH, PRISONER_HEIGHT)
    rect.midbottom = (SCREEN_WIDTH - 50, y)
    return Actor(rect=rect, floor=floor)


def create_police_list() -> list[Actor]:
    # A 在 3 楼，B 在 2 楼，C 在 1 楼，均在左侧
    # A巡逻3-4-5楼（初始目标3楼），B巡逻2-3-4楼（初始目标2楼），C巡逻1-2-3楼（初始目标1楼）
    positions = [(2, 60, 3), (1, 100, 2), (0, 140, 1)]  # (floor_0based, x, initial_search_target)
    result: list[Actor] = []
    for floor, x, initial_target in positions:
        y = floor_index_to_y(floor)
        rect = pygame.Rect(0, 0, POLICE_WIDTH, POLICE_HEIGHT)
        rect.midbottom = (x, y)
        result.append(Actor(
            rect=rect, floor=floor, last_shot_time=-POLICE_SHOOT_COOLDOWN,
            mode="patrol",
            patrol_dir=1, c_search_target=initial_target, c_search_going_up=True,
        ))
    return result


def get_ladders_from_floor(floor_1_based: int) -> list[dict]:
    return [l for l in LADDER_CONNECTIONS if l["from_floor"] == floor_1_based]


def find_best_ladder_towards_floor(
    current_floor_1: int,
    target_floor_1: int,
    actor_x: float,
    prisoner_x: float | None = None,
) -> dict | None:
    """
    选择爬梯：往目标楼层方向，优先选落地后离逃犯更近的梯子（便于拦截）。
    """
    ladders = get_ladders_from_floor(current_floor_1)
    if not ladders:
        return None

    direction = 1 if target_floor_1 > current_floor_1 else -1
    candidates = []
    for l in ladders:
        next_floor = l["to_floor"]
        if (next_floor - current_floor_1) * direction <= 0:
            continue
        if direction == 1 and not (current_floor_1 < next_floor <= target_floor_1):
            continue
        if direction == -1 and not (target_floor_1 <= next_floor < current_floor_1):
            continue
        candidates.append(l)

    if not candidates:
        candidates = ladders

    # 优先选梯子 X 离逃犯更近的（落地后更容易堵到人）；否则选离自己最近的
    target_x = prisoner_x if prisoner_x is not None else actor_x
    candidates.sort(key=lambda l: (abs(l["x"] - target_x), abs(l["x"] - actor_x)))
    return candidates[0]


def ladder_rect_for_connection(conn: dict) -> pygame.Rect:
    """
    根据 from_floor 和 to_floor 构造一个覆盖两层之间的竖直矩形，作为爬梯判定区。
    """
    from_y = floor_index_to_y(conn["from_floor"] - 1)
    to_y = floor_index_to_y(conn["to_floor"] - 1)
    top = min(from_y, to_y)
    bottom = max(from_y, to_y)
    x_center = conn["x"]
    rect = pygame.Rect(0, 0, LADDER_WIDTH, bottom - top)
    rect.midtop = (x_center, top)
    return rect


def is_actor_in_ladder_x(actor: Actor, conn: dict) -> bool:
    ladder_rect = ladder_rect_for_connection(conn)
    return actor.rect.centerx >= ladder_rect.left and actor.rect.centerx <= ladder_rect.right


def ladder_passes_through_floors(conn: dict) -> range:
    """返回该梯子经过的楼层索引（0-based）的 range。"""
    lo = min(conn["from_floor"], conn["to_floor"]) - 1
    hi = max(conn["from_floor"], conn["to_floor"]) - 1
    return range(lo, hi + 1)


def get_floor_index_at_y(bottom_y: float) -> int:
    """根据角色底部 y 返回所在楼层索引（0-4），取最近的平台。"""
    best_k = 0
    best_d = 1e9
    for k in range(FLOOR_COUNT):
        d = abs(bottom_y - floor_index_to_y(k))
        if d < best_d:
            best_d = d
            best_k = k
    return best_k


def get_floor_below_y(bottom_y: float) -> int:
    """返回脚底 y 所在或下方的楼层（自由落体时应落到的层），不会算到上一层。"""
    for k in range(FLOOR_COUNT - 1, -1, -1):
        if floor_index_to_y(k) >= bottom_y:
            return k
    return 0


# 与楼层“平行”的判定容差（像素）
FLOOR_ALIGN_TOLERANCE = 10


def update_prisoner(
    prisoner: Actor,
    keys,
    dt: float,
    current_time: float = 0.0,
    sound_ladder: pygame.mixer.Sound | None = None,
    sound_slide: pygame.mixer.Sound | None = None,
    want_jump: bool = False,
):
    # 滑梯：按 ↑ 可中途停下，落在当前高度最近的楼层
    if prisoner.sliding:
        prisoner.vx = 0
        prisoner.vy = 0
        if keys[pygame.K_UP]:
            cur_y = prisoner.rect.bottom
            best_floor = 0
            best_d = 1e9
            for i in range(FLOOR_COUNT):
                fy = floor_index_to_y(i)
                d = abs(cur_y - fy)
                if d < best_d:
                    best_d = d
                    best_floor = i
            prisoner.floor = best_floor
            prisoner.rect.bottom = floor_index_to_y(best_floor)
            prisoner.sliding = False
            return
        start_y = floor_index_to_y(SLIDE_START_FLOOR - 1)
        end_y = floor_index_to_y(SLIDE_END_FLOOR - 1)
        slide_length = ((_slide_end_x - _slide_start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
        if slide_length <= 0:
            prisoner.sliding = False
            return
        t_increment = (SLIDE_SPEED * dt) / slide_length
        prisoner.slide_t += t_increment
        if prisoner.slide_t >= 1.0:
            prisoner.slide_t = 1.0
            prisoner.sliding = False
        x, y = slide_position(prisoner.slide_t)
        prisoner.rect.midbottom = (int(x), int(y))
        best_floor = 0
        best_d = 1e9
        for i in range(FLOOR_COUNT):
            fy = floor_index_to_y(i)
            d = abs(y - fy)
            if d < best_d:
                best_d = d
                best_floor = i
        prisoner.floor = best_floor
        return

    # 正常移动
    prisoner.vx = 0
    prisoner.vy = 0

    # 平地按 Enter 起跳（仅地面、非梯子、非滑梯）
    on_ground = not prisoner.on_ladder and not prisoner.sliding and prisoner.jump_vy == 0
    if want_jump and on_ground:
        prisoner.jump_vy = JUMP_INITIAL_VY

    # 左右移动总是允许（受楼层约束）
    if keys[pygame.K_LEFT]:
        prisoner.vx = -PRISONER_SPEED
    elif keys[pygame.K_RIGHT]:
        prisoner.vx = PRISONER_SPEED

    # 已在梯子上时：可随时按 UP/DOWN 切换上下方向
    if prisoner.on_ladder and prisoner.climbing_ladder is not None:
        if keys[pygame.K_UP]:
            prisoner.vy = -PRISONER_LADDER_SPEED
        elif keys[pygame.K_DOWN]:
            prisoner.vy = PRISONER_LADDER_SPEED
        # 不按则保持当前 vy，继续原方向
    else:
        # 站在楼层上时：只要梯子经过当前楼层且处于梯子 X 位置，就可按 UP/DOWN 上梯或下梯
        for conn in LADDER_CONNECTIONS:
            if prisoner.floor not in ladder_passes_through_floors(conn):
                continue
            ladder_rect = ladder_rect_for_connection(conn)
            in_ladder_x = prisoner.rect.centerx >= ladder_rect.left and prisoner.rect.centerx <= ladder_rect.right
            if in_ladder_x:
                if keys[pygame.K_UP]:
                    if max(conn["from_floor"], conn["to_floor"]) - 1 > prisoner.floor:
                        prisoner.on_ladder = True
                        prisoner.climbing_ladder = conn
                        prisoner.vy = -PRISONER_LADDER_SPEED
                        prisoner.jump_vy = 0.0
                elif keys[pygame.K_DOWN]:
                    if min(conn["from_floor"], conn["to_floor"]) - 1 < prisoner.floor:
                        prisoner.on_ladder = True
                        prisoner.climbing_ladder = conn
                        prisoner.vy = PRISONER_LADDER_SPEED
                        prisoner.jump_vy = 0.0
                break

    # 应用水平移动
    new_x = prisoner.rect.x + int(prisoner.vx * dt)
    new_x = int(clamp(new_x, 0, SCREEN_WIDTH - prisoner.rect.width))
    prisoner.rect.x = new_x

    if prisoner.on_ladder and prisoner.climbing_ladder is not None:
        if sound_ladder and prisoner.vy != 0 and (current_time - prisoner.last_ladder_sound_t >= LADDER_SOUND_INTERVAL):
            sound_ladder.play()
            prisoner.last_ladder_sound_t = current_time
        prisoner.rect.y += int(prisoner.vy * dt)
        conn = prisoner.climbing_ladder
        ladder_rect = ladder_rect_for_connection(conn)
        want_move_h = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
        landed = False

        if want_move_h:
            # 爬梯时按方向键：可移到旁边梯子、落到滑梯、或自由落体落到下方梯子/滑梯/地板（不用“最近层”避免跳到上一层）
            current_floor = get_floor_below_y(prisoner.rect.bottom)
            cx = prisoner.rect.centerx
            # 1) 边上是否有其他梯子（同高度经过 current_floor 的梯子，且 x 与当前梯子不同）
            for l in LADDERS:
                if l["x"] == conn["x"]:
                    continue
                half_w = prisoner.rect.width // 2
                if cx + half_w < l["x"] - LADDER_WIDTH // 2 or cx - half_w > l["x"] + LADDER_WIDTH // 2:
                    continue
                lo_f, hi_f = min(l["from_floor"], l["to_floor"]), max(l["from_floor"], l["to_floor"])
                if not (lo_f <= current_floor + 1 <= hi_f):
                    continue
                new_conn = next((c for c in LADDER_CONNECTIONS if c["x"] == l["x"] and c["from_floor"] == l["from_floor"] and c["to_floor"] == l["to_floor"]), None)
                if new_conn is not None:
                    prisoner.climbing_ladder = new_conn
                    prisoner.rect.centerx = l["x"]
                    landed = True
                    break
            if not landed and current_floor == SLIDE_START_FLOOR - 1:
                entry_width = SLIDE_WIDTH * 3
                start_y = floor_index_to_y(SLIDE_START_FLOOR - 1)
                entry_rect = pygame.Rect(
                    _slide_start_x - entry_width // 2,
                    start_y - 30 - 40,
                    entry_width,
                    80,
                )
                if prisoner.rect.colliderect(entry_rect):
                    prisoner.sliding = True
                    prisoner.slide_t = 0.0
                    prisoner.jump_vy = 0.0
                    prisoner.on_ladder = False
                    prisoner.climbing_ladder = None
                    if sound_slide:
                        sound_slide.play()
                    landed = True
            if not landed:
                # 自由落体下落，落到下方楼层（或途中梯子/滑梯由重力分支后续处理）
                prisoner.floor = get_floor_below_y(prisoner.rect.bottom)
                prisoner.jump_vy = -1.0
                prisoner.on_ladder = False
                prisoner.climbing_ladder = None
                landed = True
        else:
            prisoner.rect.centerx = ladder_rect.centerx
            want_step_off = (not keys[pygame.K_UP] and not keys[pygame.K_DOWN])
            if want_step_off:
                for k in ladder_passes_through_floors(conn):
                    floor_y = floor_index_to_y(k)
                    if abs(prisoner.rect.bottom - floor_y) <= FLOOR_ALIGN_TOLERANCE:
                        prisoner.rect.bottom = floor_y
                        prisoner.floor = k
                        prisoner.on_ladder = False
                        prisoner.climbing_ladder = None
                        landed = True
                        break
            if not landed:
                if prisoner.vy < 0:
                    target_floor_index = max(conn["from_floor"], conn["to_floor"]) - 1
                    target_y = floor_index_to_y(target_floor_index) - prisoner.rect.height
                    if prisoner.rect.top <= target_y:
                        prisoner.rect.top = target_y
                        prisoner.floor = target_floor_index
                        prisoner.on_ladder = False
                        prisoner.climbing_ladder = None
                elif prisoner.vy > 0:
                    target_floor_index = min(conn["from_floor"], conn["to_floor"]) - 1
                    target_y = floor_index_to_y(target_floor_index) - prisoner.rect.height
                    if prisoner.rect.top >= target_y:
                        prisoner.rect.top = target_y
                        prisoner.floor = target_floor_index
                        prisoner.on_ladder = False
                        prisoner.climbing_ladder = None
    elif prisoner.jump_vy != 0:
        # 跳跃中：重力与落地判定
        prisoner.jump_vy -= GRAVITY * dt
        prisoner.rect.y -= int(prisoner.jump_vy * dt)
        floor_y = floor_index_to_y(prisoner.floor)
        if prisoner.rect.bottom >= floor_y:
            prisoner.rect.bottom = floor_y
            prisoner.jump_vy = 0.0
    else:
        # 不在梯子上且未跳跃，保持在当前楼层的平台高度
        y = floor_index_to_y(prisoner.floor)
        prisoner.rect.bottom = y

    # 检查是否进入滑梯（仅 5F -> 1F，通过顶部一个矩形区域触发）
    if prisoner.floor == SLIDE_START_FLOOR - 1:
        entry_width = SLIDE_WIDTH * 3
        start_y = floor_index_to_y(SLIDE_START_FLOOR - 1)
        top_y = start_y - 30
        entry_rect = pygame.Rect(
            _slide_start_x - entry_width // 2,
            top_y - 40,
            entry_width,
            80,
        )
        # 滑梯可选：站在入口内且按 ↓ 才进入，不按则不会滑下去
        if prisoner.rect.colliderect(entry_rect) and keys[pygame.K_DOWN]:
            prisoner.sliding = True
            prisoner.slide_t = 0.0
            prisoner.jump_vy = 0.0
            if sound_slide:
                sound_slide.play()

# 预测时间（秒）：警察朝逃犯“即将到达”的位置移动
POLICE_PREDICT_TIME = 0.28

# 共享情报：任一警察看到逃犯时更新，无情报时 A/B/C 按分工巡逻
_last_known_prisoner_floor: int | None = None
_last_known_prisoner_x: float | None = None
_last_intel_update_time: float = -999.0  # 最后更新情报的时间


def reset_police_intel() -> None:
    global _last_known_prisoner_floor, _last_known_prisoner_x, _last_intel_update_time
    _last_known_prisoner_floor = None
    _last_known_prisoner_x = None
    _last_intel_update_time = -999.0


def _police_sees_prisoner(police: Actor, prisoner: Actor) -> bool:
    """同层或同梯（同一梯子列且楼层在梯子范围内）即视为看到逃犯。"""
    if police.floor == prisoner.floor:
        return True
    for conn in LADDER_CONNECTIONS:
        lr = ladder_rect_for_connection(conn)
        if abs(police.rect.centerx - lr.centerx) > LADDER_WIDTH + 4:
            continue
        if abs(prisoner.rect.centerx - lr.centerx) > LADDER_WIDTH + 4:
            continue
        lo = min(conn["from_floor"], conn["to_floor"]) - 1
        hi = max(conn["from_floor"], conn["to_floor"]) - 1
        if lo <= police.floor <= hi and lo <= prisoner.floor <= hi:
            return True
    return False


def update_police(
    police: Actor,
    prisoner: Actor,
    all_police: list[Actor],
    dt: float,
    police_index: int = 0,
    current_time: float = 0.0,
):
    global _last_known_prisoner_floor, _last_known_prisoner_x, _last_intel_update_time

    was_on_ladder = police.on_ladder
    police.sliding = False
    police.on_ladder = False
    police.climbing_ladder = None
    police.vx = 0.0
    police.vy = 0.0

    current_floor_1 = police.floor + 1
    sees_prisoner = _police_sees_prisoner(police, prisoner)

    # 检查情报是否失效（5秒内无发现则失效）
    intel_valid = False
    if _last_known_prisoner_floor is not None and _last_known_prisoner_x is not None:
        if current_time - _last_intel_update_time <= INTEL_EXPIRE_TIME:
            intel_valid = True

    if sees_prisoner:
        _last_known_prisoner_floor = prisoner.floor
        _last_known_prisoner_x = prisoner.rect.centerx + prisoner.vx * POLICE_PREDICT_TIME
        _last_intel_update_time = current_time
        intel_valid = True

    spread = 40
    chase_offset = (police_index - 1) * spread

    if sees_prisoner or intel_valid:
        # 有情报：追捕模式
        police.mode = "chase"
        if sees_prisoner:
            target_floor_0 = prisoner.floor
            target_x = (_last_known_prisoner_x or prisoner.rect.centerx) + chase_offset
        else:
            target_floor_0 = _last_known_prisoner_floor
            target_x = _last_known_prisoner_x + chase_offset
    else:
        # 无情报：巡逻模式
        police.mode = "patrol"
        # A: 3-4-5楼巡逻（floor index 2,3,4），B: 2-3-4楼巡逻（1,2,3），C: 1-2-3楼巡逻（0,1,2）
        patrol_ranges = [(2, 3, 4), (1, 2, 3), (0, 1, 2)]  # A, B, C 的巡逻楼层范围
        patrol_min, patrol_mid, patrol_max = patrol_ranges[police_index]
        
        # 如果不在巡逻范围内，先到中间楼层
        if police.floor < patrol_min:
            target_floor_0 = patrol_min
            target_x = SCREEN_WIDTH / 2
        elif police.floor > patrol_max:
            target_floor_0 = patrol_max
            target_x = SCREEN_WIDTH / 2
        else:
            # 在巡逻范围内，到达目标楼层且站在平台上时切换下一目标
            if police.floor == police.c_search_target and not was_on_ladder:
                if police.c_search_going_up:
                    if police.c_search_target >= patrol_max:
                        police.c_search_target = patrol_mid
                        police.c_search_going_up = False
                    else:
                        police.c_search_target += 1
                else:
                    if police.c_search_target <= patrol_min:
                        police.c_search_target = patrol_mid
                        police.c_search_going_up = True
                    else:
                        police.c_search_target -= 1
            target_floor_0 = police.c_search_target
            
            # 到达目标楼层后左右巡逻
            if police.floor == target_floor_0:
                if police.rect.x <= 40:
                    police.patrol_dir = 1
                if police.rect.x >= SCREEN_WIDTH - police.rect.width - 40:
                    police.patrol_dir = -1
                target_x = police.rect.centerx + police.patrol_dir * 300
            else:
                target_x = SCREEN_WIDTH / 2

    target_x = clamp(target_x, 30, SCREEN_WIDTH - 30)
    target_floor_1 = target_floor_0 + 1

    # 根据模式选择速度
    if police.mode == "patrol":
        current_speed = POLICE_PATROL_SPEED
        current_ladder_speed = POLICE_LADDER_SPEED
    elif police.mode == "chase":
        current_speed = POLICE_CHASE_SPEED
        current_ladder_speed = POLICE_LADDER_SPEED
    else:
        current_speed = 0.0
        current_ladder_speed = 0.0

    occupied_ladder_x = set()
    for other in all_police:
        if other is police:
            continue
        if other.on_ladder and other.climbing_ladder:
            occupied_ladder_x.add(other.climbing_ladder["x"])

    goal_x = _last_known_prisoner_x if _last_known_prisoner_x is not None else target_x

    goal_x = _last_known_prisoner_x if _last_known_prisoner_x is not None else target_x

    if current_floor_1 == target_floor_1:
        dx = target_x - police.rect.centerx
        if abs(dx) >= 8:
            if dx < 0:
                police.vx = -current_speed
            elif dx > 0:
                police.vx = current_speed
        elif sees_prisoner:
            # 同层且接近目标，朝逃犯移动（便于射击）
            toward = prisoner.rect.centerx - police.rect.centerx
            if toward < 0:
                police.vx = -current_speed
            elif toward > 0:
                police.vx = current_speed
    else:
        best_ladder = find_best_ladder_towards_floor(
            current_floor_1, target_floor_1, police.rect.centerx, prisoner_x=goal_x
        )
        if best_ladder and best_ladder["x"] in occupied_ladder_x:
            ladders_same_from = [l for l in get_ladders_from_floor(current_floor_1) if l["x"] not in occupied_ladder_x]
            if ladders_same_from:
                ladders_same_from.sort(key=lambda l: (abs(l["x"] - goal_x), abs(l["x"] - police.rect.centerx)))
                best_ladder = ladders_same_from[0]

        if best_ladder:
            ladder_rect = ladder_rect_for_connection(best_ladder)
            if abs(police.rect.centerx - ladder_rect.centerx) > 4:
                if police.rect.centerx < ladder_rect.centerx:
                    police.vx = current_speed
                else:
                    police.vx = -current_speed
            else:
                police.on_ladder = True
                police.climbing_ladder = best_ladder
                if target_floor_0 > police.floor:
                    police.vy = -current_ladder_speed
                else:
                    police.vy = current_ladder_speed

    new_x = police.rect.x + int(police.vx * dt)
    new_x = int(clamp(new_x, 0, SCREEN_WIDTH - police.rect.width))
    police.rect.x = new_x

    if police.on_ladder and police.climbing_ladder:
        police.rect.y += int(police.vy * dt)
        ladder_rect = ladder_rect_for_connection(police.climbing_ladder)
        police.rect.centerx = ladder_rect.centerx
        top_floor = max(police.climbing_ladder["from_floor"], police.climbing_ladder["to_floor"]) - 1
        bottom_floor = min(police.climbing_ladder["from_floor"], police.climbing_ladder["to_floor"]) - 1
        next_floor_index = top_floor if police.vy < 0 else bottom_floor
        target_y = floor_index_to_y(next_floor_index) - police.rect.height
        if police.vy < 0 and police.rect.top <= target_y:
            police.rect.top = target_y
            police.floor = next_floor_index
            police.on_ladder = False
            police.climbing_ladder = None
        elif police.vy > 0 and police.rect.top >= target_y:
            police.rect.top = target_y
            police.floor = next_floor_index
            police.on_ladder = False
            police.climbing_ladder = None
    else:
        y = floor_index_to_y(police.floor)
        police.rect.bottom = y


def try_police_shoot(
    police_list: list[Actor],
    prisoner: Actor,
    bullets: list[Bullet],
    current_time: float,
    sound_shoot: pygame.mixer.Sound | None = None,
) -> None:
    """至多一名警察射击：同层、距离>10像素、冷却完成的第一个警察立即发射一发。"""
    for police in police_list:
        if police.floor != prisoner.floor:
            continue
        # 距离检查：必须大于10像素
        dist = abs(police.rect.centerx - prisoner.rect.centerx)
        if dist <= POLICE_SHOOT_MIN_DISTANCE:
            continue
        # 冷却检查
        if current_time < police.last_shot_time + POLICE_SHOOT_COOLDOWN:
            continue
        # 射击：子弹速度等于逃犯奔跑速度
        police.last_shot_time = current_time
        vx = BULLET_SPEED if prisoner.rect.centerx > police.rect.centerx else -BULLET_SPEED
        rect = pygame.Rect(0, 0, BULLET_WIDTH, BULLET_HEIGHT)
        rect.centery = police.rect.centery
        if vx > 0:
            rect.left = police.rect.right
        else:
            rect.right = police.rect.left
        bullets.append(Bullet(rect=rect, vx=vx, owner="police"))
        if sound_shoot:
            sound_shoot.play()
        return


def draw_platforms(surface: pygame.Surface):
    for i in range(FLOOR_COUNT):
        y = floor_index_to_y(i)
        pygame.draw.line(surface, COLOR_PLATFORM, (40, y), (SCREEN_WIDTH - 40, y), 4)


def draw_floor_labels(surface: pygame.Surface, font: pygame.font.Font):
    """每层最左侧标出楼层数 1-5。"""
    for i in range(FLOOR_COUNT):
        y = floor_index_to_y(i)
        text = font.render(str(i + 1), True, COLOR_TEXT)
        rect = text.get_rect(midright=(38, y - 8))
        surface.blit(text, rect)


def draw_ladders(surface: pygame.Surface):
    # 只画原始LADDERS（避免重复）
    for l in LADDERS:
        from_y = floor_index_to_y(l["from_floor"] - 1)
        to_y = floor_index_to_y(l["to_floor"] - 1)
        top = min(from_y, to_y)
        bottom = max(from_y, to_y)
        x = l["x"]
        # 竖线
        pygame.draw.line(surface, COLOR_LADDER, (x - LADDER_WIDTH // 2, top), (x - LADDER_WIDTH // 2, bottom), 2)
        pygame.draw.line(surface, COLOR_LADDER, (x + LADDER_WIDTH // 2, top), (x + LADDER_WIDTH // 2, bottom), 2)
        # 横档
        step = 18
        y = top
        while y <= bottom:
            pygame.draw.line(surface, COLOR_LADDER, (x - LADDER_WIDTH // 2, y), (x + LADDER_WIDTH // 2, y), 2)
            y += step


def draw_slide(surface: pygame.Surface):
    start_y = floor_index_to_y(SLIDE_START_FLOOR - 1)
    end_y = floor_index_to_y(SLIDE_END_FLOOR - 1)
    start = (_slide_start_x, start_y)
    end = (_slide_end_x, end_y)
    pygame.draw.line(surface, COLOR_SLIDE, start, end, 6)


def draw_prisoner(surface: pygame.Surface, prisoner: Actor):
    pygame.draw.rect(surface, COLOR_PRISONER, prisoner.rect)
    stripe_height = 5
    y = prisoner.rect.top
    toggle = False
    while y < prisoner.rect.bottom:
        color = COLOR_PRISONER_STRIPE if toggle else COLOR_PRISONER
        stripe_rect = pygame.Rect(prisoner.rect.left, y, prisoner.rect.width, stripe_height)
        pygame.draw.rect(surface, color, stripe_rect)
        y += stripe_height
        toggle = not toggle


def update_bullets(
    bullets: list[Bullet],
    prisoner: Actor,
    police_list: list[Actor],
    dt: float,
    current_time: float,
    sound_explosion: pygame.mixer.Sound | None = None,
) -> bool:
    """更新子弹：警察子弹打逃犯。击中逃犯返回 True。"""
    prisoner_hit = False
    to_remove_b: list[Bullet] = []
    for b in bullets:
        b.rect.x += int(b.vx * dt)
        if b.rect.right < 0 or b.rect.left > SCREEN_WIDTH:
            to_remove_b.append(b)
            continue
        if b.owner == "police":
            if b.rect.colliderect(prisoner.rect):
                to_remove_b.append(b)
                prisoner_hit = True
                if sound_explosion:
                    sound_explosion.play()
                continue
            # 警察子弹碰到其他警察时只移除子弹（不伤害警察）
            for p in police_list:
                if b.rect.colliderect(p.rect):
                    to_remove_b.append(b)
                    break
    for x in to_remove_b:
        if x in bullets:
            bullets.remove(x)
    return prisoner_hit


def draw_bullets(surface: pygame.Surface, bullets: list[Bullet]):
    for b in bullets:
        pygame.draw.rect(surface, COLOR_BULLET, b.rect)


def draw_police(surface: pygame.Surface, police: Actor, label: str = ""):
    pygame.draw.rect(surface, COLOR_POLICE, police.rect)
    if label and surface.get_width() > 0:
        try:
            font = _get_chinese_font(14)
            text = font.render(label, True, COLOR_TEXT)
            r = text.get_rect(center=police.rect.center)
            surface.blit(text, r)
        except Exception:
            pass


def main():
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
    pygame.init()
    init_ladders()
    sound_ladder = _make_ladder_sound()
    slide_len_y = abs(floor_index_to_y(SLIDE_END_FLOOR - 1) - floor_index_to_y(SLIDE_START_FLOOR - 1))
    slide_duration_sec = max(0.5, slide_len_y / SLIDE_SPEED)
    sound_slide = _make_slide_sound(slide_duration_sec)
    sound_shoot = _make_shoot_sound()
    sound_explosion = _make_explosion_sound()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("警察与逃犯 - 复古追逃")
    clock = pygame.time.Clock()
    font = _get_chinese_font(32)
    small_font = _get_chinese_font(20)
    big_font = _get_chinese_font(72)

    prisoner = create_prisoner()
    police_list = create_police_list()
    bullets: list[Bullet] = []

    running = True
    game_over = False
    winner_text = ""
    elapsed_frozen = None  # 被抓或获胜时冻结的秒数

    start_time = pygame.time.get_ticks() / 1000.0
    last_slide_change_time = start_time
    restart_rect: pygame.Rect | None = None
    paused = False
    pause_start_time: float = 0.0
    elapsed_when_paused: float = 0.0

    while running:
        dt_ms = clock.tick(FPS)
        dt = dt_ms / 1000.0

        # 游戏结束时提前算好重新开始按钮区域，供点击判定
        if game_over:
            btn_w, btn_h = 160, 44
            restart_rect = pygame.Rect(0, 0, btn_w, btn_h)
            # 放在 1 层楼下面，不挡画面
            restart_rect.midtop = (SCREEN_WIDTH // 2, floor_index_to_y(0) + 18)
        else:
            restart_rect = None

        current_time = pygame.time.get_ticks() / 1000.0
        want_jump = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.ACTIVEEVENT:
                if event.gain == 0:
                    paused = True
                    pause_start_time = current_time
                    elapsed_when_paused = current_time - start_time
                else:
                    paused = False
                    start_time += current_time - pause_start_time
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not game_over:
                want_jump = True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and restart_rect and restart_rect.collidepoint(event.pos):
                init_ladders()
                prisoner = create_prisoner()
                police_list = create_police_list()
                bullets.clear()
                game_over = False
                winner_text = ""
                elapsed_frozen = None
                paused = False
                start_time = pygame.time.get_ticks() / 1000.0
                last_slide_change_time = start_time
                reset_police_intel()
                for p in police_list:
                    p.mode = "patrol"

        keys = pygame.key.get_pressed()
        if elapsed_frozen is not None:
            elapsed = elapsed_frozen
        else:
            elapsed = current_time - start_time

        if not game_over and not paused:
            # 滑梯每 5 秒改变上下顶点位置
            if current_time - last_slide_change_time >= SLIDE_CHANGE_INTERVAL:
                last_slide_change_time = current_time
                set_slide_vertices(
                    random.randint(80, SCREEN_WIDTH - 80),
                    random.randint(80, SCREEN_WIDTH - 80),
                )
            update_prisoner(prisoner, keys, dt, current_time, sound_ladder, sound_slide, want_jump)
            for idx, p in enumerate(police_list):
                update_police(p, prisoner, police_list, dt, police_index=idx, current_time=current_time)
            try_police_shoot(police_list, prisoner, bullets, current_time, sound_shoot)
            prisoner_hit = update_bullets(bullets, prisoner, police_list, dt, current_time, sound_explosion)
            if prisoner_hit:
                game_over = True
                elapsed_frozen = elapsed
                winner_text = "Police Won"

            # 碰撞检测：任意警察撞到逃犯即失败（安全区内且未超时则不受抓）
            for p in police_list:
                if p.rect.colliderect(prisoner.rect):
                    game_over = True
                    elapsed_frozen = elapsed
                    winner_text = "Police Won"
                    break

            # 时间达标且未碰撞则逃犯胜利
            if not game_over and elapsed >= WIN_TIME_SECONDS:
                game_over = True
                elapsed_frozen = elapsed
                winner_text = "Prisoner Win"

        # 绘制
        screen.fill(COLOR_BG)
        draw_platforms(screen)
        draw_floor_labels(screen, small_font)
        draw_ladders(screen)
        draw_slide(screen)
        draw_bullets(screen, bullets)
        # 角色
        draw_prisoner(screen, prisoner)
        labels = ["A", "B", "C"]
        for idx, p in enumerate(police_list):
            draw_police(screen, p, labels[idx] if idx < len(labels) else "")

        display_elapsed = elapsed_when_paused if paused else elapsed
        time_text = small_font.render(f"Time: {int(display_elapsed):02d} s", True, COLOR_TEXT)
        screen.blit(time_text, (20, 20))

        if paused and not game_over:
            pause_surf = small_font.render("Pause", True, COLOR_TEXT)
            pause_rect = pause_surf.get_rect(midtop=(SCREEN_WIDTH // 2, 8))
            screen.blit(pause_surf, pause_rect)

        if game_over:
            text_surf = font.render(winner_text, True, COLOR_TEXT)
            y_above_5f = floor_index_to_y(4) - 50
            text_rect = text_surf.get_rect(midtop=(SCREEN_WIDTH // 2, y_above_5f))
            screen.blit(text_surf, text_rect)

            # 重新开始按钮（屏幕中下方）
            if restart_rect is not None:
                pygame.draw.rect(screen, COLOR_PLATFORM, restart_rect)
                pygame.draw.rect(screen, COLOR_TEXT, restart_rect, 2)
                btn_label = font.render("Restart", True, (20, 20, 20))
                btn_label_rect = btn_label.get_rect(center=restart_rect.center)
                screen.blit(btn_label, btn_label_rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

