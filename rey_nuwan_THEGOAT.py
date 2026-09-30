# erebus 2026 controller - left wall following + finds victims and cognitive targets
import math
import os
import struct
from collections import Counter
import cv2
import numpy as np
from controller import Robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())
#devices
left_motor = robot.getDevice("wheel2 motor")
right_motor = robot.getDevice("wheel1 motor")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))
    motor.setVelocity(0)
left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
gps = robot.getDevice("gps")
imu = robot.getDevice("imu")
lidar = robot.getDevice("lidar")
colour_sensor = robot.getDevice("colour_sensor")
camera1 = robot.getDevice("camera1")
camera2 = robot.getDevice("camera2")
emitter = robot.getDevice("emitter")
for device in (left_encoder, right_encoder, gps, imu, lidar, colour_sensor, camera1, camera2):
    device.enable(timestep)
print("Emitter ready." if emitter else "WARNING: emitter not found. Token reporting is unavailable.")
#settings
MAPSIZE = 241
CELL = 0.01  # 1 char on the map is 1cm
START_X = MAPSIZE // 2
START_Z = MAPSIZE // 2
STEP_ENC = 2.95  # about 6cm
MAX_STEPS = 500  # just for testing
SPEED = 6.275
BACK_SPEED = 6.275
TURN_FAST = 6.275
TURN_MED = 6.275
TURN_SLOW = 2.0
TURN_TOL = 1.0
KP = 0.04
MAX_CORR = 0.15
SCAN_DIST = 0.095  # side walls closer then this get marked
FRONT_CLEAR = 0.097
STOP_DIST = 0.05
SIDE_CLEAR = 0.096
MIN_STEPS_LEFT = 2  # steps to go after a hole before trying left again
#floor colours (from the sensor readings)
BLACK_MAX = 80
BLUE_MIN, BLUE_OTHER, BLUE_DIFF = 180, 150, 60
GREEN_MIN, GREEN_OTHER, GREEN_DIFF = 170, 160, 45
YEL_R, YEL_G = 170, 170
YEL_B, YEL_RG = 150, 100
BRN_R_MIN, BRN_R_MAX = 210, 255
BRN_G_MIN, BRN_G_MAX = 180, 245
BRN_B_MIN, BRN_B_MAX = 100, 180
BRN_RG, BRN_GB = 50, 40
SILV_MIN, SILV_MAX, SILV_DIFF = 250, 255, 8
PUR_R, PUR_G, PUR_B = 150, 140, 200
PUR_BR, PUR_BG = 25, 60
RED_MIN, RED_OTHER, RED_DIFF = 200, 150, 60
# colour sensor is a bit in front of the gps
FLOOR_OFFSET = 2
# symbol, blocked or not, name for printing
FLOORS = {"black":  {"symbol": "H", "blocked": True,  "hazard": "black hole",     "log": "BLACK FLOOR"}, "blue":   {"symbol": "B", "blocked": True,  "hazard": "blue passage",   "log": "BLUE PASSAGE"}, "brown":  {"symbol": "M", "blocked": True,  "hazard": "brown floor",    "log": "BROWN FLOOR"}, "green":  {"symbol": "G", "blocked": True,  "hazard": "green passage",  "log": "GREEN PASSAGE"}, "yellow": {"symbol": "Y", "blocked": True,  "hazard": "yellow passage", "log": "YELLOW PASSAGE"}, "purple": {"symbol": "P", "blocked": True,  "hazard": "purple passage", "log": "PURPLE PASSAGE"}, "red":    {"symbol": "r", "blocked": True,  "hazard": "red passage",    "log": "RED PASSAGE"}, "silver": {"symbol": "C", "blocked": False, "hazard": "checkpoint",     "log": "SILVER CHECKPOINT"},}
for region in FLOORS.values():
    region["cells"] = set()
FLOOR_SYMS = {region["symbol"] for region in FLOORS.values()}
#token stuff, distances are from robot centre to the wall
MAX_WALL = 0.12
REPORT_DIST = 0.057  # rules say half a tile (6cm)
APPROACH_DIST = 0.050
APPROACH_MAX = 4.9
APPROACH_SPD = 2.0
LENS_OFFSET = 0.0275  # worked out from the camera pics, it fixes itself while driving
TARGET_R = 0.025  # target is 5cm across
ALIGN_TOL = 3
TRIGGER_PX = 10  # wait till its this close to the middle when driving
ALIGN_KP = 0.12
CREEP_MIN = 0.5
CREEP_MAX = 1.5
ALIGN_MAX = 2.5
LOST_FRAMES = 6
SETTLE = 3
CALIB_MOVE = 0.25
STOP_TIME = 1.3  # need to stop atleast 1s
READ_TIME = 0.4
SCAN_MOVE = 0.8  # ~1.6cm each way if rings arent clear
MIN_VOTES = 8
MIN_SHARE = 0.7
HOLD_STEPS = 8
SAVE_PICS = True  # saves camera pics at every stop, turn off for comp
PIC_FOLDER = "camera_frames"
SAME_TOKEN_DIST = 0.05
MAX_TRIES = 2
RING_VALS = {"black": -2, "red": -1, "yellow": 0, "green": 1, "blue": 2}
HAZARDS = {0: "F", 1: "P", 2: "C", 3: "O"}
LABEL_COLOURS = {1: "black", 2: "red", 3: "yellow", 4: "green", 5: "blue"}
# lidar ray index for each side
resolution = lidar.getHorizontalResolution()
RIGHT_IDX = resolution // 4
LEFT_IDX = 3 * resolution // 4
# creep_sign is a guess, it gets corrected the first time
cams = [{"name": "camera1-right", "device": camera1, "side": 1,  "turn": -90, "ray": RIGHT_IDX, "creep_sign": -1, "calibrated": False, "lens_offset": LENS_OFFSET}, {"name": "camera2-left",  "device": camera2, "side": -1, "turn": 90,  "ray": LEFT_IDX, "creep_sign": 1,  "calibrated": False, "lens_offset": LENS_OFFSET},]
#state
grid = [["?"] * MAPSIZE for _ in range(MAPSIZE)]
side_walls = set()
state = "WAIT"
wait_steps = 20
heading = 0.0
target_heading = 0.0
start_heading = None
start_gps = None
passes = 0
pass_steps = 0
total_steps = 0
start_l = 0.0
start_r = 0.0
rev_l = 0.0
rev_r = 0.0
rev_target = 0.0
turn_after = None  # -90 right 90 left
last_hazard = ""
dodging_hole = False
detour_steps = 0
seen_tokens = []  # every token we dealt with already
tok = None
last_summary = -1

#helpers
def set_wheels(left_speed, right_speed):
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

def normalize_heading(value):
    return value % 360

# shortest way to turn, eg 270 vs 0 gives -90
def angle_error(target, current):
    return (target - current + 180) % 360 - 180

def cardinal(value):
    return normalize_heading(round(value / 90) * 90)

def imu_is_ready():
    return all(math.isfinite(v) for v in imu.getRollPitchYaw())

def get_absolute_imu_heading():
    return normalize_heading(math.degrees(imu.getRollPitchYaw()[2]))

# left turns are positive
def get_relative_heading():
    if start_heading is None:
        return 0.0
    return normalize_heading(get_absolute_imu_heading() - start_heading)

def average_encoder():
    return (left_encoder.getValue() + right_encoder.getValue()) / 2

def in_map(x, z):
    return 0 <= x < MAPSIZE and 0 <= z < MAPSIZE

# robot position in the start frame (map x, map z, forward m, right m)
def gps_to_map():
    position = gps.getValues()
    change_x = position[0] - start_gps[0]
    change_z = position[2] - start_gps[1]
    start_angle = math.radians(start_heading)
    forward_x, forward_z = -math.sin(start_angle), -math.cos(start_angle)
    right_x, right_z = math.cos(start_angle), -math.sin(start_angle)
    relative_forward = change_x * forward_x + change_z * forward_z
    relative_right = change_x * right_x + change_z * right_z
    map_x = START_X + round(relative_forward / CELL)
    map_z = START_Z + round(relative_right / CELL)
    return map_x, map_z, relative_forward, relative_right

# drives forward and steers back onto the heading
def drive_straight(speed, hold_heading):
    correction = angle_error(hold_heading, heading) * KP
    correction = max(-MAX_CORR, min(MAX_CORR, correction))
    if correction > 0:
        set_wheels(speed - correction, speed)
    else:
        set_wheels(speed, speed + correction)

# spins on the spot, returns true once its facing the right way
def turn_toward(goal_heading):
    error = angle_error(goal_heading, heading)
    if abs(error) < TURN_TOL:
        set_wheels(0, 0)
        return True
    if abs(error) < 10:
        speed = TURN_SLOW
    elif abs(error) < 30:
        speed = TURN_MED
    else:
        speed = TURN_FAST
    set_wheels(-speed, speed) if error > 0 else set_wheels(speed, -speed)
    return False

#floor
def read_colour():
    image = colour_sensor.getImage()
    width = colour_sensor.getWidth()
    return (colour_sensor.imageGetRed(image, width, 0, 0), colour_sensor.imageGetGreen(image, width, 0, 0), colour_sensor.imageGetBlue(image, width, 0, 0))

# brown goes before yellow cause swamps pass the yellow check too
def classify_floor(red, green, blue):
    if red < BLACK_MAX and green < BLACK_MAX and blue < BLACK_MAX:
        return "black"
    if (blue >= BLUE_MIN and red <= BLUE_OTHER and green <= BLUE_OTHER and blue >= red + BLUE_DIFF and blue >= green + BLUE_DIFF):
        return "blue"
    if (green >= GREEN_MIN and red <= GREEN_OTHER and blue <= GREEN_OTHER and green >= red + GREEN_DIFF and green >= blue + GREEN_DIFF):
        return "green"
    if (BRN_R_MIN <= red <= BRN_R_MAX and BRN_G_MIN <= green <= BRN_G_MAX and BRN_B_MIN <= blue <= BRN_B_MAX and abs(red - green) <= BRN_RG and green >= blue + BRN_GB):
        return "brown"
    if (red >= YEL_R and green >= YEL_G and blue <= YEL_B and abs(red - green) <= YEL_RG):
        return "yellow"
    if (max(red, green, blue) - min(red, green, blue) <= SILV_DIFF and SILV_MIN <= (red + green + blue) / 3 <= SILV_MAX):
        return "silver"
    if (red >= PUR_R and green <= PUR_G and blue >= PUR_B and blue >= red + PUR_BR and blue >= green + PUR_BG):
        return "purple"
    if (red >= RED_MIN and green <= RED_OTHER and blue <= RED_OTHER and red >= green + RED_DIFF and red >= blue + RED_DIFF):
        return "red"
    return None

def blocked_floor_detected():
    name = classify_floor(*read_colour())
    return name is not None and FLOORS[name]["blocked"]

def blocked_cells():
    cells = set()
    for region in FLOORS.values():
        if region["blocked"]:
            cells |= region["cells"]
    return cells

# checks the next 12cm in that direction for known bad floor
def direction_has_blocked_floor(turn_amount):
    map_x, map_z, _, _ = gps_to_map()
    angle = math.radians(-normalize_heading(heading + turn_amount))
    blocked = blocked_cells()
    for distance in range(1, 13):
        cell = (map_x + round(distance * math.cos(angle)), map_z + round(distance * math.sin(angle)))
        if cell in blocked:
            return True
    return False

# paints the floor colour forward on the map till it hits a wall
def mark_floor_region(detected_x, detected_z, robot_heading, name):
    region = FLOORS[name]
    symbol, saved_cells = region["symbol"], region["cells"]
    forward_x, forward_z, side_x, side_z = {0:   (1, 0, 0, 1), 90:  (0, -1, 1, 0), 180: (-1, 0, 0, 1), 270: (0, 1, 1, 0),}[int(cardinal(robot_heading))]
    for side_offset in range(-4, 5):
        line_x = detected_x + FLOOR_OFFSET * forward_x + side_offset * side_x
        line_z = detected_z + FLOOR_OFFSET * forward_z + side_offset * side_z
        for forward_offset in range(15):
            x = line_x + forward_offset * forward_x
            z = line_z + forward_offset * forward_z
            if not in_map(x, z):
                break
            cell = grid[z][x]
            if cell == ".":
                saved_cells.add((x, z))
                grid[z][x] = symbol
            elif cell == symbol:
                saved_cells.add((x, z))
            else:
                break
    print(f"Known {region['hazard']} cells:", len(saved_cells))

#lidar (only layer 2 works properly)
def get_layer_2():
    return lidar.getRangeImage()[2 * resolution:3 * resolution]

# closest reading around that ray
def sector_distance(centre_index, width=15):
    layer = get_layer_2()
    values = [layer[(centre_index + offset) % resolution] for offset in range(-width, width + 1)]
    values = [v for v in values if math.isfinite(v)]
    return min(values) if values else float("inf")

def get_front_lidar():
    return sector_distance(0)

def get_right_lidar():
    return sector_distance(RIGHT_IDX)

def get_back_lidar():
    return sector_distance(resolution // 2)

def get_left_lidar():
    return sector_distance(LEFT_IDX)

def get_left_turn_clearance():
    return sector_distance(LEFT_IDX, width=5)

def side_wall_distance(camera):
    return sector_distance(camera["ray"], width=2)

#mapping - free space along each ray and a wall where it ends
def scan_map(map_x, map_z, robot_heading, mark_scanned=False):
    layer = get_layer_2()
    heading_radians = math.radians(-robot_heading)
    special_floor = set().union(*(region["cells"] for region in FLOORS.values()))
    for index in range(resolution):
        distance = layer[index]
        if not math.isfinite(distance):
            continue
        angle = heading_radians + index * 2 * math.pi / resolution
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        ray = 0.0
        while ray < distance - CELL:
            x = map_x + round(ray * cos_a / CELL)
            z = map_z + round(ray * sin_a / CELL)
            if in_map(x, z) and grid[z][x] == "?":
                grid[z][x] = "."
            ray += CELL / 2
        x = map_x + round(distance * cos_a / CELL)
        z = map_z + round(distance * sin_a / CELL)
        if in_map(x, z):
            if (x, z) not in special_floor:
                grid[z][x] = "#"
            if mark_scanned and distance <= SCAN_DIST and index in (RIGHT_IDX, LEFT_IDX):
                side_walls.add((x, z))
    for region in FLOORS.values():
        for x, z in region["cells"]:
            if in_map(x, z):
                grid[z][x] = region["symbol"]

#navigation - left hand rule: left, front, right, back
def choose_boundary_direction():
    clearances = {"Left":  (get_left_turn_clearance(), SIDE_CLEAR, 90), "Front": (get_front_lidar(), FRONT_CLEAR, 0), "Right": (get_right_lidar(), SIDE_CLEAR, -90), "Back":  (get_back_lidar(), SIDE_CLEAR, 180),}
    safe = {}
    print()
    print("Boundary-following decision")
    print("Current heading:", round(normalize_heading(heading), 2))
    for name, (clearance, needed, turn) in clearances.items():
        has_hole = direction_has_blocked_floor(turn)
        safe[name] = clearance >= needed and not has_hole
        print(f"{name}:", round(clearance, 4), "m | hole:", has_hole, "| safe:", safe[name])
    for name, decision, text in (("Left", 90, "turn left"), ("Front", 0, "continue straight"), ("Right", -90, "turn right"), ("Back", -90, "turn around")):
        if safe[name]:
            print(f"Decision: {text}.")
            return decision  # back turns right first, next decision does the rest
    print("Decision: no safe direction.")
    return None

def start_new_step():
    global state, target_heading, start_l, start_r
    start_l = left_encoder.getValue()
    start_r = right_encoder.getValue()
    target_heading = cardinal(heading)
    state = "MOVE"

def start_turn(turn_amount):
    global state, target_heading
    current = cardinal(heading)
    target_heading = normalize_heading(current + turn_amount)
    print("Starting turn:", round(current, 2), "degrees to", round(target_heading, 2), "degrees")
    state = "TURN"

def begin_floor_avoidance(average_change, hazard_name):
    global state, rev_target, rev_l, rev_r
    global turn_after, dodging_hole, detour_steps, last_hazard
    set_wheels(0, 0)
    last_hazard = hazard_name
    rev_target = max(average_change, 0.5)
    rev_l = left_encoder.getValue()
    rev_r = right_encoder.getValue()
    turn_after = -90
    dodging_hole = True
    detour_steps = 0
    state = "REVERSE"

#printing the map
def print_map(robot_x, robot_z):
    display = [row.copy() for row in grid]
    for x, z in side_walls:
        if in_map(x, z) and display[z][x] == "#":
            display[z][x] = "W"
    for region in FLOORS.values():
        for x, z in region["cells"]:
            if in_map(x, z):
                display[z][x] = region["symbol"]
    display[START_Z][START_X] = "S"
    if in_map(robot_x, robot_z):
        display[robot_z][robot_x] = "R"
    known = [(r, c) for r in range(MAPSIZE) for c in range(MAPSIZE) if display[r][c] != "?"]
    if not known:
        print("No map data was recorded.")
        return
    top = max(0, min(r for r, _ in known) - 2)
    bottom = min(MAPSIZE - 1, max(r for r, _ in known) + 2)
    left = max(0, min(c for _, c in known) - 2)
    right = min(MAPSIZE - 1, max(c for _, c in known) + 2)
    print()
    print("========================================")
    print("HAZARD-AWARE BOUNDARY MAP")
    print("? unknown | . free | # wall | W scanned wall | S start | R robot")
    print("H black hole | M brown floor | C checkpoint")
    print("B blue | Y yellow | G green | P purple | r red passage")
    print("========================================")
    for row in range(top, bottom + 1):
        print("".join(display[row][left:right + 1]))
    print("========================================")

def finish(reason):
    global state
    set_wheels(0, 0)
    final_x, final_z, _, _ = gps_to_map()
    scan_map(final_x, final_z, heading)
    print()
    print("Finished:", reason)
    print("Straight passes completed:", passes)
    print("Steps completed in current pass:", pass_steps)
    print("Total 6 cm steps:", total_steps)
    print("Final heading:", round(heading, 2), "degrees")
    print("Scanned wall cells:", len(side_walls))
    for region in FLOORS.values():
        print(f"{region['hazard'].capitalize()} cells:", len(region["cells"]))
    reported = [t for t in seen_tokens if t["status"] == "reported"]
    print("Wall tokens reported:", len(reported), [t["type"] for t in reported])
    print("Wall tokens ignored:", sum(1 for t in seen_tokens if t["status"] != "reported"))
    print_map(final_x, final_z)
    state = "FINISHED"

#vision - erebus colours are flat so simple thresholds work fine
def get_frame(camera):
    device = camera["device"]
    raw = device.getImage()
    if raw is None:
        return None
    return np.frombuffer(raw, np.uint8).reshape((device.getHeight(), device.getWidth(), 4))[:, :, :3]

# 0 other 1 black 2 red 3 yellow 4 green 5 blue 6 white
def colour_classes(bgr):
    b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
    brightest = np.maximum(np.maximum(b, g), r)
    spread = brightest - np.minimum(np.minimum(b, g), r)
    labels = np.zeros(b.shape, np.uint8)
    teal_tint = (g - r > 10) & (b - r > 10)  # wall shadow is dark but a bit teal, real black isnt
    labels[(brightest <= 60) & (spread <= 16) & ~teal_tint] = 1
    labels[(np.minimum(np.minimum(b, g), r) >= 140) & (spread <= 40)] = 6
    vivid = (brightest >= 120) & (spread >= 90)
    half = brightest // 2
    high_r, high_g, high_b = r > half, g > half, b > half
    labels[vivid & high_r & ~high_g & ~high_b] = 2
    labels[vivid & high_r & high_g & ~high_b] = 3
    labels[vivid & ~high_r & high_g & ~high_b] = 4
    labels[vivid & ~high_r & ~high_g & high_b] = 5
    return labels

#cognitive targets
def fit_circle(points):
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    (c0, c1, c2), *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), x * x + y * y, rcond=None)
    cx, cy = c0 / 2, c1 / 2
    return cx, cy, math.sqrt(max(c2 + cx * cx + cy * cy, 0.0))

def ring_component(labels, min_area=60):
    ring = ((labels >= 1) & (labels <= 5)).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(ring, connectivity=4)
    best = None
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        blob = components == i
        values = labels[blob]
        if not np.any((values >= 2) & (values <= 5)):
            continue
        if best is None or stats[i, cv2.CC_STAT_AREA] > best[1]:
            best = (blob, stats[i, cv2.CC_STAT_AREA])
    return None if best is None else best[0]

# pixels where the colour changes
def boundary_points(labels, blob):
    code = np.where((labels >= 1) & (labels <= 5), labels, 0).astype(np.int16)
    h, w = code.shape
    edge = np.zeros((h, w), bool)
    outer = np.zeros((h, w), bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neighbour = np.full((h, w), -1, np.int16)
        neighbour[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)] = code[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
        edge |= blob & (neighbour >= 0) & (neighbour != code)
        outer |= blob & (neighbour == 0)
    ys, xs = np.nonzero(edge)
    return np.column_stack([xs, ys]).astype(np.float64), outer[ys, xs]

def circle_from_3(points):
    (x1, y1), (x2, y2), (x3, y3) = points
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-9:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return cx, cy, math.hypot(x1 - cx, y1 - cy)

# ransac, every ring edge has the same centre so any edge works
def best_circle(points, max_radius, iterations=250, tolerance=1.0):
    n = len(points)
    if n < 20:
        return None
    rng = np.random.default_rng(0)
    best, best_count = None, 0
    for _ in range(iterations):
        circle = circle_from_3(points[rng.choice(n, 3, replace=False)])
        if circle is None or not 4 <= circle[2] <= max_radius:
            continue
        count = int(np.sum(np.abs(np.hypot(points[:, 0] - circle[0], points[:, 1] - circle[1]) - circle[2]) <= tolerance))
        if count > best_count:
            best, best_count = circle, count
    if best is None:
        return None
    for _ in range(2):
        inliers = np.abs(np.hypot(points[:, 0] - best[0], points[:, 1] - best[1]) - best[2]) <= 1.5 * tolerance
        if inliers.sum() < 12:
            return None
        best = fit_circle(points[inliers])
    inliers = np.abs(np.hypot(points[:, 0] - best[0], points[:, 1] - best[1]) - best[2]) <= 1.5 * tolerance
    angles = np.arctan2(points[inliers, 1] - best[1], points[inliers, 0] - best[0])
    span = np.unique(((angles + math.pi) / (2 * math.pi) * 72).astype(int) % 72).size * 5
    return best, int(inliers.sum()), span

# works even when the target is bigger then the frame, centre can be off screen
def detect_cognitive(labels, expected_radius, min_band_samples=6):
    h, w = labels.shape
    blob = ring_component(labels)
    if blob is None:
        return None
    points, is_outer = boundary_points(labels, blob)
    max_radius = 3.0 * expected_radius
    found = best_circle(points, max_radius)
    if found is None:
        return None
    (cx, cy, _), inliers, span = found
    if inliers < 20 or span < 45:
        return None
    distance = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    outer_distance = distance[is_outer]
    outer_fit = (len(outer_distance) >= 12 and np.median(np.abs(outer_distance - np.median(outer_distance))) <= max(1.0, 0.04 * np.median(outer_distance)))
    if outer_fit:
        radius = float(np.median(outer_distance))  # outer edge is visible
    else:
        ys, xs = np.nonzero(blob)
        lower = float(np.percentile(np.hypot(xs - cx, ys - cy), 99)) - 1
        wall_ys, wall_xs = np.nonzero(~((labels >= 1) & (labels <= 5)))
        upper = (float(np.percentile(np.hypot(wall_xs - cx, wall_ys - cy), 1)) + 1) if len(wall_xs) else max_radius
        inner_distance = distance[~is_outer]
        if len(inner_distance) < 10 or min(upper, max_radius) <= max(lower, 4):
            return None
        best_cost, radius = None, None
        for candidate in np.arange(max(lower, 4), min(upper, max_radius), 0.25):
            error = np.min(np.abs(inner_distance[:, None] - candidate * np.arange(1, 5)[None, :] / 5), axis=1)
            cost = float(np.mean(np.minimum(error, 3.0))) + 2.0 * abs(math.log(candidate / expected_radius))
            if best_cost is None or cost < best_cost:
                best_cost, radius = cost, float(candidate)
        if radius is None:
            return None
    ring_index = np.clip(np.round(distance / (radius / 5)), 1, 5)
    if float(np.median(np.abs(distance - ring_index * radius / 5))) > max(1.0, 0.03 * radius):
        return None
    bands = []
    for band in range(5):
        votes, other = Counter(), 0
        for fraction in (0.3, 0.5, 0.7):
            sample_radius = (band + fraction) * radius / 5
            count = max(8, int(2 * math.pi * sample_radius / 1.5))
            angle = np.linspace(0, 2 * math.pi, count, endpoint=False)
            px = np.round(cx + sample_radius * np.cos(angle)).astype(int)
            py = np.round(cy + sample_radius * np.sin(angle)).astype(int)
            inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            for value in labels[py[inside], px[inside]]:
                if value in LABEL_COLOURS:
                    votes[LABEL_COLOURS[value]] += 1
                else:
                    other += 1
        total = sum(votes.values())
        bands.append(votes if total >= min_band_samples and total >= 4 * other else Counter())
    colours = {votes.most_common(1)[0][0] for votes in bands if votes}
    if len(colours) < 2:
        return None
    return {"kind": "cognitive", "cx": cx, "cy": cy, "size": radius, "area": int(blob.sum()), "bands": bands, "outer_fit": outer_fit, "type": None}

# ring colours added up over lots of frames
def decide_rings(bands):
    rings = []
    for votes in bands:
        total = sum(votes.values())
        if total < MIN_VOTES:
            return None
        colour, count = votes.most_common(1)[0]
        if count / total < MIN_SHARE:
            return None
        rings.append(colour)
    return rings

# 'fake' if the sum is wrong, 'flat' if its all one colour
def cognitive_type(bands):
    rings = decide_rings(bands)
    if rings is None:
        return None
    if len(set(rings)) == 1:
        return "flat"
    return HAZARDS.get(sum(RING_VALS[c] for c in rings), "fake")

def add_bands(total, bands):
    for band, votes in enumerate(bands):
        total[band] += votes

def band_summary(bands):
    return [votes.most_common(1)[0][0] if votes else "?" for votes in bands]

#victims - merges all the white bits and reads the letter inside
def find_plaque(labels, near_wall):
    h, w = labels.shape
    white = (labels == 6).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    pieces = np.zeros((h, w), np.uint8)
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < 4:
            continue
        if not near_wall and (area > 0.6 * w * h or (y + bh >= h and bh > 0.4 * h)):
            continue
        pieces[components == i] = 1
    if pieces.sum() < 20:
        return None
    ys, xs = np.nonzero(pieces)
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    outline = cv2.fillConvexPoly(np.zeros((h, w), np.uint8), hull, 1)
    glyph = ((outline > 0) & (labels == 1)).astype(np.uint8)
    glyph_pixels = int(glyph.sum())
    if glyph_pixels < 12 or glyph_pixels > 0.7 * outline.sum():
        return None
    gy, gx = np.nonzero(glyph)
    box = (int(gx.min()), int(gy.min()), int(gx.max() - gx.min() + 1), int(gy.max() - gy.min() + 1))
    if not 0.3 <= box[2] / box[3] <= 3.0:
        return None
    return glyph, box

# phi has 2 holes, omega has one big gap, psi has two equal ones
def classify_glyph(glyph):
    area = int(np.sum(glyph))
    if area < 12:
        return None, {}
    contours, hierarchy = cv2.findContours(glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = 0
    for i, info in enumerate(hierarchy[0]):
        if info[3] >= 0:
            hole = cv2.drawContours(np.zeros_like(glyph), contours, i, 1, -1)
            if np.sum(hole & (1 - glyph)) >= max(3, 0.05 * area):
                holes += 1
    hull = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in contours]))
    hull_mask = cv2.fillConvexPoly(np.zeros_like(glyph), hull, 1)
    hull_area = max(int(np.sum(hull_mask)), 1)
    gaps = (hull_mask & (1 - glyph)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=4)
    bays = sorted((stats[i, cv2.CC_STAT_AREA] / hull_area for i in range(1, count)), reverse=True) + [0.0, 0.0]
    big, second = bays[0], bays[1]
    features = {"holes": holes, "bay1": round(big, 2), "bay2": round(second, 2)}
    omega_like = big >= 0.22 and second < 0.5 * big
    if holes >= 2:
        letter = "H"
    elif holes == 1:
        letter = "U" if omega_like else "H"
    elif omega_like:
        letter = "U"
    elif big < 0.07:
        letter = "H"
    elif second >= 0.5 * big:
        letter = "S"
    else:
        letter = "U" if big >= 0.2 else "S"
    return letter, features

# only reads it if the whole letter is in the pic
def detect_victim(labels, near_wall, min_read_side=11):
    found = find_plaque(labels, near_wall)
    if found is None:
        return None
    glyph, (x, y, bw, bh) = found
    h, w = labels.shape
    sighting = {"kind": "victim", "cx": x + bw / 2, "cy": y + bh / 2, "size": max(bw, bh), "area": int(glyph.sum()) * 3, "type": None, "features": None}
    fully_visible = x > 0 and y > 0 and x + bw < w and y + bh < h
    if fully_visible and max(bw, bh) >= min_read_side:
        sighting["type"], sighting["features"] = classify_glyph(glyph)
    return sighting

#camera maths
def focal_length(camera):
    device = camera["device"]
    return (device.getWidth() / 2) / math.tan(device.getFov() / 2)

def expected_target_radius(camera, wall):
    if not math.isfinite(wall):
        wall = MAX_WALL
    return focal_length(camera) * TARGET_R / max(wall - camera["lens_offset"], 0.008)

# best thing this camera can see right now
def look(camera, kind=None):
    frame = get_frame(camera)
    if frame is None:
        return None
    labels = colour_classes(frame)
    candidates = []
    wall = side_wall_distance(camera)
    if kind in (None, "cognitive"):
        sighting = detect_cognitive(labels, expected_target_radius(camera, wall))
        if sighting and sighting["outer_fit"] and math.isfinite(wall) and wall < MAX_WALL:
            measured = wall - focal_length(camera) * TARGET_R / sighting["size"]
            if 0.0 <= measured <= 0.05:
                # whole edge visible so we can tell where the lens is
                camera["lens_offset"] += 0.2 * (measured - camera["lens_offset"])
        candidates.append(sighting)
    if kind in (None, "victim"):
        candidates.append(detect_victim(labels, near_wall=math.isfinite(wall) and wall < 0.08))
    candidates = [c for c in candidates if c is not None]
    return max(candidates, key=lambda c: c["area"]) if candidates else None

def describe(sighting):
    if sighting is None:
        return "nothing"
    text = (f"{sighting['kind']} {'radius' if sighting['kind'] == 'cognitive' else 'size'}={sighting['size']:.1f}px " f"centre=({sighting['cx']:.0f},{sighting['cy']:.0f}) type={sighting['type']}")
    if sighting["kind"] == "cognitive":
        text += f" visible rings={band_summary(sighting['bands'])}"
    if sighting["kind"] == "victim" and sighting["features"]:
        text += f" shape={sighting['features']}"
    return text

#token bookkeeping
def image_error(camera, sighting):
    return sighting["cx"] - camera["device"].getWidth() / 2

# rough world position, just so we dont do the same one twice
def estimate_token_position(camera, wall_distance, error_px):
    camera_to_wall = max(wall_distance - camera["lens_offset"], 0.005)
    ahead = camera["creep_sign"] * error_px / focal_length(camera) * camera_to_wall
    angle = math.radians(get_absolute_imu_heading())
    forward_x, forward_z = -math.sin(angle), -math.cos(angle)
    right_x, right_z = math.cos(angle), -math.sin(angle)
    position = gps.getValues()
    return (position[0] + forward_x * ahead + camera["side"] * right_x * wall_distance, position[2] + forward_z * ahead + camera["side"] * right_z * wall_distance)

def find_known_token(position):
    for entry in seen_tokens:
        if math.hypot(entry["x"] - position[0], entry["z"] - position[1]) < SAME_TOKEN_DIST:
            return entry
    return None

def token_is_handled(position):
    entry = find_known_token(position)
    return entry is not None and (entry["status"] in ("reported", "ignored") or entry["attempts"] >= MAX_TRIES)

def remember_token(position, status, token_type=None):
    entry = find_known_token(position)
    if entry is None:
        entry = {"x": position[0], "z": position[1], "status": status, "attempts": 0, "type": None}
        seen_tokens.append(entry)
    entry["status"] = status
    entry["type"] = token_type or entry["type"]
    if status == "failed":
        entry["attempts"] += 1

# saves both cameras to the pics folder
def save_frames():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), PIC_FOLDER)
    os.makedirs(folder, exist_ok=True)
    for camera in cams:
        frame = get_frame(camera)
        if frame is not None:
            path = os.path.join(folder, f"step_{total_steps:03d}_{camera['name']}.png")
            cv2.imwrite(path, frame)
    print("Camera frames saved to", folder)

def print_vision_summary():
    print("VISION CHECK after step", total_steps)
    if SAVE_PICS:
        save_frames()
    for camera in cams:
        sighting = look(camera)
        wall = side_wall_distance(camera)
        note = ""
        if sighting is not None:
            if not math.isfinite(wall) or wall > MAX_WALL:
                note = " | wall too far, ignored"
            elif token_is_handled(estimate_token_position(camera, wall, image_error(camera, sighting))):
                note = " | already handled"
        print(f"  {camera['name']:14s} | wall {wall:.3f} m | {describe(sighting)}{note}")

#token states
def learn_creep_direction(camera, image_shift):
    if abs(image_shift) < 0.5:
        return
    camera["creep_sign"] = -1 if image_shift > 0 else 1
    camera["calibrated"] = True
    print(f"Calibration: {camera['name']} creep direction {camera['creep_sign']:+d} " f"(token moved {image_shift:+.1f} px while driving forward)")

# starts on the first new token either camera sees
def look_for_new_token(resume_state):
    for camera in cams:
        sighting = look(camera)
        previous, camera["track"] = camera.get("track"), None
        track_bands = camera.get("track_bands")
        camera["track_bands"] = None
        if sighting is None:
            continue
        wall = side_wall_distance(camera)
        if not math.isfinite(wall) or wall > MAX_WALL:
            continue
        position = estimate_token_position(camera, wall, image_error(camera, sighting))
        if token_is_handled(position):
            continue
        if sighting["kind"] == "cognitive":
            if track_bands is None or previous is None:
                track_bands = [Counter() for _ in range(5)]
            add_bands(track_bands, sighting["bands"])
            camera["track_bands"] = track_bands
        if resume_state == "MOVE":
            camera["track"] = (sighting["cx"], average_encoder())
            if not camera["calibrated"]:
                if previous is None or average_encoder() - previous[1] < 0.03:
                    continue
                learn_creep_direction(camera, sighting["cx"] - previous[0])
            if abs(image_error(camera, sighting)) > TRIGGER_PX:
                continue
        begin_token(camera, sighting, position, resume_state, track_bands)
        camera["track"] = camera["track_bands"] = None
        return True
    return False

def begin_token(camera, sighting, position, resume_state, track_bands=None):
    global state, tok
    set_wheels(0, 0)
    bands = [Counter() for _ in range(5)]
    if sighting["kind"] == "cognitive":
        add_bands(bands, track_bands or sighting["bands"])
    tok = {"bands": bands, "scanned": False, "scan_targets": [], "after_scan": "", "read_type": None, "camera": camera, "kind": sighting["kind"], "position": position, "resume_state": resume_state, "move_progress": ((left_encoder.getValue() - start_l) + (right_encoder.getValue() - start_r)) / 2, "pause_encoder": average_encoder(), "base_heading": cardinal(heading), "align_start": average_encoder(), "segment_direction": None, "segment_start": 0.0, "segment_error": 0.0, "lost_frames": 0, "last_drive": 0, "recover_drive": 0, "settle": 0, "approached": False, "approach_start": 0.0, "approach_distance": 0.0, "align_votes": Counter(), "votes": Counter(), "report_start": 0.0, "hold": 0,}
    print()
    print("========================================")
    print("NEW WALL TOKEN SEEN")
    print("Camera:", camera["name"])
    print("Sighting:", describe(sighting))
    print("Side wall distance:", round(side_wall_distance(camera), 4), "m")
    print("Image offset:", round(image_error(camera, sighting), 1), "px")
    print("Paused navigation state:", resume_state)
    if sighting["kind"] == "cognitive":
        print("Rings seen so far:", band_summary(bands), "| votes:", [sum(v.values()) for v in bands])
    print("========================================")
    state = "TOKEN_ALIGN"

# give up (can retry once)
def abort_token(reason):
    set_wheels(0, 0)
    remember_token(tok["position"], "failed")
    print()
    print("TOKEN ABORTED:", reason)
    end_token()

def end_token():
    global state
    if tok["approached"] and tok["approach_distance"] > 0:
        state = "TOKEN_RETURN_TURN_IN"
    else:
        resume_navigation()

# goes back to driving, counts any creeping as part of the step
def resume_navigation():
    global state, tok, wait_steps, start_l, start_r
    set_wheels(0, 0)
    state = tok["resume_state"]
    if state == "MOVE":
        net = average_encoder() - tok["pause_encoder"]
        progress = max(0.0, tok["move_progress"] + net)
        start_l = left_encoder.getValue() - progress
        start_r = right_encoder.getValue() - progress
    elif state == "AFTER_STEP":
        wait_steps = 0
    print("Resuming navigation state:", state)
    tok = None

# slow move along the wall, returns a reason if its not safe
def creep(direction, speed):
    if abs(average_encoder() - tok["align_start"]) > ALIGN_MAX:
        return "token could not be centred within the travel limit"
    if direction > 0 and (blocked_floor_detected() or get_front_lidar() < STOP_DIST):
        return "blocked ahead while aligning"
    if direction < 0 and get_back_lidar() < STOP_DIST:
        return "blocked behind while aligning"
    tok["last_drive"] = direction
    if direction > 0:
        drive_straight(speed, tok["base_heading"])
    else:
        set_wheels(-speed, -speed)
    return None

# sends robot gps in cm + the letter
def send_report(token_type):
    if emitter is None:
        print("REPORT FAILED: emitter device is unavailable")
        return False
    position = gps.getValues()
    x_cm, z_cm = round(position[0] * 100), round(position[2] * 100)
    emitter.send(struct.pack("i i c", x_cm, z_cm, token_type.encode("utf-8")))
    print()
    print("========================================")
    print("SUPERVISOR REPORT SENT")
    print("Kind:", tok["kind"], "| Code:", token_type)
    print("Camera:", tok["camera"]["name"])
    print("Robot GPS cm:", (x_cm, z_cm))
    print("Side wall distance:", round(side_wall_distance(tok["camera"]), 4), "m")
    print_reading()
    print("========================================")
    return True

def collect_reading(sighting):
    if sighting is None:
        return
    if sighting["kind"] == "cognitive":
        add_bands(tok["bands"], sighting["bands"])
    elif sighting["type"]:
        tok["votes"][sighting["type"]] += 1

def current_type():
    if tok["kind"] == "cognitive":
        return cognitive_type(tok["bands"])
    votes = tok["votes"] + tok["align_votes"]
    return votes.most_common(1)[0][0] if votes else None

def print_reading():
    if tok["kind"] == "cognitive":
        print("Rings:", band_summary(tok["bands"]), "| votes per ring:", [sum(v.values()) for v in tok["bands"]], "| reading:", current_type())
    else:
        print("Letter votes:", dict(tok["votes"] + tok["align_votes"]), "| reading:", current_type())

# creeps a bit each way to see more rings
def start_scan(after):
    global state
    here = average_encoder()
    sign = tok["camera"]["creep_sign"]
    tok.update(scanned=True, after_scan=after, scan_targets=[here + SCAN_MOVE * sign, here - SCAN_MOVE * sign, here])
    print("Reading unclear: scanning along the wall to see more of the token.")
    state = "TOKEN_SCAN"

# gives back the letter to send or deals with fake/unclear ones
def resolve_reading(after):
    reading = current_type()
    print_reading()
    if reading is None:
        if not tok["scanned"]:
            start_scan(after)
        else:
            abort_token("token could not be read")
        return None
    if reading in ("fake", "flat"):
        print("Not a valid hazard (" + reading + "): not reported.")
        remember_token(tok["position"], "ignored", reading)
        end_token()
        return None
    return reading

def send_and_hold(code):
    global state
    if send_report(code):
        remember_token(tok["position"], "reported", code)
        tok["hold"] = HOLD_STEPS
        state = "TOKEN_HOLD"
    else:
        abort_token("report could not be sent")

def finish_report():
    print()
    print("REPORT WINDOW COMPLETE | kind:", tok["kind"])
    code = resolve_reading("TOKEN_MEASURE")
    if code:
        send_and_hold(code)
#main loop
print("Hazard-aware left-boundary wall-following controller started. [token controller v4]")
print("Camera FOV (rad):", round(camera1.getFov(), 3), round(camera2.getFov(), 3))
while robot.step(timestep) != -1:
    if start_heading is not None:
        heading = get_relative_heading()
    # first scan
    if state == "WAIT":
        set_wheels(0, 0)
        if not imu_is_ready():
            continue
        wait_steps -= 1
        if wait_steps <= 0:
            start_heading = get_absolute_imu_heading()
            heading = 0.0
            position = gps.getValues()
            start_gps = (position[0], position[2])
            print("Absolute starting IMU heading:", round(start_heading, 2), "degrees")
            print("Initial left wall distance:", round(get_left_lidar(), 4), "m")
            print("Initial right wall distance:", round(get_right_lidar(), 4), "m")
            scan_map(START_X, START_Z, heading, mark_scanned=True)
            print("Initial scan complete. Beginning first straight pass.")
            pass_steps = 0
            start_new_step()
    # drive one step
    elif state == "MOVE":
        average_change = (abs(left_encoder.getValue() - start_l) + abs(right_encoder.getValue() - start_r)) / 2
        rgb = read_colour()
        floor = classify_floor(*rgb)
        if floor == "silver":
            cell = gps_to_map()[:2]
            if cell not in FLOORS["silver"]["cells"]:
                FLOORS["silver"]["cells"].add(cell)
                print()
                print("SILVER CHECKPOINT DETECTED | RGB:", rgb, "| map position:", cell)
        elif floor is not None and FLOORS[floor]["blocked"]:
            set_wheels(0, 0)
            map_x, map_z, _, _ = gps_to_map()
            mark_floor_region(map_x, map_z, heading, floor)
            print()
            print(f"{FLOORS[floor]['log']} DETECTED | RGB:", rgb)
            begin_floor_avoidance(average_change, FLOORS[floor]["hazard"])
            continue
        front_clearance = get_front_lidar()
        if front_clearance < STOP_DIST:
            set_wheels(0, 0)
            print()
            print("Front wall reached. Front clearance:", round(front_clearance, 4), "m")
            state = "END_PASS"
            continue
        if average_change >= STEP_ENC:
            set_wheels(0, 0)
            pass_steps += 1
            total_steps += 1
            wait_steps = 8
            state = "AFTER_STEP"
            continue
        if look_for_new_token("MOVE"):
            continue
        map_x, map_z, _, _ = gps_to_map()
        scan_map(map_x, map_z, heading, mark_scanned=True)
        drive_straight(SPEED, target_heading)
    # stopped after a step, decide where to go
    elif state == "AFTER_STEP":
        set_wheels(0, 0)
        wait_steps -= 1
        if wait_steps > 0:
            continue
        if last_summary != total_steps:
            last_summary = total_steps
            print_vision_summary()
        if look_for_new_token("AFTER_STEP"):
            continue
        map_x, map_z, _, _ = gps_to_map()
        scan_map(map_x, map_z, heading)
        front_clearance = get_front_lidar()
        left_clearance = get_left_turn_clearance()
        print("Total Step", total_steps, "| Current pass step", pass_steps, "complete. Front clearance:", round(front_clearance, 4), "m")
        if total_steps >= MAX_STEPS:
            finish("temporary total-step limit reached")
            continue
        if dodging_hole:
            detour_steps += 1
            print("Black-hole detour step:", detour_steps, "of at least", MIN_STEPS_LEFT)
            if detour_steps < MIN_STEPS_LEFT:
                print("Continuing straight to clear the black hole.")
                start_new_step()
                continue
            left_has_hole = direction_has_blocked_floor(90)
            print("Left route contains known black hole:", left_has_hole)
            if left_clearance >= SIDE_CLEAR and not left_has_hole:
                print("Safe left opening found after clearing the black hole.")
                dodging_hole = False
                detour_steps = 0
                passes += 1
                pass_steps = 0
                start_turn(90)
                continue
            if front_clearance >= FRONT_CLEAR and not direction_has_blocked_floor(0):
                print("Left is not safe yet. Continuing detour forward.")
                start_new_step()
                continue
            print("Cannot continue detour forward.")
            state = "END_PASS"
            continue
        action = choose_boundary_direction()
        if action is None:
            finish("no safe boundary direction")
        elif action == 0:
            start_new_step()
        else:
            passes += 1
            pass_steps = 0
            start_turn(action)
    # back off a hole or passage
    elif state == "REVERSE":
        reversed_by = (abs(left_encoder.getValue() - rev_l) + abs(right_encoder.getValue() - rev_r)) / 2
        if reversed_by < rev_target:
            set_wheels(-BACK_SPEED, -BACK_SPEED)
            continue
        set_wheels(0, 0)
        map_x, map_z, _, _ = gps_to_map()
        scan_map(map_x, map_z, heading)
        print("Reverse from", last_hazard, "complete.")
        if turn_after is not None:
            saved_turn, turn_after = turn_after, None
            print("Turning right to avoid", last_hazard, ".")
            passes += 1
            pass_steps = 0
            start_turn(saved_turn)
        else:
            state = "END_PASS"
    # end of a straight bit
    elif state == "END_PASS":
        set_wheels(0, 0)
        passes += 1
        map_x, map_z, _, _ = gps_to_map()
        scan_map(map_x, map_z, heading)
        print()
        print("Straight pass", passes, "complete.")
        action = choose_boundary_direction()
        if action is None:
            finish("no safe direction after straight pass")
        elif action == 0:
            print("Forward remains safe. Continuing straight.")
            pass_steps = 0
            start_new_step()
        else:
            pass_steps = 0
            start_turn(action)
    elif state == "TURN":
        if turn_toward(target_heading):
            wait_steps = 10
            state = "AFTER_TURN"
    elif state == "AFTER_TURN":
        set_wheels(0, 0)
        wait_steps -= 1
        if wait_steps > 0:
            continue
        map_x, map_z, _, _ = gps_to_map()
        scan_map(map_x, map_z, heading)
        front_clearance = get_front_lidar()
        print()
        print("Turn complete. New heading:", round(heading, 2))
        print("New front clearance:", round(front_clearance, 4), "m")
        # just turned right so the hole is beside us
        if dodging_hole and detour_steps == 0:
            if front_clearance < FRONT_CLEAR:
                print("Forced-right detour direction is physically blocked.")
                state = "END_PASS"
            else:
                print("Beginning forced-right black-hole detour.")
                pass_steps = 0
                start_new_step()
        else:
            front_has_hole = direction_has_blocked_floor(0)
            if front_clearance < FRONT_CLEAR or front_has_hole:
                print("New direction leads into known black floor." if front_has_hole else "New direction does not have enough clearance.")
                state = "END_PASS"
            else:
                print("Beginning next straight section.")
                pass_steps = 0
                start_new_step()
    # get the token in the middle of the camera
    elif state == "TOKEN_ALIGN":
        camera = tok["camera"]
        sighting = look(camera, tok["kind"])
        if sighting is None:
            tok["lost_frames"] += 1
            if not camera["calibrated"] and tok["last_drive"]:
                # pushed it out of view so we went the wrong way
                camera["creep_sign"] *= -1
                camera["calibrated"] = True
                tok["recover_drive"] = -tok["last_drive"]
                print(f"Calibration: token lost, {camera['name']} creep direction flipped " f"to {camera['creep_sign']:+d}; creeping back to find it")
            if tok["recover_drive"] and tok["lost_frames"] < 4 * LOST_FRAMES:
                problem = creep(tok["recover_drive"], CREEP_MIN * 2)
                if problem:
                    abort_token(problem)
                continue
            set_wheels(0, 0)
            if tok["lost_frames"] >= LOST_FRAMES:
                abort_token("token lost from view while aligning")
            continue
        tok["lost_frames"] = 0
        tok["recover_drive"] = 0
        error = image_error(camera, sighting)
        if sighting["kind"] == "cognitive":
            add_bands(tok["bands"], sighting["bands"])
        elif sighting["type"] and abs(error) <= 2 * ALIGN_TOL:
            tok["align_votes"][sighting["type"]] += 1
        if abs(error) <= ALIGN_TOL:
            set_wheels(0, 0)
            tok["settle"] = SETTLE
            print(f"Token centred in {camera['name']} (offset {error:.1f} px): {describe(sighting)}")
            state = "TOKEN_MEASURE"
            continue
        direction = camera["creep_sign"] * (1 if error > 0 else -1)
        if direction != tok["segment_direction"]:
            tok.update(segment_direction=direction, segment_start=average_encoder(), segment_error=abs(error))
        if not camera["calibrated"] and abs(average_encoder() - tok["segment_start"]) >= CALIB_MOVE:
            if abs(error) > tok["segment_error"] + 1:
                camera["creep_sign"] *= -1
                camera["calibrated"] = True
                tok["segment_direction"] = None
                set_wheels(0, 0)
                print(f"Calibration: {camera['name']} creep direction flipped to {camera['creep_sign']:+d}")
                continue
            if abs(error) < tok["segment_error"] - 1:
                camera["calibrated"] = True
                print(f"Calibration: {camera['name']} creep direction confirmed ({camera['creep_sign']:+d})")
        speed = max(CREEP_MIN, min(CREEP_MAX, ALIGN_KP * abs(error)))
        problem = creep(direction, speed)
        if problem:
            abort_token(problem)
    # close enough to report?
    elif state == "TOKEN_MEASURE":
        set_wheels(0, 0)
        tok["settle"] -= 1
        if tok["settle"] > 0:
            continue
        camera = tok["camera"]
        wall = side_wall_distance(camera)
        tok["position"] = estimate_token_position(camera, wall, 0.0)
        print("Side wall distance at token:", round(wall, 4), "m (report limit", REPORT_DIST, "m)")
        if wall <= REPORT_DIST:
            tok["report_start"] = robot.getTime()
            print(f"Holding still for {STOP_TIME} s and reading the token.")
            state = "TOKEN_REPORT"
        elif wall <= MAX_WALL:
            tok["report_start"] = robot.getTime()
            print("Too far from the wall to report here: reading the token before driving in.")
            state = "TOKEN_READ"
        else:
            abort_token("wall too far to report")
    # read it before driving closer
    elif state == "TOKEN_READ":
        set_wheels(0, 0)
        collect_reading(look(tok["camera"], tok["kind"]))
        if robot.getTime() - tok["report_start"] >= READ_TIME:
            code = resolve_reading("TOKEN_MEASURE")
            if code:
                tok["read_type"] = code
                print("Token read as", code, "- turning in to report from closer.")
                state = "TOKEN_TURN_IN"
    elif state == "TOKEN_SCAN":
        collect_reading(look(tok["camera"], tok["kind"]))
        goal = tok["scan_targets"][0]
        offset = goal - average_encoder()
        if abs(offset) < 0.05:
            set_wheels(0, 0)
            tok["scan_targets"].pop(0)
            if not tok["scan_targets"]:
                print("Scan complete.")
                tok["settle"] = SETTLE
                state = tok["after_scan"]
            continue
        problem = creep(1 if offset > 0 else -1, CREEP_MIN * 2)
        if problem:
            tok["scan_targets"] = [tok["scan_targets"][-1]]
            if abs(tok["scan_targets"][0] - average_encoder()) < 0.05:
                set_wheels(0, 0)
                tok["settle"] = SETTLE
                state = tok["after_scan"]
    # wall too far so turn and drive to it
    elif state == "TOKEN_TURN_IN":
        if turn_toward(normalize_heading(tok["base_heading"] + tok["camera"]["turn"])):
            tok["approach_start"] = average_encoder()
            tok["approached"] = True
            state = "TOKEN_DRIVE_IN"
    elif state == "TOKEN_DRIVE_IN":
        travelled = abs(average_encoder() - tok["approach_start"])
        front_clearance = get_front_lidar()
        stop_reason = None
        if blocked_floor_detected():
            stop_reason = "blocked floor ahead"
        elif front_clearance <= APPROACH_DIST:
            stop_reason = f"wall reached ({front_clearance:.3f} m)"
        elif travelled >= APPROACH_MAX:
            stop_reason = "approach travel limit"
        if stop_reason:
            set_wheels(0, 0)
            tok["approach_distance"] = travelled
            print("Approach stopped:", stop_reason, "| encoder distance:", round(travelled, 3))
            if front_clearance <= REPORT_DIST:
                tok["report_start"] = robot.getTime()
                print(f"Facing the token {front_clearance:.3f} m away: holding still for {STOP_TIME} s.")
                state = "TOKEN_REPORT_FACING"
            else:
                abort_token("could not get close enough to the wall")
        else:
            drive_straight(APPROACH_SPD, normalize_heading(tok["base_heading"] + tok["camera"]["turn"]))
    # it was centred before turning so its right in front
    elif state == "TOKEN_REPORT_FACING":
        set_wheels(0, 0)
        if robot.getTime() - tok["report_start"] >= STOP_TIME:
            send_and_hold(tok["read_type"])
    # stay still, read and send
    elif state == "TOKEN_REPORT":
        set_wheels(0, 0)
        sighting = look(tok["camera"], tok["kind"])
        if sighting and (sighting["kind"] == "cognitive" or abs(image_error(tok["camera"], sighting)) <= 2 * ALIGN_TOL):
            collect_reading(sighting)
        if robot.getTime() - tok["report_start"] >= STOP_TIME:
            finish_report()
    elif state == "TOKEN_HOLD":
        set_wheels(0, 0)
        tok["hold"] -= 1
        if tok["hold"] <= 0:
            end_token()
    # drive back out after going in
    elif state == "TOKEN_RETURN_TURN_IN":
        if turn_toward(normalize_heading(tok["base_heading"] + tok["camera"]["turn"])):
            tok["approach_start"] = average_encoder()
            state = "TOKEN_DRIVE_OUT"
    elif state == "TOKEN_DRIVE_OUT":
        if abs(average_encoder() - tok["approach_start"]) < tok["approach_distance"]:
            set_wheels(-APPROACH_SPD, -APPROACH_SPD)
        else:
            set_wheels(0, 0)
            state = "TOKEN_RETURN_TURN_BACK"
    elif state == "TOKEN_RETURN_TURN_BACK":
        if turn_toward(tok["base_heading"]):
            resume_navigation()
    elif state == "FINISHED":
        set_wheels(0, 0)
set_wheels(0, 0)
