"""
Strategy:
  1. Low-level safety layer: Stops the robot crashing into walls or pits
  2. High-level exploration layer: Keeps track of squares visited and uses that to decide where to go next
  3. Side-branch detection: Stops the robot jsut cruising past side branches and is always on the lookout for them
  4. Stuck / lack-of-progress recovery: Makes robot turn randomly if robot gets stuck for long time
"""

import math
import random
import struct
import numpy as np
import cv2
import typing
from collections import Counter, deque
from dataclasses import dataclass, field
from controller import Robot

# Config
TIME_STEP = 32
MAX_VELOCITY = 6.28

WALL_THRESHOLD = 0.05          # When wall will get detected
OPEN_THRESHOLD = 0.25          # How big a branch has to be for it to be considered open
PIT_THRESHOLD = 80             # What is considered black floor
REAR_WALL_THRESHOLD = 0.05     # What value to stop reversing at

CELL_SIZE = 0.06               # metres per grid cell
STUCK_TIME = 4.0                # When the robot will be considered stuck
STUCK_DIST = 0.02               # Any movement less than this is considered stuck
CELL_PROGRESS_TIMEOUT = 15.0     # How much seconds with no new cells visited, forces the recover func
GYRO_DEADBAND = 0.02             # rad/s - yaw rates below this are treated as noise/bias, not real turning

BRANCH_CHECK_COOLDOWN = 1.5     # How much time between side branch evaluations
BRANCH_SCORE_MARGIN = 1         # how much better a side branch's score must be to go to it
BRANCH_TURN_MS = 550            # how long it takes to turn toward a branch

TURN_SPEED_FAST = 0.6 * MAX_VELOCITY
NUDGE_DIFF = 0.15 * MAX_VELOCITY  # To centre easily in narrow gaps with small turns
TURN_SPEED_SLOW = -0.2 * MAX_VELOCITY
SPIN_SPEED = 0.6 * MAX_VELOCITY

# Victim / hazard-sign detection
FRONT_APPROACH_THRESHOLD = 0.08  # When to start the victim detection (how close you are to wall)
APPROACH_SLOWDOWN_THRESHOLD = 0.10  # When it thinks it is approaching a wall
APPROACH_SLOWDOWN_SPEED = 0.35 * MAX_VELOCITY  # How fast it goes while in the threshold of approaching wall
DETECTION_STABILIZE_MS = 250    # Fully stops robot for that much time
REPORT_DEDUPE_DIST = 0.20        # Skip reporting within this much cm to stop double detection
MIN_REPORT_INTERVAL = 3.0        # How much times report func can run in a second
STOP_BEFORE_REPORT_MS = 1300     # Have to stop for at least 1s to report
MIN_HAZARD_CONFIDENCE = 0.55     # a cognitive-target match below this confidence doesn't get reported

# Hazard/cognitive-target ring colours -> numeric values
RING_COLOR_VALUES = {"black": -2, "red": -1, "yellow": 0, "green": 1, "blue": 2}
HAZARD_SUM_TO_TYPE = {0: "F", 1: "P", 2: "C", 3: "O"}


# Robot / device setup
robot = Robot()


def list_all_devices():
    """Prints every device Webots has for this robot, for debugging puroposes"""
    print("=== Devices found on robot ===")
    for i in range(robot.getNumberOfDevices()):
        d = robot.getDeviceByIndex(i)
        print(f"  [{i}] {d.getName()!r}")
    print("==============================")


list_all_devices()
print("=== CONTROLLER VERSION: plaque-area-cap-v2 ===")


def _normalize(name):
    return name.lower().replace(" ", "").replace("_", "")


def get_device(*candidate_names):
    """Try each candidate and if it doesnt work use the normalise function to try every other capitalised and lower case version of it"""
    for name in candidate_names:
        d = robot.getDevice(name)
        if d is not None:
            return d

    normalized_candidates = [_normalize(c) for c in candidate_names]
    for i in range(robot.getNumberOfDevices()):
        d = robot.getDeviceByIndex(i)
        if _normalize(d.getName()) in normalized_candidates:
            return d

    print(f"WARNING: none of these device names were found: {candidate_names}")
    return None


wheel_right = get_device("wheel1", "wheel1 motor")
wheel_left = get_device("wheel2", "wheel2 motor")
wheel_left.setPosition(float("inf"))
wheel_right.setPosition(float("inf"))

ds1 = get_device("distance sensor1")
ds2 = get_device("distance sensor2")
ds3 = get_device("distance sensor3")
ds4 = get_device("distance sensor4")
ds5 = get_device("distance sensor5")   
ds6 = get_device("distance sensor6")   
ds7 = get_device("distance sensor7")   
ds8 = get_device("distance sensor8")   

front_sensors = [ds1, ds2, ds3, ds4]
rear_sensors = [ds7, ds8]
for s in front_sensors + [ds5, ds6] + rear_sensors:
    if s is not None:
        s.enable(TIME_STEP)

# Downward colour sensor
color_sensor = get_device("colour_sensor")
if color_sensor is not None:
    color_sensor.enable(TIME_STEP)

gps = get_device("gps")
gps.enable(TIME_STEP)

# LIDAR
lidar = get_device("lidar")
if lidar is not None:
    lidar.enable(TIME_STEP)
    LIDAR_FOV = lidar.getFov()
    LIDAR_RES = lidar.getHorizontalResolution()
else:
    LIDAR_FOV = None
    LIDAR_RES = None

# Gyro
gyro = get_device("gyro")
if gyro is not None:
    gyro.enable(TIME_STEP)

# Front camera - used for victim/hazard-sign detection ahead
camera = get_device("camera3", "camera")
CAM_WIDTH, CAM_HEIGHT = None, None
if camera is not None:
    camera.enable(TIME_STEP)
    CAM_WIDTH, CAM_HEIGHT = camera.getWidth(), camera.getHeight()

# Right camera - second camera, used to catch victims front camera doesnt catch
camera_right = get_device("camera1")
if camera_right is not None:
    camera_right.enable(TIME_STEP)

# Receiver picks up LOP notifications from the supervisor
receiver = get_device("receiver")
if receiver is not None:
    receiver.enable(TIME_STEP)

# Emitter repots victims/hazards to the supervisor.
emitter = get_device("emitter")
if emitter is None:
    print("!!! Victim/hazard reporting is DISABLED - no 'emitter' device found. "
          "Add an Emitter component in the robot customiser to enable this. !!!")

speeds = [MAX_VELOCITY, MAX_VELOCITY]

# Four cardinal directions
CARDINALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

visit_counts = {}          # cell, number of times physically visited
grid_walls = {}            # cell, set of (dx,dz) co-ords known to be blocked


def get_cell(pos):
    """Convert an (x, y, z) GPS reading into a grid cell coordinate."""
    return (round(pos[0] / CELL_SIZE), round(pos[2] / CELL_SIZE))


def mark_visited(pos):
    cell = get_cell(pos)
    visit_counts[cell] = visit_counts.get(cell, 0) + 1
    return cell


def snap_cardinal(deg):
    """Round a heading to the nearest of 0 or 90 or 180 or 270 degrees."""
    return (round(deg / 90.0) % 4) * 90


def cardinal_delta(deg):
    d = snap_cardinal(deg)
    return {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[d]


def mark_walls(cell, heading_deg, front_blocked, left_blocked, right_blocked):
    """Saves which directions the walls are at this cell"""
    walls = grid_walls.setdefault(cell, set())
    if front_blocked:
        walls.add(cardinal_delta(heading_deg))
    if left_blocked:
        walls.add(cardinal_delta(heading_deg + 90))
    if right_blocked:
        walls.add(cardinal_delta(heading_deg - 90))


def frontier_flood_score(start_cell, max_nodes=12):
    """
    Uses BFS logic to 'flood-like' find out how much unexplored squares are at each direction and generate a score based on that
    """
    seen = {start_cell}
    queue = [start_cell]
    score = 0
    while queue and len(seen) < max_nodes:
        cell = queue.pop(0)
        blocked = grid_walls.get(cell, set())
        for d in CARDINALS:
            if d in blocked:
                continue  # known wall, don't cross it
            neighbour = (cell[0] + d[0], cell[1] + d[1])
            if neighbour in seen:
                continue
            seen.add(neighbour)
            if neighbour in visit_counts:
                queue.append(neighbour)  # known territory, keep expanding
            else:
                score += 1  # frontier: unexplored cell reachable from here
    return score


def direction_score(cell, delta):
    """How attractive is heading in this cardinal direction from `cell`?
    Based on how much unexplored squares there are."""
    if delta in grid_walls.get(cell, set()):
        return -1  # known wall - never attractive
    neighbour = (cell[0] + delta[0], cell[1] + delta[1])
    if neighbour not in visit_counts:
        return 5
    return frontier_flood_score(neighbour)


last_map_print = 0.0


def print_map():
    """Prints a map with # being a wall ? being a wall know by neighbouring but not passed through and . is unexplored."""
    global last_map_print
    now = robot.getTime()
    if now - last_map_print < 10.0:
        return
    last_map_print = now

    known_cells = set(visit_counts.keys())
    for cell, walls in grid_walls.items():
        known_cells.add(cell)

    if not known_cells:
        return

    xs = [c[0] for c in known_cells]
    zs = [c[1] for c in known_cells]
    print(f"--- map @ {now:.0f}s | visited cells: {len(visit_counts)} ---")
    for z in range(max(zs), min(zs) - 1, -1):
        row = ""
        for x in range(min(xs), max(xs) + 1):
            cell = (x, z)
            row += "#" if cell in visit_counts else ("?" if cell in grid_walls else ".")
        print(row)


def current_heading():
    """
    Heading estimate. Uses the gyro if it is avialable else use the less efficient GPS calculation method
    """
    global last_heading_pos, last_heading_deg

    if gyro is not None:
        # Get the vertical rotation on Z axis
        yaw_rate = gyro.getValues()[2]
        if abs(yaw_rate) > GYRO_DEADBAND:
            last_heading_deg = (last_heading_deg + math.degrees(yaw_rate * TIME_STEP / 1000.0)) % 360 # Remove noise from the gyro reading
        return last_heading_deg

    #estimate heading from consecutive GPS positions if gyro not available
    pos = gps.getValues()
    if last_heading_pos is None:
        last_heading_pos = pos
        return 0.0
    dx = pos[0] - last_heading_pos[0]
    dz = pos[2] - last_heading_pos[2]
    if math.hypot(dx, dz) < 0.01:
        # Too little movement to get a reliable heading, keep the last one
        return last_heading_deg
    heading = math.degrees(math.atan2(dz, dx))
    last_heading_pos = pos
    last_heading_deg = heading
    return heading


last_heading_pos = None
last_heading_deg = 0.0


# Movement


def turn_right():
    speeds[0] = TURN_SPEED_FAST
    speeds[1] = TURN_SPEED_SLOW


def turn_left():
    speeds[0] = TURN_SPEED_SLOW
    speeds[1] = TURN_SPEED_FAST


def nudge_right():
    """Small correction (not a hard turn) - used to centre between two
    close walls, e.g. squeezing through a narrow doorway, without
    over-steering into the opposite wall."""
    speeds[0] = MAX_VELOCITY
    speeds[1] = MAX_VELOCITY - NUDGE_DIFF


def nudge_left():
    speeds[0] = MAX_VELOCITY - NUDGE_DIFF
    speeds[1] = MAX_VELOCITY


def spin(direction=1):
    speeds[0] = SPIN_SPEED * direction
    speeds[1] = -SPIN_SPEED * direction


def forward():
    speeds[0] = MAX_VELOCITY
    speeds[1] = MAX_VELOCITY


def reverse():
    speeds[0] = -MAX_VELOCITY * 0.6
    speeds[1] = -MAX_VELOCITY * 0.6


def delay(ms):
    init_time = robot.getTime()
    while robot.step(TIME_STEP) != -1:
        if (robot.getTime() - init_time) * 1000.0 > ms:
            break


def safe_read(sensor):
    """Read the sensor's reading and if it returns nothing then return infinte to make sure program doesnt crash midway through"""
    return sensor.getValue() if sensor is not None else float("inf")


def get_front():
    values = [s.getValue() for s in front_sensors if s is not None]
    return min(values) if values else float("inf")


def get_rear():
    values = [s.getValue() for s in rear_sensors if s is not None]
    return min(values) if values else float("inf")


def timed_turn(duration_ms, abort_margin=0.6):
    """
    Runs the currently-set turn speeds for up to duration_ms, but - unlike
    delay() - checks the front sensor every step and bails out early if
    the turn is swinging the robot straight into a wall. This is what a
    fixed blocking delay() can't do: without it, a branch-turn executed
    near a tight corner junction can blindly drive into the wall for the
    full duration, wedge the robot, then retry the exact same blind turn
    every cooldown cycle - which looks like being permanently stuck at
    that spot.
    Returns True if it completed the full duration, False if it aborted
    early because a wall got too close.
    """
    wheel_left.setVelocity(speeds[0])
    wheel_right.setVelocity(speeds[1])
    start = robot.getTime()
    while robot.step(TIME_STEP) != -1:
        if (robot.getTime() - start) * 1000.0 > duration_ms:
            return True
        if get_front() < WALL_THRESHOLD * abort_margin:
            return False
    return False


def get_color():
    if color_sensor is None:
        return PIT_THRESHOLD + 1  # no sensor - assume "not a pit" rather than crash
    img = color_sensor.getImage()
    return color_sensor.imageGetGray(img, color_sensor.getWidth(), 0, 0)


PIT_SATURATION_MAX = 30  # a real black hole is desaturated (R~G~B); a saturated colour
                          # (passage tiles, swamps, checkpoints) can compute a dark GRAYSCALE
                          # value too (blue in particular contributes very little to perceived
                          # brightness) without being remotely black. Requiring low saturation
                          # too is what tells a genuine hole apart from "floor tile that happens
                          # to look dark in grayscale" - this was almost certainly why the robot
                          # wouldn't cross the blue/yellow passage tiles: it was reading them as
                          # pits and reversing away every time, which also explains the repeated
                          # self-restarts (LoP condition c - stuck in a repeating motion sequence).


def get_color_rgb():
    """Returns (r,g,b) 0-255 from the same downward pixel used by
    get_color(). Needed to tell a real black hole apart from a merely
    dark-reading saturated colour."""
    if color_sensor is None:
        return (200, 200, 200)
    img = color_sensor.getImage()
    r = color_sensor.imageGetRed(img, color_sensor.getWidth(), 0, 0)
    g = color_sensor.imageGetGreen(img, color_sensor.getWidth(), 0, 0)
    b = color_sensor.imageGetBlue(img, color_sensor.getWidth(), 0, 0)
    return (r, g, b)


def is_real_pit():
    """A genuine hole is dark AND desaturated. A saturated colour tile
    (area-passage colour codes, swamps, checkpoints) can be just as dark
    in plain grayscale without being a hole at all - see PIT_SATURATION_MAX
    above for why this distinction matters."""
    gray = get_color()
    if gray >= PIT_THRESHOLD:
        return False
    r, g, b = get_color_rgb()
    saturation = max(r, g, b) - min(r, g, b)
    return saturation <= PIT_SATURATION_MAX


if color_sensor is None:
    print("!!! CRITICAL: colour sensor device not found - pit avoidance is DISABLED. "
          "Check the device list printed above for the real name and fix get_device('colour_sensor') below. !!!")

_last_color_debug_print = 0.0


def debug_print_color():
    """Prints the raw colour sensor reading every 2s so you can confirm
    the sensor is actually being read (not silently defaulted) and
    check PIT_THRESHOLD/PIT_SATURATION_MAX are calibrated against real
    values, not guessed."""
    global _last_color_debug_print
    now = robot.getTime()
    if now - _last_color_debug_print < 2.0:
        return
    _last_color_debug_print = now
    gray = get_color()
    r, g, b = get_color_rgb()
    sat = max(r, g, b) - min(r, g, b)
    print(f"[color debug] gray={gray} rgb=({r},{g},{b}) sat={sat} -> "
          f"{'PIT' if is_real_pit() else 'not a pit'}")


_last_heading_debug_print = 0.0


def debug_print_heading(heading_deg):
    """Prints the current heading estimate every 2s. Use this to check
    the gyro's sign convention: manually rotate the robot LEFT (counter-
    clockwise, seen from above) in Webots and watch this value - it
    should INCREASE. If it decreases instead, the gyro sign is inverted
    and current_heading()'s gyro branch needs the sign flipped, or every
    frontier/wall-mapping decision downstream will be working off a
    heading that silently drifts from reality - which looks exactly like
    the robot re-exploring the same physical area over and over."""
    global _last_heading_debug_print
    now = robot.getTime()
    if now - _last_heading_debug_print < 2.0:
        return
    _last_heading_debug_print = now
    print(f"[heading debug] heading={heading_deg:.1f} deg  (source: {'gyro' if gyro is not None else 'gps-diff'})")


# ----------------------------------------------------------------------
# LIDAR-based gap finding (fixes corner traps)
# ----------------------------------------------------------------------
# A concave corner (two walls meeting near a gap) can make the two narrow
# front point-sensors flicker between "wall" and "clear" as their beams
# graze past the opening, causing the robot to oscillate instead of
# committing to a direction. Instead of reacting to single points, this
# scans the LIDAR's full range image, finds the widest open sector, and
# steers toward the MIDDLE of that gap - which is what actually gets a
# robot cleanly out of a corner pocket.
def lidar_widest_gap_heading(min_gap_range=0.15):
    """
    Returns a heading offset (degrees, relative to the robot's current
    facing) pointing at the middle of the widest open gap seen by the
    LIDAR, or None if LIDAR isn't available.
    """
    if lidar is None:
        return None

    ranges = lidar.getRangeImage()
    n = LIDAR_RES
    fov_deg = math.degrees(LIDAR_FOV)

    # index -> angle: index 0 assumed at +fov/2 (left edge), sweeping
    # clockwise to -fov/2 (right edge) at the last index. If your gaps
    # come out inverted (steers into walls instead of away), flip the
    # sign on this line.
    def index_to_angle(i):
        return fov_deg / 2 - (i / (n - 1)) * fov_deg

    best_start = None
    best_len = 0
    cur_start = None
    cur_len = 0

    for i in range(n):
        is_open = ranges[i] > min_gap_range and ranges[i] != float("inf")
        # Treat "inf" (no detection = fully open) as open too
        if ranges[i] == float("inf"):
            is_open = True
        if is_open:
            if cur_start is None:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_start = None
            cur_len = 0
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start

    if best_start is None or best_len == 0:
        return None

    mid_index = best_start + best_len // 2
    return index_to_angle(mid_index)


# ----------------------------------------------------------------------
# Stuck / lack-of-progress tracking
# ----------------------------------------------------------------------
_stuck_timer_start = None
_stuck_ref_pos = None


def update_stuck_tracker(pos):
    """Returns True if a recovery manoeuvre should be triggered."""
    global _stuck_timer_start, _stuck_ref_pos
    now = robot.getTime()

    if _stuck_ref_pos is None:
        _stuck_ref_pos = pos
        _stuck_timer_start = now
        return False

    moved = math.hypot(pos[0] - _stuck_ref_pos[0], pos[2] - _stuck_ref_pos[2])

    if moved > STUCK_DIST:
        # Real progress made - reset the clock
        _stuck_ref_pos = pos
        _stuck_timer_start = now
        return False

    if now - _stuck_timer_start > STUCK_TIME:
        # No meaningful movement for STUCK_TIME seconds -> recover
        print(f"[stuck] no progress for {STUCK_TIME}s @ t={now:.0f}s pos=({pos[0]:.3f},{pos[2]:.3f}) "
              f"front={get_front():.3f} rear={get_rear():.3f} left={safe_read(ds6):.3f} right={safe_read(ds5):.3f} "
              f"- triggering recover()")
        _stuck_ref_pos = pos
        _stuck_timer_start = now
        return True

    return False


_last_new_cell_time = 0.0
_last_visited_count = 0


def update_cell_progress_tracker():
    """Returns True if no NEW cell has been reached in CELL_PROGRESS_TIMEOUT
    seconds. This catches a failure mode update_stuck_tracker() can't:
    shuffling back and forth inside ONE cell with movements bigger than
    STUCK_DIST (2cm) but never crossing into a neighbouring cell (6cm) -
    each shuffle resets the fine-grained timer, so it never fires, while
    exploration is completely stalled. Logs showed exactly this: heading
    slowly oscillating for 80+ seconds with visited-cell count frozen and
    not one [stuck] trigger in that window."""
    global _last_new_cell_time, _last_visited_count
    now = robot.getTime()
    current_count = len(visit_counts)

    if current_count > _last_visited_count:
        _last_visited_count = current_count
        _last_new_cell_time = now
        return False

    if now - _last_new_cell_time > CELL_PROGRESS_TIMEOUT:
        print(f"[cell-stuck] no NEW cell reached for {CELL_PROGRESS_TIMEOUT}s @ t={now:.0f}s "
              f"(still at {current_count} visited cells) - forcing recover()")
        _last_new_cell_time = now  # reset so this doesn't fire every single step
        return True

    return False


def reset_stuck_trackers():
    """Called after a deliberate stop (victim/hazard reporting) so that
    stop doesn't get misread as being physically stuck. Without this, a
    1.3s mandatory report-stop could immediately trip update_stuck_tracker
    once movement resumes (STUCK_TIME=4.0s, so several reports in
    succession - including false positives - can stack toward that
    without the robot ever really being stuck), triggering an unwanted
    recover() spin/reverse right after reporting. That reposition can
    then cause the SAME physical target to be seen again from a new
    angle, getting reported a second time inconsistently instead of
    being cleanly identified once - the reporting stop should never by
    itself count against these timers."""
    global _stuck_timer_start, _stuck_ref_pos, _last_new_cell_time
    now = robot.getTime()
    _stuck_timer_start = now
    _stuck_ref_pos = gps.getValues()
    _last_new_cell_time = now


def recover():
    """Break out of a stuck loop: reverse briefly (only if the rear is
    actually clear - now that we have real rear sensors, no need to back
    blindly into something), then spin toward the widest open gap
    (LIDAR) if available, otherwise toward whichever side sensor reads
    more open space."""
    rear_clear = get_rear() > REAR_WALL_THRESHOLD
    if rear_clear:
        reverse()
        wheel_left.setVelocity(speeds[0])
        wheel_right.setVelocity(speeds[1])
        delay(400)

    gap_heading = lidar_widest_gap_heading()
    if gap_heading is not None:
        direction = 1 if gap_heading > 0 else -1
    else:
        direction = 1 if safe_read(ds6) > safe_read(ds5) else -1  # ds6=left, ds5=right

    print(f"[recover] rear_clear={rear_clear} gap_heading={gap_heading} direction={direction}")
    spin(direction)
    completed = timed_turn(random.randint(500, 900), abort_margin=0.4)  # spin in place; lower margin since a rotating body can still clip a side wall in a tight nook
    print(f"[recover] spin completed={completed}")


# ----------------------------------------------------------------------
# Side-branch detection (opening in a wall while cruising a corridor)
# ----------------------------------------------------------------------
_last_branch_check = 0.0


def check_side_branch(cell, heading, l, r, front_blocked):
    """
    Runs periodically (not just when blocked) to catch side openings
    that a "react only when you hit a wall" controller would drive
    straight past. Returns True if it took control and turned into a
    branch (caller should skip normal wall-following logic this step).
    """
    global _last_branch_check
    now = robot.getTime()
    if front_blocked or now - _last_branch_check < BRANCH_CHECK_COOLDOWN:
        return False
    _last_branch_check = now

    straight_score = direction_score(cell, cardinal_delta(heading))
    left_open = l > OPEN_THRESHOLD
    right_open = r > OPEN_THRESHOLD

    best_dir = None
    best_score = straight_score + BRANCH_SCORE_MARGIN  # must clearly beat straight

    if left_open:
        s = direction_score(cell, cardinal_delta(heading + 90))
        if s > best_score:
            best_score = s
            best_dir = "left"

    if right_open:
        s = direction_score(cell, cardinal_delta(heading - 90))
        if s > best_score:
            best_score = s
            best_dir = "right"

    if best_dir is None:
        return False

    if best_dir == "left":
        turn_left()
    else:
        turn_right()
    timed_turn(BRANCH_TURN_MS)  # aborts early if this swings into a wall instead of the opening
    return True


# ----------------------------------------------------------------------
# Victim / hazard-sign detection and reporting
# ----------------------------------------------------------------------
# Per rulebook 3.7: letter victims are flat black Greek-letter glyphs on
# walls (Phi=Harmed/H, Psi=Stable/S, Omega=Unharmed/U); cognitive targets
# are 5cm circular hazard signs with up to 5 concentric colour rings
# whose values sum to a hazard type (F/P/C/O) or are fake if the sum
# doesn't match a known type.
#
# HONEST LIMITATIONS - read before trusting this in a real run:
#   - Shape classification (telling Phi/Psi/Omega apart) uses hole-count
#     and convex-hull-defect heuristics, NOT a trained classifier. It
#     will misclassify some real tokens - this needs testing against
#     actual rendered glyphs and likely further tuning.
#   - Ring-colour sampling assumes the 5cm target is close enough and
#     large enough in the 40x32 camera image to resolve 5 distinct
#     bands. At typical approach distances this may not hold - if
#     classification looks wrong, the camera resolution/distance is the
#     first thing to check, not just the colour thresholds below.
#   - FAKE 3D-raised letter tokens (rulebook 3.7.4) require a sensor
#     that can detect the raised depth up close - this robot's LIDAR is
#     mounted too high/wide-FOV for that, and camera3 gives no depth
#     information at all. This is NOT implemented: every detected
#     letter shape is reported as real, which will cost misidentification
#     penalty (VMI) if fake tokens are present. Fixing this needs a
#     dedicated short-range distance sensor co-located with the camera.
reported_positions = []  # list of (x, z) already reported, for basic dedupe


def get_camera_image_array(cam):
    """Returns the given camera's image as a (H, W, 4) BGRA numpy array,
    or None if that camera is missing."""
    if cam is None:
        return None
    img = cam.getImage()
    if img is None:
        return None
    return np.frombuffer(img, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))


# ----------------------------------------------------------------------
# Letter victim detection - ported from a more robust reference
# implementation. This REPLACES the earlier hole-count/convexity-defect
# heuristic entirely. That approach worked on a perfectly square-on
# contour but fell apart at any real viewing angle, since a Phi/Psi/Omega
# glyph seen at an angle is a distorted shape - hole counts and defect
# geometry both become unreliable. This version instead:
#   1. Finds the black glyph via an HLS black-mask + connected-components
#      pass (more robust than a single global threshold).
#   2. Finds the WHITE PLAQUE the glyph sits on, as a quadrilateral.
#   3. Perspective-warps that quad to a clean 128x128 square - this is
#      the key step: classification now happens on a corrected, head-on
#      view regardless of the actual viewing angle.
#   4. Classifies the corrected symbol via nearest-neighbour Hamming
#      distance against precomputed 16x16 bit-pattern templates for
#      Phi/Psi/Omega, instead of guessing from raw contour geometry.
# ----------------------------------------------------------------------

VICTIM_TEMPLATES = {
    "phi": [
        1356965290174887078260658139151792462531515355764419497235790369195504435200,
        1696227535118164165313297396658471694034075724282135822372598802646008071136,
        53488692158476348183598498430100367090559075374227014885355631482503360,
        3505534143027102032144499775377872444596270038101538951363447058293713798080,
        904652441617935856407258998661218128133511940592145989457881135404749423584,
        108704176840279312488427664568946020531104448827875980121847871026759648,
        3505448735976296333602386420076882507690851417271122125653811064062770937920,
        3562018466702272537121523523583073491901426591400133011780680660578219327488,
    ],
    "psi": [
        57896044630864119125899510604019369303310824779359608407372389616838814728320,
        1766955113550534429266272198782777782015814295541459639054064760315969540,
        452337005064078800555505553432020310142675351433991253685624892279631118337,
        14474449422244877354909931831240859248506830817194820739326557228737188888576,
        16322133190520788448039884764659077198528098191667357341219800108920821776384,
        29188313510140967587291021394597934890768158582627530525575360280073611313156,
        51790373489292272410800977361548994325620839516807457124674085318584356,
        14475329408384066488506700530652353749753191475387676821576439677081117266178,
    ],
    "omega": [
        54784109081968660734453430457451272521466463116645811843544598190227007,
        101319624094889588307577058341777675232217890344109662737218590028516680589312,
        114208970106971076732864685052108733422482557671666562031851237324387526901760,
        12421610072011271318885805634925389317519707776457985835167159001983418375,
        6903424800357653358143347505224808529315243039020300074290472130870830,
        14475557166895834010568870883870994981823554809498281128860282033880856264704,
        52691772387346089718203146136158325611025491948799020556226871806321781374976,
        406059154071349609351488448980309530115045050246342667917444331798532,
    ],
}
VICTIM_TEMPLATE_TO_TYPE = {"phi": "H", "psi": "S", "omega": "U"}


@dataclass
class VictimCandidate:
    label: int
    center: tuple
    area: int
    bounding_box: tuple


def find_victim_candidates(image, cam_width, cam_height):
    """Finds black-glyph-shaped connected components against the wall,
    using HLS lightness (more robust than a single grayscale threshold)
    and explicitly excluding a wall-shadow hue band that could otherwise
    be mistaken for a black glyph."""
    bgr = image[:, :, :3]
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]

    wall_shadow_mask = (h > 85) & (h < 105) & (l > 15) & (l < 35) & (s > 60) & (s < 80)
    black_mask = ((l < 35) & ~wall_shadow_mask).astype(np.uint8)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)

    image_area = cam_width * cam_height
    min_area = int(image_area * 0.003)
    max_area = int(image_area * 0.4)
    min_aspect, max_aspect = 0.2, 1.4
    max_vertical_pos = 0.75

    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        if h_box == 0:
            continue
        aspect = w / h_box
        if aspect < min_aspect or aspect > max_aspect:
            continue
        cx, cy = x + h_box // 2, y + h_box // 2
        if cy / cam_height > max_vertical_pos:
            continue
        candidates.append(VictimCandidate(label=label, center=(cx, cy), area=area, bounding_box=(x, y, w, h_box)))
    return candidates


def order_quad_points(points):
    points = points.astype(np.float32)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    top_left_index = np.argmin(points[:, 0] + points[:, 1])
    return np.roll(points, -top_left_index, axis=0)


def find_plaque_quad(image, candidate, debug=False):
    """Finds the white plaque backing the glyph, as a 4-point quad -
    this is what lets classification correct for viewing angle instead
    of working on a raw, possibly-distorted contour."""
    bgr = image[:, :, :3]
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    l = hls[:, :, 1]
    x, y, w, h = candidate.bounding_box
    pad = int(max(w, h) * 2)
    rx1, ry1 = max(0, x - pad), max(0, y - pad)
    rx2, ry2 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
    roi_lightness = l[ry1:ry2, rx1:rx2]

    white_mask = (roi_lightness > 140).astype(np.uint8) * 255
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad, best_area = None, 0.0
    candidate_x, candidate_y = candidate.center
    min_contour_area = candidate.area * 2
    # A real plaque is only somewhat larger than the glyph sitting on it -
    # if the wall itself is bright (as it is here: your logs show plain
    # wall reading gray~242, essentially indistinguishable from a white
    # plaque by lightness alone), the ENTIRE visible wall can qualify as
    # "white" and produce one huge contour that swamps every real plaque
    # candidate. Capping the accepted area is what tells "a plaque" apart
    # from "the whole wall got picked up".
    max_contour_area = candidate.area * 30

    if debug:
        print(f"[detect] plaque search: {len(contours)} white contour(s) in ROI, "
              f"need area in ({min_contour_area:.0f}, {max_contour_area:.0f})")

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        if area > max_contour_area:
            if debug:
                print(f"[detect-reject] white contour too large ({area:.0f} > {max_contour_area:.0f}) "
                      f"- likely the whole wall, not a plaque")
            continue
        hull = cv2.convexHull(contour)
        quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2).astype(np.float32)
        if len(quad) < 4:
            if debug:
                print(f"[detect-reject] white contour (area={area:.0f}) didn't approximate to a quad "
                      f"({len(quad)} points)")
            continue
        quad[:, 0] += rx1
        quad[:, 1] += ry1
        if cv2.pointPolygonTest(quad, (float(candidate_x), float(candidate_y)), False) < 0:
            if debug:
                print(f"[detect-reject] quad (area={area:.0f}) found but glyph centre isn't inside it")
            continue
        if area > best_area:
            best_area, best_quad = area, quad

    return best_quad


def warp_plaque(image, plaque_quad, output_size=128):
    ordered = order_quad_points(plaque_quad)
    destination = np.asarray(
        [[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]],
        dtype=np.float32)
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image[:, :, :3], transform, (output_size, output_size))


def classify_victim(warped):
    """Classifies a perspective-corrected plaque image via nearest-
    neighbour Hamming distance against precomputed Phi/Psi/Omega
    templates. Returns (letter_type_or_None, confidence)."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    points = cv2.findNonZero(binary)
    if points is None:
        return None, 0.0

    x, y, w, h = cv2.boundingRect(points)
    symbol = binary[y:y + h, x:x + w]
    symbol = cv2.resize(symbol, (16, 16), interpolation=cv2.INTER_NEAREST)
    symbol = (symbol > 0).astype(np.uint8)

    value = 0
    for bit in symbol.flatten():
        value = (value << 1) | int(bit)

    best_label, best_distance = None, float("inf")
    for label, template_list in VICTIM_TEMPLATES.items():
        for template in template_list:
            distance = (value ^ template).bit_count()
            if distance < best_distance:
                best_distance, best_label = distance, label

    total_bits = 16 * 16
    confidence = (total_bits - best_distance) / total_bits
    if best_label is None or confidence < 0.5:
        return None, confidence
    return VICTIM_TEMPLATE_TO_TYPE[best_label], confidence


def approximate_plaque_quad(candidate, pad_ratio=1.8):
    """Fallback for when brightness-based plaque segmentation can't find
    a real quad - this happens when the plaque is nearly the same
    brightness as the wall it's on (confirmed in logs: this wall reads
    ~242 grayscale, indistinguishable from a white plaque by lightness
    alone, so there's no real contrast boundary for a contour to find).
    Builds an approximate axis-aligned quad around the known glyph
    position instead of giving up entirely. Less accurate at oblique
    viewing angles than a real detected quad (no true perspective
    correction, just a scaled rectangle), but produces something to
    classify instead of nothing."""
    x, y, w, h = candidate.bounding_box
    cx, cy = x + w / 2.0, y + h / 2.0
    half_w, half_h = (w * pad_ratio) / 2.0, (h * pad_ratio) / 2.0
    return np.array([
        [cx - half_w, cy - half_h],
        [cx + half_w, cy - half_h],
        [cx + half_w, cy + half_h],
        [cx - half_w, cy + half_h],
    ], dtype=np.float32)


def detect_victim(image, cam_width, cam_height, debug=False):
    """Top-level letter-victim detector. Returns (type_or_None, confidence)."""
    candidates = find_victim_candidates(image, cam_width, cam_height)
    if debug:
        print(f"[detect] {len(candidates)} victim candidate(s) found this frame")

    plaque_quad = None
    used_fallback = False
    best_candidate = None
    for candidate in candidates:
        plaque_quad = find_plaque_quad(image, candidate, debug=debug)
        if plaque_quad is not None:
            best_candidate = candidate
            break

    if plaque_quad is not None and len(plaque_quad) > 4:
        if debug:
            print("[detect-reject] plaque quad has >4 points - likely cropped/occluded, skipping")
        plaque_quad = None

    if plaque_quad is None and candidates:
        # Real plaque detection failed - fall back to an approximate quad
        # around the largest/most central candidate rather than giving up.
        best_candidate = max(candidates, key=lambda c: c.area)
        plaque_quad = approximate_plaque_quad(best_candidate)
        used_fallback = True
        if debug:
            print("[detect] no real plaque found - using approximate quad fallback "
                  "(less accurate at an angle, but attempts classification instead of skipping)")

    if plaque_quad is None:
        return None, 0.0

    warped = warp_plaque(image, plaque_quad)
    label, confidence = classify_victim(warped)
    if used_fallback:
        confidence *= 0.8  # fallback quad isn't real perspective correction - discount confidence accordingly
        if confidence < 0.5:
            label = None  # re-apply the same confidence bar classify_victim uses, after discounting
    if debug:
        print(f"[detect] victim classification: {label} confidence={confidence:.2f} "
              f"(fallback quad: {used_fallback})")
    return label, confidence


# ----------------------------------------------------------------------
# Cognitive target (hazard sign) detection - ported from a more robust
# reference implementation. This replaces the earlier naive "sample 5
# fixed points along one radius" approach with: HLS-based colour masks
# per channel, ellipse-fitting on each mask's contours (not just bounding
# boxes), clustering same-target detections found across different
# colour masks into one candidate, then densely sampling each ring band
# (many angles x several radii, not one point) and taking a majority-vote
# dominant colour with a confidence score. Geometry (circularity, aspect
# ratio, occupancy, cluster strength) is scored before ring sampling even
# runs, so malformed/noise contours are rejected early.
# ----------------------------------------------------------------------

Colour = typing.Literal["red", "green", "blue", "yellow", "black", "combined"]


@dataclass
class ViewCorrection:
    lateral: float = 0.0
    forward: float = 0.0
    yaw: float = 0.0


@dataclass
class CognitiveTargetDetection:
    found: bool = False
    type: str | None = None
    confidence: float = 0
    rings: list = field(default_factory=list)
    view_correction: ViewCorrection = field(default_factory=ViewCorrection)
    center: tuple | None = None
    radius: float = 0


@dataclass
class EllipseCandidate:
    cx: float
    cy: float
    major: float
    minor: float
    angle: float
    area: float
    circularity: float
    contour_index: int
    source_colour: Colour


@dataclass
class CandidateEvaluation:
    result: CognitiveTargetDetection
    score: float
    agreements: list
    geometry_score: float
    valid_classification: bool
    target_mask: np.ndarray = None


@dataclass
class EllipseCluster:
    members: list
    cx: float
    cy: float
    outer: EllipseCandidate


@dataclass
class RingSampleResult:
    colour: typing.Optional[Colour]
    agreement: float
    sample_count: int


def classify_hls(h, l, s):
    if l < 25:
        return "black"
    if s < 180:
        return None  # low-saturation surfaces (e.g. plain walls) aren't a ring colour
    hue = h * 2
    if hue <= 15 or hue >= 345:
        return "red"
    if 45 <= hue <= 75:
        return "yellow"
    if 100 <= hue <= 140:
        return "green"
    if 210 <= hue <= 270:
        return "blue"
    return None


def cluster_ellipses(ellipses):
    max_distance = min(CAM_WIDTH, CAM_HEIGHT) * 0.08 if CAM_WIDTH else 5.0
    clusters = []
    for ellipse in ellipses:
        for cluster in clusters:
            mean_cx = np.mean([e.cx for e in cluster])
            mean_cy = np.mean([e.cy for e in cluster])
            distance = np.hypot(ellipse.cx - mean_cx, ellipse.cy - mean_cy)
            if distance <= max_distance:
                cluster.append(ellipse)
                break
        else:
            clusters.append([ellipse])

    result = []
    for cluster in clusters:
        outer = max(cluster, key=lambda e: e.major * e.minor)
        result.append(EllipseCluster(
            members=cluster,
            cx=float(np.mean([e.cx for e in cluster])),
            cy=float(np.mean([e.cy for e in cluster])),
            outer=outer,
        ))
    return result


def estimate_yaw_from_cluster(cluster):
    if len(cluster.members) < 3:
        return 0.0
    members = sorted(cluster.members, key=lambda e: e.major * e.minor)
    inner, outer = members[0], members[-1]
    dx = outer.cx - inner.cx
    return max(-1.0, min(1.0, dx))


def sample_ring_region(hls_img, cx, cy, a, b, rotation_deg, inner_frac, outer_frac):
    colours = []
    avg_radius = (a + b) / 2.0
    angle_count = max(24, int(avg_radius))
    radius_count = 4
    radius_samples = np.linspace(inner_frac, outer_frac, radius_count)
    theta_samples = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    cos_values = np.cos(theta_samples)
    sin_values = np.sin(theta_samples)
    rot = np.deg2rad(rotation_deg)
    cos_rot = np.cos(rot)
    sin_rot = np.sin(rot)

    for frac in radius_samples:
        scaled_a = frac * a
        scaled_b = frac * b
        for cos_theta, sin_theta in zip(cos_values, sin_values):
            xr = scaled_a * cos_theta
            yr = scaled_b * sin_theta
            x = int(round(cx + xr * cos_rot - yr * sin_rot))
            y = int(round(cy + xr * sin_rot + yr * cos_rot))
            if x < 0 or y < 0 or x >= hls_img.shape[1] or y >= hls_img.shape[0]:
                continue
            h, l, s = hls_img[y, x]
            colour = classify_hls(int(h), int(l), int(s))
            if colour is None:
                continue
            colours.append(colour)

    histogram = Counter(colours)
    if len(colours) < 10:
        return RingSampleResult(None, 0.0, len(colours))
    dominant_colour, count = histogram.most_common(1)[0]
    agreement = count / len(colours)
    return RingSampleResult(dominant_colour, agreement, len(colours))


def classify_candidate(rings):
    total = sum(RING_COLOR_VALUES[colour] for colour in rings)
    return HAZARD_SUM_TO_TYPE.get(total)


def colour_evidence_score(cluster):
    colours = {m.source_colour for m in cluster.members if m.source_colour != "combined"}
    return min(len(colours) / 5.0, 1.0)


def score_candidate(cluster, rings, agreements, outer):
    mean_agreement = sum(agreements) / len(agreements)
    ratio = min(outer.major, outer.minor) / max(outer.major, outer.minor)
    changes = sum(rings[i] != rings[i + 1] for i in range(len(rings) - 1))
    cluster_strength = min(len(cluster.members) / 4.0, 1.0)
    colour_evidence = colour_evidence_score(cluster)
    geometry_bonus = outer.circularity * ratio
    return (mean_agreement * geometry_bonus * (1.0 + 0.25 * changes)
            * (0.5 + 0.5 * cluster_strength) * (0.5 + colour_evidence))


def populate_navigation_geometry(result, image, cx, cy, major, minor):
    a, b = major / 2.0, minor / 2.0
    result.center = (int(round(cx)), int(round(cy)))
    result.radius = min(a, b)
    image_height, image_width = image.shape[0], image.shape[1]
    outside_left = max(0.0, -(cx - a))
    outside_right = max(0.0, (cx + a) - image_width)
    outside_top = max(0.0, -(cy - b))
    outside_bottom = max(0.0, (cy + b) - image_height)
    max_outside = max(outside_left, outside_right, outside_top, outside_bottom)
    crop_fraction = max_outside / max(a, b)

    correction = ViewCorrection()
    image_centre_x = image_width / 2.0
    correction.lateral = max(-1.0, min(1.0, (cx - image_centre_x) / (image_width / 2.0)))
    diameter_fraction = max(major, minor) / min(image_width, image_height)
    MIN_DIAMETER_FRAC, MAX_DIAMETER_FRAC = 0.25, 0.8
    if crop_fraction > 0.15:
        correction.forward = max(-1.0, -crop_fraction * 2.0)
    elif diameter_fraction < MIN_DIAMETER_FRAC:
        correction.forward = (MIN_DIAMETER_FRAC - diameter_fraction) / MIN_DIAMETER_FRAC
    elif diameter_fraction > MAX_DIAMETER_FRAC:
        correction.forward = -(diameter_fraction - MAX_DIAMETER_FRAC) / MAX_DIAMETER_FRAC
    else:
        correction.forward = 0.0
    result.view_correction = correction


def build_target_mask(bgr):
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    black_mask = (l < 25).astype(np.uint8) * 255
    red_mask = (((h <= 8) | (h >= 172)) & (s > 180)).astype(np.uint8) * 255
    yellow_mask = ((h >= 22) & (h <= 38) & (s > 180)).astype(np.uint8) * 255
    green_mask = ((h >= 50) & (h <= 70) & (s > 180)).astype(np.uint8) * 255
    blue_mask = ((h >= 105) & (h <= 135) & (s > 180)).astype(np.uint8) * 255
    target_mask = cv2.bitwise_or(red_mask, yellow_mask)
    target_mask = cv2.bitwise_or(target_mask, green_mask)
    target_mask = cv2.bitwise_or(target_mask, blue_mask)
    target_mask = cv2.bitwise_or(target_mask, black_mask)
    kernel = np.ones((3, 3), np.uint8)
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
    return {"red": red_mask, "blue": blue_mask, "yellow": yellow_mask,
            "green": green_mask, "black": black_mask, "combined": target_mask}


def extract_ellipse_candidates(masks, cam_width, cam_height):
    candidates = []
    for source_colour, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < cam_width * cam_height * 0.005:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter < min(cam_width, cam_height) * 0.234:
                continue
            if len(contour) < 5:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.40:
                continue
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            ratio = min(major, minor) / max(major, minor)
            if ratio < 0.50:
                continue
            candidates.append(EllipseCandidate(
                cx=cx, cy=cy, major=major, minor=minor, angle=angle,
                area=area, circularity=circularity, contour_index=i, source_colour=source_colour))
    return candidates


def evaluate_cluster(cluster, image, hls, bgr, debug=False):
    result = CognitiveTargetDetection()
    outer = cluster.outer
    cx, cy, major, minor = outer.cx, outer.cy, outer.major, outer.minor
    a, b = major / 2.0, minor / 2.0

    ring_bounds = [(0.00, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]
    ellipse_area = np.pi * a * b
    occupancy = outer.area / ellipse_area if ellipse_area > 0 else 0

    if occupancy < 0.55:
        # Was 0.70 - logs from an actual run showed real-looking candidates
        # repeatedly landing at 0.60-0.69, just under that cutoff, on this
        # camera's 40x32 resolution. Loosened based on that observed data,
        # not a blind guess.
        if debug:
            print(f"[detect-reject] occupancy too low: {occupancy:.2f} (need >=0.55)")
        return CandidateEvaluation(result, 0.0, [], 0.0, False)

    ratio = min(major, minor) / max(major, minor)
    populate_navigation_geometry(result, image, cx, cy, major, minor)

    image_height = image.shape[0]
    vertical_position = cy / image_height
    vertical_penalty = 0.2 if vertical_position > 0.80 else 1.0

    geometry_score = (outer.circularity * ratio * min(len(cluster.members) / 4.0, 1.0)
                       * occupancy * vertical_penalty)
    if geometry_score < 0.25:
        # Was 0.35 - same reasoning: real candidates were landing at 0.30-0.35
        # and getting rejected right at the boundary.
        if debug:
            print(f"[detect-reject] geometry score too weak: {geometry_score:.2f} (need >=0.25) "
                  f"circ={outer.circularity:.2f} ratio={ratio:.2f} members={len(cluster.members)}")
        return CandidateEvaluation(result, 0.0, [], geometry_score, False)

    rings, agreements = [], []
    for inner_frac, outer_frac in ring_bounds:
        sample = sample_ring_region(hls, cx, cy, a, b, outer.angle, inner_frac, outer_frac)
        if sample.colour is None:
            if debug:
                print(f"[detect-reject] ring [{inner_frac:.1f}-{outer_frac:.1f}] colour unresolved "
                      f"(samples={sample.sample_count})")
            return CandidateEvaluation(result, 0.0, [], geometry_score, False)
        rings.append(sample.colour)
        agreements.append(sample.agreement)

    result.view_correction.yaw = estimate_yaw_from_cluster(cluster) * (1 - ratio)

    target_type = classify_candidate(rings)
    if target_type is None:
        if debug:
            print(f"[detect-reject] ring sum doesn't match a known type: rings={rings} "
                  f"sum={sum(RING_COLOR_VALUES[c] for c in rings)}")
        return CandidateEvaluation(result, 0.0, agreements, geometry_score, False)

    result.found = True
    result.type = target_type
    result.rings = rings
    mean_agreement = sum(agreements) / len(agreements)
    result.confidence = max(0.0, min(1.0, mean_agreement * geometry_score))

    score = score_candidate(cluster, rings, agreements, outer)
    return CandidateEvaluation(result, score, agreements, geometry_score, True)


def detect_cognitive_target(image, cam_width, cam_height, debug=False):
    """Top-level cognitive-target detector. Returns a
    CognitiveTargetDetection - check .found and .type."""
    result = CognitiveTargetDetection()
    bgr = image[:, :, :3]
    masks = build_target_mask(bgr)
    candidates = extract_ellipse_candidates(masks, cam_width, cam_height)
    if len(candidates) == 0:
        return result

    if debug:
        print(f"[detect] {len(candidates)} ellipse candidate(s) found this frame")

    clusters = cluster_ellipses(candidates)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    best_valid, best_geometry = None, None
    for cluster in clusters:
        candidate = evaluate_cluster(cluster, image, hls, bgr, debug=debug)
        if best_geometry is None or candidate.geometry_score > best_geometry.geometry_score:
            best_geometry = candidate
        if candidate.valid_classification:
            if best_valid is None or candidate.score > best_valid.score:
                best_valid = candidate

    if best_valid is not None:
        return best_valid.result
    return CognitiveTargetDetection()  # geometry-only matches aren't reported - avoid guessing a type


def report(victim_type):
    """Reports a victim/hazard to the supervisor. Matches the verified
    official sample format exactly: stop, wait 1.3s, then send
    struct.pack('i i c', x_cm, z_cm, type_char) via the emitter."""
    global _last_report_time
    if emitter is None:
        print("[victim] cannot report - no emitter device")
        return

    wheel_left.setVelocity(0)
    wheel_right.setVelocity(0)
    delay(STOP_BEFORE_REPORT_MS)

    pos = gps.getValues()
    pos_x = int(pos[0] * 100)
    pos_z = int(pos[2] * 100)

    reported_positions.append((pos[0], pos[2]))
    _last_report_time = robot.getTime()
    message = struct.pack("i i c", pos_x, pos_z, bytes(victim_type, "utf-8"))
    emitter.send(message)
    print(f"[victim] reported '{victim_type}' at ({pos_x}cm, {pos_z}cm)")
    robot.step(TIME_STEP)

    # The deliberate stop above must never look like the robot being
    # physically stuck to our own recovery system - see the docstring on
    # reset_stuck_trackers() for why this matters.
    reset_stuck_trackers()


_last_report_time = -999.0


def already_reported_nearby(pos):
    return any(math.hypot(pos[0] - rx, pos[2] - rz) < REPORT_DEDUPE_DIST for rx, rz in reported_positions)


_last_victim_check = 0.0
VICTIM_CHECK_INTERVAL = 0.2  # seconds - runs OpenCV at ~5Hz instead of every
                             # 32ms simulation step. This is real CPU work
                             # (image conversion, thresholding, contour
                             # finding) with no benefit to running it faster
                             # than the robot can physically move past a
                             # token - throttling keeps it from competing
                             # for cycles with the timing-sensitive turn/
                             # recovery/corner-gap logic elsewhere in the
                             # loop. Detection itself never touches the
                             # wheels except through report(), which only
                             # fires on a confident match - so this can't
                             # silently interfere with ordinary driving
                             # decisions, but it can waste CPU if unthrottled.


def run_detection_for_camera(cam, get_dist_fn, label):
    """Runs the full tiered detection pipeline (cheap distance gate ->
    cheap pre-check, no stop -> real stop + full analysis) for a single
    camera/co-located-sensor pair. Returns True if it reported something
    (caller should stop checking other cameras this step)."""
    if cam is None:
        return False

    if get_dist_fn() > FRONT_APPROACH_THRESHOLD:
        return False

    pos = gps.getValues()
    if already_reported_nearby(pos):
        return False

    # TIER 1 - cheap pre-check, NO stop.
    img = get_camera_image_array(cam)
    if img is None:
        return False
    cam_w, cam_h = cam.getWidth(), cam.getHeight()
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    masks = build_target_mask(bgr)
    preliminary_hazard_candidates = extract_ellipse_candidates(masks, cam_w, cam_h)
    preliminary_victims = find_victim_candidates(img, cam_w, cam_h)

    if not preliminary_hazard_candidates and not preliminary_victims:
        return False  # plain wall, nothing candidate-shaped in view - no stop, keep driving normally

    # TIER 2 - something worth a closer look. NOW stop for a genuinely
    # stable frame before running the expensive full analysis.
    wheel_left.setVelocity(0)
    wheel_right.setVelocity(0)
    delay(DETECTION_STABILIZE_MS)

    img = get_camera_image_array(cam)
    if img is None:
        return False
    cam_w, cam_h = cam.getWidth(), cam.getHeight()

    hazard = detect_cognitive_target(img, cam_w, cam_h, debug=True)
    if hazard.found:
        print(f"[detect:{label}] cognitive target candidate: type={hazard.type} "
              f"confidence={hazard.confidence:.2f} rings={hazard.rings}")
        if hazard.confidence >= MIN_HAZARD_CONFIDENCE:
            report(hazard.type)
            return True
        else:
            print(f"[detect-reject:{label}] confidence {hazard.confidence:.2f} below "
                  f"{MIN_HAZARD_CONFIDENCE} threshold - not reporting")
        return False

    letter_type, confidence = detect_victim(img, cam_w, cam_h, debug=True)
    if letter_type is not None:
        report(letter_type)
        return True
    return False


def check_for_victim_or_hazard():
    """Runs detection for BOTH cameras (front, then right), whichever
    finds something first. Both go through the same performance-gating
    structure: a cheap distance check, then a cheap TIER 1 pre-check (no
    stop), and only a real stabilization stop if something candidate-
    shaped is actually in view - plain walls (which the distance gate
    alone hits at every turn in a maze) never trigger a stop or the
    expensive full analysis."""
    global _last_victim_check
    now = robot.getTime()
    if now - _last_victim_check < VICTIM_CHECK_INTERVAL:
        return
    _last_victim_check = now

    if now - _last_report_time < MIN_REPORT_INTERVAL:
        return  # hard cooldown - see MIN_REPORT_INTERVAL's config comment

    if run_detection_for_camera(camera, get_front, "front"):
        return
    run_detection_for_camera(camera_right, lambda: safe_read(ds5), "right")


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
while robot.step(TIME_STEP) != -1:
    pos = gps.getValues()
    cell = mark_visited(pos)

    # Lack-of-progress message from supervisor (sent as 'L' style byte)
    if receiver is not None:
        while receiver.getQueueLength() > 0:
            receiver.nextPacket()  # just drain it; recover() below handles it

    if update_stuck_tracker(pos) or update_cell_progress_tracker():
        recover()
        continue

    f = get_front()       # min across all 4 front sensors - computed early so the
                           # approach-speed decision below can use it

    if f < APPROACH_SLOWDOWN_THRESHOLD:
        # Slow, steady approach zone - well before collision-avoidance
        # turning (WALL_THRESHOLD) or detection (FRONT_APPROACH_THRESHOLD)
        # kick in, so neither has to fight residual motion from cruising
        # in at full speed.
        speeds[0] = APPROACH_SLOWDOWN_SPEED
        speeds[1] = APPROACH_SLOWDOWN_SPEED
    else:
        forward()

    l = safe_read(ds6)   # true left-side sensor
    r = safe_read(ds5)   # true right-side sensor

    front_blocked = f < WALL_THRESHOLD
    left_blocked = l < WALL_THRESHOLD
    right_blocked = r < WALL_THRESHOLD

    heading = current_heading()
    debug_print_heading(heading)

    # Record what we can see from this cell into the persistent wall map
    mark_walls(cell, heading, front_blocked, left_blocked, right_blocked)
    print_map()
    debug_print_color()

    check_for_victim_or_hazard()

    # Check for a worthwhile side opening before falling back to plain
    # wall-following - this is what catches branches/rooms off a corridor.
    if check_side_branch(cell, heading, l, r, front_blocked):
        continue

    if front_blocked:
        # Back up for clearance before turning - turning while already
        # nearly touching the wall drags the chassis along the surface
        # (looks like sliding sideways while still facing the wall)
        # instead of rotating cleanly away from it.
        if get_rear() > REAR_WALL_THRESHOLD:
            reverse()
            timed_turn(250, abort_margin=0.0)  # abort_margin=0 - reversing, not turning into anything

        # Corner-trap fix: prefer steering at the middle of the widest
        # LIDAR gap over the old point-sensor left/right guess, since
        # that's what actually clears a concave corner cleanly instead
        # of flickering between two narrow readings.
        gap_heading = lidar_widest_gap_heading()

        if gap_heading is not None:
            if gap_heading > 10:
                turn_left()
            elif gap_heading < -10:
                turn_right()
            # else: gap is roughly dead ahead but sensors say blocked -
            # fall through to the point-sensor fallback below
            else:
                gap_heading = None

        if gap_heading is None:
            # Fallback: pick the direction toward more unexplored territory,
            # now using the real wall map instead of a straight-line guess.
            left_score = direction_score(cell, cardinal_delta(heading + 90))
            right_score = direction_score(cell, cardinal_delta(heading - 90))

            if left_blocked and not right_blocked:
                turn_right()
            elif right_blocked and not left_blocked:
                turn_left()
            elif left_score > right_score:
                turn_left()
            elif right_score > left_score:
                turn_right()
            else:
                # Genuine tie - previously this always defaulted to
                # turn_left(), which combined with near-identical scores
                # (see frontier_flood_score's max_nodes note above) meant
                # the SAME direction got picked every single time this
                # junction was reached, permanently looping a known
                # circuit instead of ever committing to the other way
                # round. Randomising the tie-break is what actually
                # breaks that cycle.
                if random.random() < 0.5:
                    turn_left()
                else:
                    turn_right()

        # Commit to the turn for a real duration - without this, the turn
        # speeds set above only apply for a single ~32ms simulation step
        # before the next loop iteration calls forward() again, which is
        # nowhere near enough to actually rotate away from the wall. This
        # is what caused it to look like it was "sliding" along the wall
        # while still facing it, rather than turning clear of it.
        timed_turn(450)
    else:
        # Narrow gap fix: if BOTH sides are close (squeezing between two
        # wall corners, like a doorway), centre using a gentle correction
        # based on which side is tighter - don't do a hard turn toward
        # one wall, which would just push the robot into that corner.
        if left_blocked and right_blocked:
            if r > l:
                nudge_right()
            elif l > r:
                nudge_left()
            # else: equally close on both sides - go straight (forward()
            # already set above)
        elif left_blocked:
            turn_right()
        elif right_blocked:
            turn_left()

    # Pit / black-floor avoidance takes priority - reverse and turn away
    if is_real_pit():
        reverse()
        wheel_left.setVelocity(speeds[0])
        wheel_right.setVelocity(speeds[1])
        delay(300)
        spin(1 if random.random() > 0.5 else -1)
        wheel_left.setVelocity(speeds[0])
        wheel_right.setVelocity(speeds[1])
        delay(500)
        continue

    wheel_left.setVelocity(speeds[0])
    wheel_right.setVelocity(speeds[1])
