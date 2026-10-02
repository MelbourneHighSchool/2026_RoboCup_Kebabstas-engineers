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

#safery thresholds
WALL_THRESHOLD = 0.05          # When wall will get detected
OPEN_THRESHOLD = 0.25          # How big a branch has to be for it to be considered open
PIT_THRESHOLD = 80             # What is considered black floor
REAR_WALL_THRESHOLD = 0.05     # What value to stop reversing at

#exploration features
CELL_SIZE = 0.06               # metres per grid cell
STUCK_TIME = 4.0                # When the robot will be considered stuck
STUCK_DIST = 0.02               # Any movement less than this is considered stuck
CELL_PROGRESS_TIMEOUT = 15.0     # How much seconds with no new cells visited, forces the recover func
GYRO_DEADBAND = 0.02             # rad/s - yaw rates below this are treated as noise/bias, not real turning

BRANCH_CHECK_COOLDOWN = 1.5     # How much time between side branch evaluations
BRANCH_SCORE_MARGIN = 1         # how much better a side branch's score must be to go to it
BRANCH_TURN_MS = 550            # how long it takes to turn toward a branch

REVISIT_PENALTY = 0.75          # subtracted from a known neighbour's score per prior visit -
                                 # stops two "explored" directions tying forever, so the robot
                                 # doesn't just loop the same known cells back and forth
FRONTIER_PULL_COOLDOWN = 2.5    # seconds between frontier-pull checks
FRONTIER_FORCE_INTERVAL = 20.0  # seconds - force a global "where's the nearest unexplored
                                 # area" check even when locally satisfied, so a small pocket
                                 # full of easy unvisited cells can't stall exploring the rest
                                 # of the maze forever

TURN_SPEED_FAST = 0.6 * MAX_VELOCITY
NUDGE_DIFF = 0.15 * MAX_VELOCITY  # To centre easily in narrow gaps with small turns
TURN_SPEED_SLOW = -0.2 * MAX_VELOCITY
SPIN_SPEED = 0.6 * MAX_VELOCITY

# Victim / hazard-sign detection
FRONT_APPROACH_THRESHOLD = 0.08  # When to start victim detection on the FRONT camera - tight
                                  # on purpose, since this fires right as the robot closes in
                                  # on a wall/dead-end head-on
SIDE_APPROACH_THRESHOLD = 0.18   # When to start victim detection on the RIGHT (side-facing)
                                 # camera - deliberately much more generous than the front
                                 # threshold. That camera looks sideways at whatever wall the
                                 # robot is cruising PAST, not one it's driving toward - the
                                 # robot doesn't hug that wall, so it's typically well beyond
                                 # 8cm away the entire time it passes a sign mounted flat on
                                 # it. Reusing the tight front threshold for this camera meant
                                 # side-wall signs were essentially never close enough to ever
                                 # trigger detection. Tune this against your maze's actual
                                 # corridor width if signs are still being missed or if it's
                                 # triggering too early/often.
APPROACH_SLOWDOWN_THRESHOLD = 0.10  # When it thinks it is approaching a wall
APPROACH_SLOWDOWN_SPEED = 0.35 * MAX_VELOCITY  # How fast it goes while in the threshold of approaching wall
DETECTION_STABILIZE_MS = 250    # Fully stops robot for that much time
REPORT_DEDUPE_DIST = 0.20        # Skip reporting within this much cm to stop double detection
MIN_REPORT_INTERVAL = 3.0        # How much times report func can run in a second
STOP_BEFORE_REPORT_MS = 1300     # Have to stop for at least 1s to report
MIN_HAZARD_CONFIDENCE = 0.48     # a cognitive-target match below this confidence doesn't get
                                  # reported - loosened from 0.55, recall weighted over
                                  # precision per explicit priority: a missed real target
                                  # scores nothing, so catching more of them matters more than
                                  # being maximally conservative


HALF_TILE_DISTANCE = 0.15

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
DELTA_TO_DEG = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}

visit_counts = {}          # cell, number of times physically visited
grid_walls = {}            # cell, set of (dx,dz) co-ords known to be blocked


def get_cell(pos):
    """Convert an (x, y, z) GPS reading into a grid cell coordinate."""
    return (round(pos[0] / CELL_SIZE), round(pos[2] / CELL_SIZE))


def mark_visited(pos):
    cell = get_cell(pos)
    visit_counts[cell] = visit_counts.get(cell, 0) + 1 #sees how many times cell is in dictionary
    return cell


def snap_cardinal(deg):
    """Round a heading to the nearest of 0 or 90 or 180 or 270 degrees."""
    return (round(deg / 90.0) % 4) * 90


def cardinal_delta(deg): #figures out which cell is infront of robot
    d = snap_cardinal(deg)
    return {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[d]


def mark_walls(cell, heading_deg, front_blocked, left_blocked, right_blocked):
    #Saves which directions the walls are at this cell
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
    queue = [start_cell] #cells waiting to be explored
    score = 0
    while queue and len(seen) < max_nodes: #while we have cells to explore and hasn't hit max limit
        cell = queue.pop(0)
        blocked = grid_walls.get(cell, set()) #checks for known walls for a cell
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
    #How good is heading in this direction from location, based on how much unexplored squares
    #there are. Known neighbours get a small penalty per prior visit so two "explored"
    #directions don't tie forever and keep sending the robot round the same loop.
    if delta in grid_walls.get(cell, set()): #checks for a known wall
        return -1  
    neighbour = (cell[0] + delta[0], cell[1] + delta[1])
    if neighbour not in visit_counts:
        return 5
    visits = visit_counts.get(neighbour, 0)
    return frontier_flood_score(neighbour) - REVISIT_PENALTY * visits


def is_fresh_direction(cell, delta):
    """True if this direction leads to a genuinely never-visited neighbour
    (and isn't a known wall). Used to let a completely unexplored branch
    win outright against ALREADY-explored straight-ahead territory,
    regardless of that territory's computed frontier score - see
    check_side_branch for why this matters (small 1-tile pockets/zones
    were otherwise permanently losing to "more open area")."""
    if delta in grid_walls.get(cell, set()):
        return False
    neighbour = (cell[0] + delta[0], cell[1] + delta[1])
    return neighbour not in visit_counts


_last_frontier_pull = 0.0
_last_forced_frontier = 0.0


def bfs_next_step_toward_frontier(start, max_nodes=800):
    """
    Full BFS over the known wall map (no small cap, unlike frontier_flood_score above) to
    find the nearest reachable cell that's unvisited, or next to an unvisited cell through a
    known-open edge. Returns the (dx,dz) direction of the FIRST step toward it, or None if
    nothing reachable is found.
    """
    if start not in visit_counts:
        return None
    parent_and_step = {start: (None, None)}
    queue = deque([start]) #works faster since there is max of 800
    nodes_checked = 0
    while queue and nodes_checked < max_nodes:
        cell = queue.popleft()
        nodes_checked += 1
        blocked = grid_walls.get(cell, set())
        for d in CARDINALS:
            if d in blocked:
                continue
            neighbour = (cell[0] + d[0], cell[1] + d[1])
            if neighbour in parent_and_step:
                continue
            first_step = d if cell == start else parent_and_step[cell][1]
            parent_and_step[neighbour] = (cell, first_step)
            if neighbour not in visit_counts:
                return first_step
            queue.append(neighbour)
    return None


def check_frontier_pull(cell, heading):
    """
    check_frontier_pull periodically runs a full breadth-first search across 
    everything the robot has mapped so far, to catch cases where nothing 
    nearby looks worth exploring but there's still unmapped territory reachable 
    further out.
    """
    global _last_frontier_pull, _last_forced_frontier
    now = robot.getTime()
    if now - _last_frontier_pull < FRONTIER_PULL_COOLDOWN:
        return False
    _last_frontier_pull = now

    open_dirs = [d for d in CARDINALS if d not in grid_walls.get(cell, set())] #every direction not a wall
    local_scores = [direction_score(cell, d) for d in open_dirs]
    locally_starved = not local_scores or max(local_scores) <= 0

    forced = now - _last_forced_frontier > FRONTIER_FORCE_INTERVAL
    if not locally_starved and not forced:
        return False
    if forced:
        _last_forced_frontier = now

    target_delta = bfs_next_step_toward_frontier(cell)
    if target_delta is None:
        return False

    target_deg = DELTA_TO_DEG[target_delta]
    diff = (target_deg - heading + 180) % 360 - 180

    if not locally_starved and abs(diff) < 45:
        return False  # already heading roughly the right way - don't interrupt good progress

    print(f"[frontier-pull] {'forced' if forced else 'starved'} @ {cell}, nearest frontier "
          f"is via {target_delta} (turn {diff:.0f} deg)")

    if diff > 15:
        turn_left()
        timed_turn(BRANCH_TURN_MS)
    elif diff < -15:
        turn_right()
        timed_turn(BRANCH_TURN_MS)
    return True


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

    xs = [c[0] for c in known_cells] #extracts all x coordinates to see size of map
    zs = [c[1] for c in known_cells]
    print(f"--- map @ {now:.0f}s | visited cells: {len(visit_counts)} ---")
    for z in range(max(zs), min(zs) - 1, -1):
        row = ""
        for x in range(min(xs), max(xs) + 1):   
            cell = (x, z)
            row += "#" if cell in visit_counts else ("?" if cell in grid_walls else ".")
        print(row)


def current_heading():

    #Heading estimate
    
    global last_heading_pos, last_heading_deg

    if gyro is not None:
        # Get the vertical rotation on Z axis
        yaw_rate = gyro.getValues()[2]
        if abs(yaw_rate) > GYRO_DEADBAND:
            last_heading_deg = (last_heading_deg + math.degrees(yaw_rate * TIME_STEP / 1000.0)) % 360 #calculates time in seconds, then multiplies with rate in radians per second, then convert to degree, then make sure number from 0-360
        return last_heading_deg

    #estimate heading if no gyro
    pos = gps.getValues()
    if last_heading_pos is None:
        last_heading_pos = pos
        return 0.0
    dx = pos[0] - last_heading_pos[0]
    dz = pos[2] - last_heading_pos[2]
    if math.hypot(dx, dz) < 0.01: #measures if straight line distance is too small
        return last_heading_deg
    heading = math.degrees(math.atan2(dz, dx)) #calculates angle of vector in radians then converts to degrees
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
    #Small correction not a hard turn 
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
    values = [s.getValue() for s in front_sensors if s is not None] #gets value from every sensor
    return min(values) if values else float("inf")


def lidar_front_min(cone_deg=15):
    """Minimum LIDAR range within a narrow forward cone. The 4 front
    point sensors (get_front above) are each a single discrete ray -
    between them there are angular gaps a thin wall or a corner hit at
    just the wrong angle can pass through without any of the 4 ever
    reading it as blocked. The LIDAR has continuous angular coverage
    across its FOV, so combining the two closes that gap instead of 
    relying on 4 fixed rays alone."""
    if lidar is None:
        return float("inf")
    ranges = lidar.getRangeImage()
    n = LIDAR_RES
    fov_deg = math.degrees(LIDAR_FOV) #total width of what lidar sees
    min_r = float("inf")
    for i in range(n):
        angle = fov_deg / 2 - (i / (n - 1)) * fov_deg
    #i / (n - 1) — this is a "how far along am I" fraction, always between 0 and 1.
    #At i = 0 (the very first beam): 0 / (n-1) = 0, At i = n - 1 (the very last beam): (n-1) / (n-1) = 1
    #Everything in between lands smoothly between 0 and 1.
    #fov_deg / 2 — sets the starting angle. Since the LIDAR's field of view is symmetric around 0°, the left edge sits at fov_deg/2
    #the * fov_deg at the end — sets the total distance being swept. (i/(n-1)) is a fraction from 0 to 1 — "what fraction of the way through the beams am I." Multiplying that fraction by fov_deg converts it into "how many degrees have I swept so far," ranging from 0° swept up to the entire fov_deg swept 
        if abs(angle) <= cone_deg:
            r = ranges[i]
            if r != float("inf") and r < min_r:
                min_r = r
    return min_r


def get_front_combined():
    """Front distance used for wall-blocked decisions: the smaller
    (more cautious) of the 4 point sensors and the LIDAR's forward cone.
    Never makes the robot MORE willing to drive forward than either
    source alone would - it only adds a second, angularly-continuous
    check that can catch what a gap between the 4 discrete rays missed."""
    return min(get_front(), lidar_front_min())


def get_rear():
    values = [s.getValue() for s in rear_sensors if s is not None]
    return min(values) if values else float("inf")


def distance_to_wall(label):
    """Approximate distance from the robot to the wall surface a given
    camera is looking at - used as a stand-in for "distance to the wall
    token", since the controller has no independent way to know a
    token's true world position (only vision, which gives bearing/shape,
    not a reliable metric distance on its own).

    This is NOT an exact measure of robot-centre-to-token distance: the
    point sensors are offset slightly from the robot's true centre, and
    this only measures perpendicular distance to the wall's plane, not
    the direct line to a token if it sits laterally along that same
    wall. It's the most reliable distance signal already available
    though, and a reasonable proxy given the robot is normally squared
    up to whatever wall it's inspecting when a token comes into frame."""
    if label == "front":
        return get_front_combined()
    if label == "right":
        return safe_read(ds5)
    return None


def timed_turn(duration_ms, abort_margin=0.6):

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
        return PIT_THRESHOLD + 1 
    img = color_sensor.getImage()
    return color_sensor.imageGetGray(img, color_sensor.getWidth(), 0, 0)


PIT_SATURATION_MAX = 30  # a real black hole is desaturated
                          

def get_color_rgb():
    #Returns (r,g,b) 0-255 from the same downward pixel used by get_color(). 
    if color_sensor is None:
        return (200, 200, 200)
    img = color_sensor.getImage()
    r = color_sensor.imageGetRed(img, color_sensor.getWidth(), 0, 0)
    g = color_sensor.imageGetGreen(img, color_sensor.getWidth(), 0, 0)
    b = color_sensor.imageGetBlue(img, color_sensor.getWidth(), 0, 0)
    return (r, g, b)


def is_real_pit():
    gray = get_color()
    if gray >= PIT_THRESHOLD: #checks if dark enough
        return False
    r, g, b = get_color_rgb()
    saturation = max(r, g, b) - min(r, g, b) #checks if it is black or colourful
    return saturation <= PIT_SATURATION_MAX

_last_color_debug_print = 0.0


def debug_print_color(): #prints colours every 2 secs
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


def debug_print_heading(heading_deg): #prints heading every 2 secs
    global _last_heading_debug_print
    now = robot.getTime()
    if now - _last_heading_debug_print < 2.0:
        return
    _last_heading_debug_print = now
    print(f"[heading debug] heading={heading_deg:.1f} deg  (source: {'gyro' if gyro is not None else 'gps-diff'})")



# LIDAR-based gap finding 
def lidar_widest_gap_heading(min_gap_range=0.15):
    """scan every beam once, tracking the longest unbroken stretch of open 
    readings as you go, then return the angle pointing at the exact middle
    of whichever stretch turned out to be the widest"""
    #points at the middle of the widest open gap 
    if lidar is None:
        return None

    ranges = lidar.getRangeImage()
    n = LIDAR_RES
    fov_deg = math.degrees(LIDAR_FOV)

    def index_to_angle(i):
        return fov_deg / 2 - (i / (n - 1)) * fov_deg

    best_start = None
    best_len = 0
    cur_start = None
    cur_len = 0

    for i in range(n):
        is_open = ranges[i] > min_gap_range and ranges[i] != float("inf") #reading can't be infinity and has to be bigger than 'min_gap_range'
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
        _stuck_ref_pos = pos
        _stuck_timer_start = now
        return False

    if now - _stuck_timer_start > STUCK_TIME:
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
    #Returns True if no NEW cell has been reached in CELL_PROGRESS_TIMEOUT seconds
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
        _last_new_cell_time = now
        return True

    return False


def reset_stuck_trackers():
    global _stuck_timer_start, _stuck_ref_pos, _last_new_cell_time
    now = robot.getTime()
    _stuck_timer_start = now
    _stuck_ref_pos = gps.getValues()
    _last_new_cell_time = now


_consecutive_recovers = 0
_recover_visited_ref = 0


def recover():
    #Break out of a stuck loop. Escalates if it keeps failing in a row: a single spin can
    #fail to free a robot that's genuinely wedged against something on multiple sides (it just
    #grinds in place instead of turning clear), which is also exactly what causes a real
    #Erebus Lack-of-Progress call - the supervisor's own "static for 20s" timer doesn't care
    #that recover() is running, only that the robot's actual position isn't changing.
    global _consecutive_recovers, _recover_visited_ref
    current_count = len(visit_counts)
    if current_count > _recover_visited_ref:
        _consecutive_recovers = 0
    _recover_visited_ref = current_count
    _consecutive_recovers += 1

    if _consecutive_recovers >= 4:
        #Wiggle sequence - alternating short reverse/spin pulses in BOTH directions, like
        #rocking a car out of a tight parking spot, instead of repeating one spin that isn't
        #working.
        print(f"[recover] escalating after {_consecutive_recovers} failed recoveries with no new cell - wiggling free")
        for i in range(3):
            direction = 1 if i % 2 == 0 else -1
            if get_rear() > REAR_WALL_THRESHOLD * 0.5:
                reverse()
                wheel_left.setVelocity(speeds[0])
                wheel_right.setVelocity(speeds[1])
                delay(300)
            spin(direction)
            wheel_left.setVelocity(speeds[0])
            wheel_right.setVelocity(speeds[1])
            delay(350)
        forward()
        wheel_left.setVelocity(speeds[0])
        wheel_right.setVelocity(speeds[1])
        delay(200)
        return

    rear_clear = get_rear() > REAR_WALL_THRESHOLD
    if rear_clear:
        reverse()
        wheel_left.setVelocity(speeds[0])
        wheel_right.setVelocity(speeds[1])
        delay(400)

    gap_heading = lidar_widest_gap_heading()
    if gap_heading is not None:
        direction = 1 if gap_heading > 0 else -1 #sees which direction to turn
    else:
        direction = 1 if safe_read(ds6) > safe_read(ds5) else -1

    if _consecutive_recovers % 2 == 0:
        #Alternate direction every other attempt - if the sensors read the same "best"
        #direction every time because the robot hasn't actually moved since the last
        #attempt, trusting them again just repeats the identical failing spin.
        direction *= -1

    print(f"[recover] attempt={_consecutive_recovers} rear_clear={rear_clear} gap_heading={gap_heading} direction={direction}")
    spin(direction)
    completed = timed_turn(random.randint(500, 900), abort_margin=0.4)
    print(f"[recover] spin completed={completed}")


_last_branch_check = 0.0


def check_side_branch(cell, heading, l, r, front_blocked):
    global _last_branch_check
    now = robot.getTime()
    if front_blocked or now - _last_branch_check < BRANCH_CHECK_COOLDOWN:
        return False
    _last_branch_check = now

    straight_delta = cardinal_delta(heading)
    straight_score = direction_score(cell, straight_delta)
    straight_fresh = is_fresh_direction(cell, straight_delta)
    left_open = l > OPEN_THRESHOLD
    right_open = r > OPEN_THRESHOLD

    best_dir = None
    # Fallback numeric baseline for the known-vs-known case (unchanged from
    # before): only require a branch to clearly beat straight when straight
    # already has real, computed frontier information behind it.
    best_score = straight_score + BRANCH_SCORE_MARGIN if straight_score > 5 else straight_score

    def consider(direction, delta):
        nonlocal best_dir, best_score
        fresh = is_fresh_direction(cell, delta)
        s = direction_score(cell, delta)
        if fresh and not straight_fresh: #candidate is fresh but straight
            # A genuinely unexplored branch wins outright against
            # ALREADY-explored straight-ahead territory, regardless of
            # that territory's computed frontier score. Without this, a
            # small 1-tile pocket (permanently capped at the flat score
            # of 5, since a dead-end has nowhere further to flood into)
            # could never beat a large already-explored open area's
            # score (up to 12) - meaning "more open area" would always
            # win and small coloured-zone alcoves would never actually
            # get visited, exactly the reported symptom.
            if best_dir is None or s >= best_score:
                best_dir, best_score = direction, s
            return
        if s > best_score:
            best_dir, best_score = direction, s

    if left_open:
        consider("left", cardinal_delta(heading + 90))
    if right_open:
        consider("right", cardinal_delta(heading - 90))

    if best_dir is None:
        return False

    if best_dir == "left":
        turn_left()
    else:
        turn_right()
    timed_turn(BRANCH_TURN_MS)
    return True


# ----------------------------------------------------------------------
# Victim / hazard-sign detection and reporting
# ----------------------------------------------------------------------
reported_positions = []


def get_camera_image_array(cam):
    if cam is None:
        return None
    img = cam.getImage()
    if img is None:
        return None
    return np.frombuffer(img, np.uint8).reshape((cam.getHeight(), cam.getWidth(), 4))


@dataclass
class VictimCandidate:
    label: int
    center: tuple
    area: int
    bounding_box: tuple


def find_victim_candidates(image, cam_width, cam_height):
    bgr = image[:, :, :3]
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h, l, s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]

    wall_shadow_mask = (h > 85) & (h < 105) & (l > 15) & (l < 35) & (s > 60) & (s < 80)
    black_mask = ((l < 35) & ~wall_shadow_mask).astype(np.uint8)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)

    image_area = cam_width * cam_height
    min_area = int(image_area * 0.003)
    max_area = int(image_area * 0.4)
    min_aspect, max_aspect = 0.2, 1.4 #min_aspect/max_aspect bound how "letter-shaped" (width-to-height ratio) a candidate blob has to be
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
        cx, cy = x + w // 2, y + h_box // 2
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
    """Finds the white plaque backing the glyph, as a 4-point quad."""
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

    candidate_x, candidate_y = candidate.center
    min_contour_area = candidate.area * 2
    max_contour_area = candidate.area * 30

    if debug:
        print(f"[detect] plaque search: {len(contours)} white contour(s) in ROI, "
              f"need area in ({min_contour_area:.0f}, {max_contour_area:.0f})")

    # Collect every valid 4-point quad, ranked by area, instead of only
    # ever considering the single largest white region. If the largest
    # region doesn't cleanly approximate to 4 points (a partly occluded
    # or cropped plaque edge), the next-best contour is tried instead of
    # discarding the search entirely in favour of the cruder, unwarped
    # fallback rectangle.
    scored_quads = []

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
        if len(quad) != 4:
            if debug:
                print(f"[detect-reject] white contour (area={area:.0f}) didn't approximate to a "
                      f"clean quad ({len(quad)} points)")
            continue
        quad[:, 0] += rx1
        quad[:, 1] += ry1
        if cv2.pointPolygonTest(quad, (float(candidate_x), float(candidate_y)), False) < 0:
            if debug:
                print(f"[detect-reject] quad (area={area:.0f}) found but glyph centre isn't inside it")
            continue
        scored_quads.append((area, quad))

    if not scored_quads:
        return None

    scored_quads.sort(key=lambda t: t[0], reverse=True)
    return scored_quads[0][1]


def warp_plaque(image, plaque_quad, output_size=128):
    ordered = order_quad_points(plaque_quad)
    destination = np.asarray(
        [[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]],
        dtype=np.float32)
    transform = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(image[:, :, :3], transform, (output_size, output_size))


def classify_victim(warped):
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    points = cv2.findNonZero(binary)
    if points is None:
        return None, 0.0
    x, y, w, h = cv2.boundingRect(points)
    if w < 3 or h < 3:
        return None, 0.0
    symbol = binary[y:y + h, x:x + w]
    sh, sw = symbol.shape

    contours, hierarchy = cv2.findContours(symbol, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_count = 0
    best_hole_area = 0
    symbol_area = sh * sw
    if hierarchy is not None:
        for idx, h_entry in enumerate(hierarchy[0]):
            parent = h_entry[3]
            if parent == -1:
                continue
            # Only count this as a genuine hole if its PARENT contour does
            # NOT touch the crop's border. A parent touching the border
            # means it's itself a cropped/cut-off region - most commonly
            # the surrounding WALL bleeding into this crop (confirmed via
            # a real reference image: an Otsu threshold run on a crop that
            # includes wall-plus-plaque picks up the wall/plaque contrast
            # as the dominant split, turning the ENTIRE plaque+glyph area
            # into one big "hole" regardless of what glyph is actually
            # inside it - which silently reports every such crop as Phi
            # ("H"), no matter the true letter. A real glyph stroke, fully
            # interior to the plaque, should never touch the crop's edge.
            px, py, pw, ph = cv2.boundingRect(contours[parent])
            touches_border = px <= 0 or py <= 0 or (px + pw) >= sw or (py + ph) >= sh
            if not touches_border:
                hole_count += 1
                hole_area = cv2.contourArea(contours[idx])
                if hole_area > best_hole_area:
                    best_hole_area = hole_area

    if hole_count >= 1:
        # Confidence scales with how large the enclosed loop is relative
        # to the symbol, rather than a flat 0.8 regardless of size. A
        # tiny few-pixel hole (compression noise, a stray gap in a thin
        # stroke) is a much weaker signal than a loop occupying a real
        # fraction of the glyph - but a genuine enclosed region is still
        # the strongest single signal for Phi, so the floor stays fairly
        # high even for a small hole.
        hole_fraction = best_hole_area / symbol_area if symbol_area > 0 else 0.0
        confidence = min(0.95, 0.65 + hole_fraction * 3.0)
        return "H", confidence

    row_widths = symbol.sum(axis=1) / 255.0  # ink pixel count per row
    band = max(1, h // 5)

    bottom_width = float(np.mean(row_widths[-band:]))
    top_width = float(np.mean(row_widths[:band])) or 1.0

    # Psi (fork wide at the TOP, narrows to a single stem at the bottom)
    # vs Omega (comparatively narrow near the top, flares back out at
    # the bottom "feet") are most different from each other at the very
    # top and very bottom of the glyph. Comparing bottom-to-TOP uses
    # that contrast directly, instead of diluting it against the middle
    # band, which both shapes tend to pass through at a similar
    # moderate width - the previous bottom-vs-middle comparison put the
    # decision boundary right where the two shapes look most alike.
    flare_ratio = bottom_width / top_width if top_width > 0 else 0.0

    if flare_ratio >= 1.0:
        # Bottom is as wide as or wider than the top - consistent with
        # Omega's flared feet.
        confidence = min(0.95, 0.55 + (flare_ratio - 1.0) * 0.5)
        return "U", confidence
    else:
        # Bottom narrower than top - consistent with Psi's stem.
        confidence = min(0.95, 0.55 + (1.0 - flare_ratio) * 0.5)
        return "S", confidence


def approximate_plaque_quad(candidate, pad_ratio=1.8):
 
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

    if plaque_quad is None and candidates:
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
        # The fallback quad is a guessed axis-aligned rectangle, not a
        # real detected plaque - no actual perspective correction, so
        # it's discounted rather than trusted at face value. But it's
        # still reportable: missing a real victim scores nothing either
        # way, so catching it via the lower-quality path beats not
        # catching it at all.
        confidence *= 0.8
    if confidence < 0.5:
        label = None
    if debug:
        print(f"[detect] victim classification: {label} confidence={confidence:.2f} "
              f"(fallback quad: {used_fallback})")
    return label, confidence


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
    # Hue is checked BEFORE the lightness gate, not after. Pure green
    # renders at a meaningfully lower lightness than red/yellow/blue at
    # the same saturation in this simulator - checking `l < 25` first
    # (as before) caught genuinely green pixels and classified them as
    # black before their hue was ever examined. The saturation gate
    # below still keeps true black/grey (near-zero saturation) from
    # coincidentally matching a hue band.
    hue = h * 2

    if s >= 180:
        if hue <= 15 or hue >= 345:
            return "red"
        if 45 <= hue <= 75:
            return "yellow"
        if 100 <= hue <= 140:
            return "green"
        if 210 <= hue <= 270:
            return "blue"

    if l < 25:
        return "black"

    return None


def cluster_ellipses(ellipses):
   
    clusters = []
    for ellipse in ellipses:
        matched = False
        for cluster in clusters:
            mean_cx = np.mean([e.cx for e in cluster])
            mean_cy = np.mean([e.cy for e in cluster])
            distance = np.hypot(ellipse.cx - mean_cx, ellipse.cy - mean_cy)
            ref_size = max(ellipse.major, max(e.major for e in cluster))
            merge_radius = max(ref_size * 0.35, 3.0)
            if distance <= merge_radius:
                cluster.append(ellipse)
                matched = True
                break
        if not matched:
            clusters.append([ellipse])

    result = []
    for cluster in clusters:
        # Prefer the "combined" mask's own ellipse as the outer/sizing
        # reference when one exists in this cluster - that mask is the
        # union of all 5 ring colours, specifically meant to capture the
        # target's TRUE full outer boundary. Falling back to "whichever
        # single ellipse fit largest" (any individual colour mask) risks
        # picking one that's actually smaller than the real target if a
        # single ring's own mask happened to fit slightly larger due to
        # noise - every ring_bounds fraction is sized against this
        # reference, so an undersized one means the sampling band meant
        # for the TRUE outermost ring lands short, re-reading an inner
        # ring's colour instead of ever reaching the real outer one.
        combined_members = [e for e in cluster if e.source_colour == "combined"]
        if combined_members:
            outer = max(combined_members, key=lambda e: e.major * e.minor)
        else:
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
    # Inset the sampled radius range away from this band's own edges, rather
    # than spanning its full inner_frac..outer_frac width. Confirmed via a
    # real reference image: rendered hazard targets have a thin BLACK
    # OUTLINE STROKE separating each ring visually - sampling right up to a
    # band's edge (where that stroke sits) lets it contaminate the majority-
    # colour vote as a false "black" reading, even when the ring's true
    # fill colour is something else entirely. Shrinking each band by ~15%
    # of the band's own width on each side keeps samples away from that
    # border while still covering enough of the band for a reliable vote.
    band_width = outer_frac - inner_frac
    inset = band_width * 0.15
    radius_samples = np.linspace(inner_frac + inset, outer_frac - inset, radius_count)
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
    if len(colours) < 4:
        # Lowered from 10 - at this camera's 40x32 resolution, a modestly
        # sized target's outer ring bands can easily have fewer than 10
        # usable pixels (especially near a frame edge), causing an outright
        # rejection instead of a lower-confidence read. 4 is still enough to
        # get a meaningful majority-colour vote from, just not as strict a
        # floor as was reasonable on a higher-resolution camera.
        return RingSampleResult(None, 0.0, len(colours))
    dominant_colour, count = histogram.most_common(1)[0]
    agreement = count / len(colours)
    return RingSampleResult(dominant_colour, agreement, len(colours))


def classify_candidate(rings):
    """Returns (type_or_None, sum_confidence_multiplier). Allows the
    computed ring-value sum to be off by exactly 1 from a valid type,
    at a discounted confidence, rather than requiring an exact match -
    at 40x32 resolution a single ring being misread by aliasing/blur is
    plausible and would otherwise throw away an otherwise-solid, real
    detection. Getting MORE correct detections matters more than being
    maximally conservative - a missed real target scores nothing either
    way, so recall is weighted over precision here."""
    total = sum(RING_COLOR_VALUES[colour] for colour in rings)
    if total in HAZARD_SUM_TO_TYPE:
        return HAZARD_SUM_TO_TYPE[total], 1.0
    best_type, best_dist = None, 999
    for valid_sum, t in HAZARD_SUM_TO_TYPE.items():
        d = abs(total - valid_sum)
        if d < best_dist:
            best_dist, best_type = d, t
    if best_dist == 1:
        return best_type, 0.7
    return None, 1.0


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

    if debug:
        print(f"[detect] ring-sizing reference: source={outer.source_colour} "
              f"major={major:.1f} minor={minor:.1f} (cluster has "
              f"{len(cluster.members)} member(s): "
              f"{[m.source_colour for m in cluster.members]})")

    ring_bounds = [(0.00, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.00)]
    ellipse_area = np.pi * a * b
    occupancy = outer.area / ellipse_area if ellipse_area > 0 else 0

    if occupancy < 0.58:
        # Moderated from 0.65 (which was itself tightened from 0.55 to cut
        # false positives after the clustering fix). Loosened back down a
        # bit given the explicit priority: getting MORE correct detections
        # matters more than being maximally conservative - a missed real
        # target scores nothing either way, so recall is weighted over
        # precision here (still above the original 0.55 that was
        # compensating for a clustering bug that's since been fixed).
        if debug:
            print(f"[detect-reject] occupancy too low: {occupancy:.2f} (need >=0.58)")
        return CandidateEvaluation(result, 0.0, [], 0.0, False)

    if len(cluster.members) < 2:
        # A genuine concentric-ring target should be picked up by at least
        # its own colour mask AND the combined mask - a single-member
        # cluster this far into the pipeline is much more likely to be an
        # isolated noise contour than a real target.
        if debug:
            print(f"[detect-reject] only 1 ellipse in cluster - too likely to be noise, not a real target")
        return CandidateEvaluation(result, 0.0, [], 0.0, False)

    ratio = min(major, minor) / max(major, minor)
    populate_navigation_geometry(result, image, cx, cy, major, minor)

    image_height = image.shape[0]
    vertical_position = cy / image_height
    vertical_penalty = 0.2 if vertical_position > 0.80 else 1.0

    # Named so it can be reused below when remapping geometry_score into
    # the final confidence, instead of just as this pass/fail gate.
    GEOMETRY_SCORE_MIN = 0.28

    geometry_score = (outer.circularity * ratio * min(len(cluster.members) / 4.0, 1.0)
                       * occupancy * vertical_penalty)
    if geometry_score < GEOMETRY_SCORE_MIN:
        # Moderated from 0.32 for the same recall-priority reasoning as
        # the occupancy threshold above.
        if debug:
            print(f"[detect-reject] geometry score too weak: {geometry_score:.2f} (need >={GEOMETRY_SCORE_MIN}) "
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

    target_type, sum_confidence = classify_candidate(rings)
    if target_type is None:
        if debug:
            print(f"[detect-reject] ring sum doesn't match a known type: rings={rings} "
                  f"sum={sum(RING_COLOR_VALUES[c] for c in rings)}")
        return CandidateEvaluation(result, 0.0, agreements, geometry_score, False)

    result.found = True
    result.type = target_type
    result.rings = rings
    mean_agreement = sum(agreements) / len(agreements)

    # geometry_score already had to clear GEOMETRY_SCORE_MIN to reach
    # this point (it's a pass/fail gate above). Using it AGAIN here as a
    # raw multiplier double-penalizes it: a genuinely valid detection
    # that only just clears the gate (e.g. geometry_score=0.30) would
    # have its confidence crushed by that same ~0.30 factor a second
    # time, even with excellent ring-colour agreement and an exact
    # ring-sum match - which is exactly what was happening to correct
    # detections landing just above the geometry floor. Remapping it
    # relative to its own threshold instead - a bare pass contributes a
    # moderate penalty, well clear of the gate contributes close to
    # none - keeps geometry meaningfully weighted without re-punishing
    # detections it already approved.
    geometry_factor = 0.5 + 0.5 * max(0.0, min(1.0,
        (geometry_score - GEOMETRY_SCORE_MIN) / GEOMETRY_SCORE_MIN))

    result.confidence = max(0.0, min(1.0, mean_agreement * geometry_factor * sum_confidence))

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
    return CognitiveTargetDetection()


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

    reset_stuck_trackers()


_last_report_time = -999.0


def already_reported_nearby(pos):
    return any(math.hypot(pos[0] - rx, pos[2] - rz) < REPORT_DEDUPE_DIST for rx, rz in reported_positions)


_last_victim_check = 0.0
VICTIM_CHECK_INTERVAL = 0.2


def run_detection_for_camera(cam, label):
    """Runs the full tiered detection pipeline for a single camera.
    Returns True if it reported something (caller should stop checking
    other cameras this step).

    No longer gated on a point-distance-sensor reading before even
    looking at the image. That gate (requiring the robot within
    FRONT_APPROACH_THRESHOLD/SIDE_APPROACH_THRESHOLD of a wall per a
    DIFFERENT sensor before the camera was even checked) meant a sign
    mounted on a wall the robot merely drives PAST rather than INTO
    (i.e. anywhere except a dead-end/T-junction for the front camera,
    or within a very tight distance for the side camera) was essentially
    never close enough to trigger detection at all - which is exactly
    what "camera isn't detecting anything" looks like. The tier-1 image
    check below (contour detection on the actual frame) is cheap enough
    to just run every cycle and is a direct check of what the camera
    sees, rather than a proxy via a different sensor."""
    if cam is None:
        return False

    pos = gps.getValues()
    if already_reported_nearby(pos):
        return False

    img = get_camera_image_array(cam)
    if img is None:
        return False
    cam_w, cam_h = cam.getWidth(), cam.getHeight()
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    masks = build_target_mask(bgr)
    preliminary_hazard_candidates = extract_ellipse_candidates(masks, cam_w, cam_h)
    preliminary_victims = find_victim_candidates(img, cam_w, cam_h)

    if not preliminary_hazard_candidates and not preliminary_victims:
        return False

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
            dist = distance_to_wall(label)
            if dist is not None and dist > HALF_TILE_DISTANCE:
                print(f"[detect-reject:{label}] cognitive target identified but robot is "
                      f"{dist:.3f}m from the wall (> {HALF_TILE_DISTANCE}m half-tile limit) "
                      f"- not reporting yet, will retry as it gets closer")
                return False
            report(hazard.type)
            return True
        else:
            print(f"[detect-reject:{label}] confidence {hazard.confidence:.2f} below "
                  f"{MIN_HAZARD_CONFIDENCE} threshold - not reporting")
        return False

    letter_type, confidence = detect_victim(img, cam_w, cam_h, debug=True)
    if letter_type is not None:
        dist = distance_to_wall(label)
        if dist is not None and dist > HALF_TILE_DISTANCE:
            print(f"[detect-reject:{label}] victim '{letter_type}' identified but robot is "
                  f"{dist:.3f}m from the wall (> {HALF_TILE_DISTANCE}m half-tile limit) "
                  f"- not reporting yet, will retry as it gets closer")
            return False
        report(letter_type)
        return True
    return False


def check_for_victim_or_hazard():
    """Runs detection for BOTH cameras (front, then right), whichever
    finds something first."""
    global _last_victim_check
    now = robot.getTime()
    if now - _last_victim_check < VICTIM_CHECK_INTERVAL:
        return
    _last_victim_check = now

    if now - _last_report_time < MIN_REPORT_INTERVAL:
        return

    if run_detection_for_camera(camera, "front"):
        return
    run_detection_for_camera(camera_right, "right")


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
while robot.step(TIME_STEP) != -1:
    pos = gps.getValues()
    cell = mark_visited(pos)

    if receiver is not None:
        while receiver.getQueueLength() > 0:
            receiver.nextPacket()

    if update_stuck_tracker(pos) or update_cell_progress_tracker():
        recover()
        continue

    f = get_front()

    if f < APPROACH_SLOWDOWN_THRESHOLD:
        speeds[0] = APPROACH_SLOWDOWN_SPEED
        speeds[1] = APPROACH_SLOWDOWN_SPEED
    else:
        forward()

    l = safe_read(ds6)
    r = safe_read(ds5)

    front_blocked = get_front_combined() < WALL_THRESHOLD  # LIDAR-augmented check - see
                                                            # get_front_combined()'s docstring
    left_blocked = l < WALL_THRESHOLD
    right_blocked = r < WALL_THRESHOLD

    heading = current_heading()
    debug_print_heading(heading)

    mark_walls(cell, heading, front_blocked, left_blocked, right_blocked)
    print_map()
    debug_print_color()

    check_for_victim_or_hazard()

    if check_side_branch(cell, heading, l, r, front_blocked):
        continue

    if check_frontier_pull(cell, heading):
        continue

    if front_blocked:
        if get_rear() > REAR_WALL_THRESHOLD:
            reverse()
            timed_turn(250, abort_margin=0.0)

        gap_heading = lidar_widest_gap_heading()
        chosen_direction = None

        if gap_heading is not None:
            if gap_heading > 10 and not left_blocked:
                turn_left()
                chosen_direction = "left"
            elif gap_heading < -10 and not right_blocked:
                turn_right()
                chosen_direction = "right"
            else:
                gap_heading = None

        if gap_heading is None:
            left_score = direction_score(cell, cardinal_delta(heading + 90))
            right_score = direction_score(cell, cardinal_delta(heading - 90))

            if left_blocked and not right_blocked:
                turn_right()
                chosen_direction = "right"
            elif right_blocked and not left_blocked:
                turn_left()
                chosen_direction = "left"
            elif left_score > right_score:
                turn_left()
                chosen_direction = "left"
            elif right_score > left_score:
                turn_right()
                chosen_direction = "right"
            else:
                if random.random() < 0.5:
                    turn_left()
                    chosen_direction = "left"
                else:
                    turn_right()
                    chosen_direction = "right"

        timed_turn(450)

        # If that turn led straight into ANOTHER wall (a short dead-end just around the
        # corner), try the OTHER side before falling through to reverse/recover - without
        # this, a bad turn choice at a tight junction just repeats the same scoring logic
        # next iteration or sends the robot into recover() instead of trying the one option
        # at this junction that hasn't been tried yet.
        if get_front() < WALL_THRESHOLD and chosen_direction is not None:
            other_direction = "right" if chosen_direction == "left" else "left"
            print(f"[nav] turned {chosen_direction} into another wall - trying {other_direction} "
                  f"before backtracking")
            if other_direction == "left":
                turn_left()
            else:
                turn_right()
            timed_turn(450)
    else:
        if left_blocked and right_blocked:
            if r > l:
                nudge_right()
            elif l > r:
                nudge_left()
        elif left_blocked:
            turn_right()
        elif right_blocked:
            turn_left()

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