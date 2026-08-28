"""
Erebus / Webots Rescue Simulation - Navigation Controller
-----------------------------------------------------------
Pure navigation (no victim detection/reporting here).
 
Strategy:
  1. Low-level safety layer: distance-sensor based wall/obstacle avoidance
     and black-floor (pit) avoidance - this stops the robot crashing or
     driving into a hole.
  2. High-level exploration layer: an occupancy grid (built from GPS
     readings, with a per-cell visit COUNT rather than a plain visited/
     not-visited flag) tracks where the robot has already been. At each
     decision point it's biased toward the direction that leads to
     unexplored or less-visited space.
  3. Side-branch detection: this is what stops the robot driving straight
     past an opening (a side corridor, a doorway into a room) while
     following a wall. Every BRANCH_CHECK_COOLDOWN seconds, even while
     cruising with nothing directly ahead, it checks whether the left or
     right side sensor shows open space AND that side scores meaningfully
     better (more unexplored cells) than continuing straight - if so, it
     turns into the branch. Without this check, a controller that only
     reacts when the FRONT sensor hits a wall will ignore every side
     opening it passes and miss whole rooms/branches.
  4. Stuck / lack-of-progress recovery: if the robot's position hasn't
     changed meaningfully for a while, it forces a randomised turn to
     break out of a loop.
 
Sensor / device names below match YOUR custom robot JSON
(MyAwesomeRobot (5).json): wheel1/wheel2 motors, distance sensor1-8,
colour_sensor, gps, lidar, gyro. Device names are looked up with a
fallback list (get_device()) since it's not 100% certain whether Webots
appends " motor" to wheel customNames or uses them verbatim - if one
name fails, the next candidate is tried automatically and a warning is
printed so you can see which one actually resolved.
"""
 
import math
import random
from controller import Robot
 
# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TIME_STEP = 32
MAX_VELOCITY = 6.28
 
WALL_THRESHOLD = 0.05          # distance sensor value below which "wall detected"
OPEN_THRESHOLD = 0.25          # distance sensor value above which "side is open" (potential branch)
PIT_THRESHOLD = 80             # color sensor grayscale value below which "black floor"
REAR_WALL_THRESHOLD = 0.05     # rear sensor value below which "don't reverse, something's there"
 
CELL_SIZE = 0.06               # metres per grid cell (~half a tile; tune to your maze)
STUCK_TIME = 4.0                # seconds with no meaningful movement before recovery
STUCK_DIST = 0.02               # metres - movement below this counts as "not moved"
CELL_PROGRESS_TIMEOUT = 15.0     # seconds with no NEW cell visited before forcing recovery -
                                 # catches micro-oscillation within one cell that never
                                 # trips STUCK_DIST (small shuffles >2cm reset that timer
                                 # even though the robot never actually crosses a cell
                                 # boundary and exploration completely stalls)
GYRO_DEADBAND = 0.02             # rad/s - yaw rates below this are treated as noise/bias, not real turning
 
BRANCH_CHECK_COOLDOWN = 1.5     # seconds between side-branch evaluations
BRANCH_SCORE_MARGIN = 1         # how much better a side branch's score must be to divert into it
BRANCH_TURN_MS = 550            # how long to turn toward a chosen branch
 
TURN_SPEED_FAST = 0.6 * MAX_VELOCITY
NUDGE_DIFF = 0.15 * MAX_VELOCITY  # gentle correction, not a hard turn - for centering in narrow gaps
TURN_SPEED_SLOW = -0.2 * MAX_VELOCITY
SPIN_SPEED = 0.6 * MAX_VELOCITY
 
# ----------------------------------------------------------------------
# Robot / device setup
# ----------------------------------------------------------------------
robot = Robot()
 
 
def list_all_devices():
    """Prints every device name Webots actually assigned to this robot.
    This is ground truth - if a getDevice() lookup below ever fails
    again, the real name is in this list, not in another guess."""
    print("=== Devices found on robot ===")
    for i in range(robot.getNumberOfDevices()):
        d = robot.getDeviceByIndex(i)
        print(f"  [{i}] {d.getName()!r}")
    print("==============================")
 
 
list_all_devices()
 
 
def _normalize(name):
    return name.lower().replace(" ", "").replace("_", "")
 
 
def get_device(*candidate_names):
    """Try each candidate device name in turn (exact match first). If
    none match exactly, fall back to a case/space/underscore-insensitive
    search across every device actually on the robot - this covers
    naming differences like 'distance sensor6' vs 'Distance_Sensor6'
    without needing another blind guess. Prints a warning if nothing
    matches at all."""
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
 
 
# wheel1 sits at x=260 (positive X = right, per the distance-sensor
# convention below) -> right wheel. wheel2 sits at x=-260 -> left wheel.
wheel_right = get_device("wheel1", "wheel1 motor")
wheel_left = get_device("wheel2", "wheel2 motor")
wheel_left.setPosition(float("inf"))
wheel_right.setPosition(float("inf"))
 
# Your robot's 8 distance sensors, by position:
#   DS1 (x=-213,z=-309) / DS2 (x=-107,z=-355): front-left pair
#   DS3 (x=108, z=-355) / DS4 (x=220, z=-309): front-right pair
#   DS5 (x=370, z=0):    true right-side sensor (faces world +X)
#   DS6 (x=-370,z=0):    true left-side sensor  (faces world -X)
#   DS7 (x=-107,z=355) / DS8 (x=108,z=355): rear pair (faces world +Z)
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
 
# Downward colour sensor - verified facing straight down (world -Y) from
# its rotation values, so pit-avoidance below can be trusted.
color_sensor = get_device("colour_sensor")
if color_sensor is not None:
    color_sensor.enable(TIME_STEP)
 
gps = get_device("gps")
gps.enable(TIME_STEP)
 
# LIDAR - shares the same "faces forward" baseline rotation as the front
# sensors, so its zero-angle point lines up with the robot's heading.
lidar = get_device("lidar")
if lidar is not None:
    lidar.enable(TIME_STEP)
    LIDAR_FOV = lidar.getFov()
    LIDAR_RES = lidar.getHorizontalResolution()
else:
    LIDAR_FOV = None
    LIDAR_RES = None
 
# Gyro - same baseline rotation, which maps local Z to world vertical
# (Y). A robot turning left/right rotates about the world-vertical axis,
# so index [2] of getValues() is the yaw rate we integrate for heading.
gyro = get_device("gyro")
if gyro is not None:
    gyro.enable(TIME_STEP)
 
# Receiver picks up lack-of-progress notifications from the supervisor
receiver = get_device("receiver")
if receiver is not None:
    receiver.enable(TIME_STEP)
 
speeds = [MAX_VELOCITY, MAX_VELOCITY]
 
# ----------------------------------------------------------------------
# Occupancy grid - persistent wall map + frontier flood-fill
# ----------------------------------------------------------------------
# This is the real upgrade over the old version: instead of guessing
# "unexplored direction" from a single sensor snapshot projected in a
# straight line, we now build a persistent map of which cardinal
# direction out of each cell is blocked by a wall (accumulated over the
# whole run, from every pass through that cell) and use a proper
# flood-fill to count how much genuinely reachable unexplored territory
# lies beyond each candidate direction.
#
# Assumption: Erebus mazes are grid-aligned to the world X/Z axes, so we
# snap the robot's continuous heading to the nearest cardinal (0/90/
# 180/270) before recording a wall. This matches the tile-based maze
# layout.
CARDINALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
 
visit_counts = {}          # cell -> number of times physically visited
grid_walls = {}            # cell -> set of (dx,dz) deltas known to be blocked
 
 
def get_cell(pos):
    """Convert an (x, y, z) GPS reading into a grid cell coordinate."""
    return (round(pos[0] / CELL_SIZE), round(pos[2] / CELL_SIZE))
 
 
def mark_visited(pos):
    cell = get_cell(pos)
    visit_counts[cell] = visit_counts.get(cell, 0) + 1
    return cell
 
 
def snap_cardinal(deg):
    """Round a heading to the nearest of 0/90/180/270."""
    return (round(deg / 90.0) % 4) * 90
 
 
def cardinal_delta(deg):
    d = snap_cardinal(deg)
    return {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[d]
 
 
def mark_walls(cell, heading_deg, front_blocked, left_blocked, right_blocked):
    """Record which absolute directions out of this cell are walled off,
    based on the current sensor readings and heading. Accumulates across
    every pass through the cell (a direction once seen open stays open;
    a direction seen blocked is remembered even if a later pass, at a
    different heading, doesn't re-check it)."""
    walls = grid_walls.setdefault(cell, set())
    if front_blocked:
        walls.add(cardinal_delta(heading_deg))
    if left_blocked:
        walls.add(cardinal_delta(heading_deg + 90))
    if right_blocked:
        walls.add(cardinal_delta(heading_deg - 90))
 
 
def frontier_flood_score(start_cell, max_nodes=12):
    """
    Flood-fills outward from start_cell through cells we've already
    visited AND that have no known wall in the direction of travel,
    counting how many still-unvisited "frontier" cells are reachable.
    This is grounded in the actual wall map, so it won't say a direction
    is promising if we already know a wall blocks it - unlike a straight-
    line distance projection, it follows real corridors around corners.
 
    max_nodes is intentionally small (12, not the original 40): with a
    large cap, a robot circling a closed loop of known corridors can
    reach the SAME distant frontier cells from either direction around
    the loop, making both directions score almost identically and
    causing it to just keep circling the loop forever (confirmed by
    logs showing visited-cell count frozen while heading drifted). A
    small cap keeps the score reflecting genuinely NEARBY unexplored
    territory, so the two directions actually differ.
    """
    seen = {start_cell}
    queue = [start_cell]
    score = 0
    while queue and len(seen) < max_nodes:
        cell = queue.pop(0)
        blocked = grid_walls.get(cell, set())
        for d in CARDINALS:
            if d in blocked:
                continue  # known wall - don't cross it
            neighbour = (cell[0] + d[0], cell[1] + d[1])
            if neighbour in seen:
                continue
            seen.add(neighbour)
            if neighbour in visit_counts:
                queue.append(neighbour)  # known territory - keep expanding
            else:
                score += 1  # frontier: unexplored cell reachable from here
    return score
 
 
def direction_score(cell, delta):
    """How attractive is heading in this cardinal direction from `cell`?
    Unknown neighbours (never visited) get a flat attractive score since
    we have no map data yet to judge them by; known neighbours are
    scored by how much unexplored territory a flood-fill finds beyond
    them."""
    if delta in grid_walls.get(cell, set()):
        return -1  # known wall - never attractive
    neighbour = (cell[0] + delta[0], cell[1] + delta[1])
    if neighbour not in visit_counts:
        return 5
    return frontier_flood_score(neighbour)
 
 
_last_map_print = 0.0
 
 
def print_map():
    """Prints a simple ASCII view of the known map to the console so you
    can watch coverage build up while the sim runs. '#' = visited,
    '?' = known to exist (a wall was checked from a neighbour) but not
    yet driven through, '.' = outside anything seen so far."""
    global _last_map_print
    now = robot.getTime()
    if now - _last_map_print < 10.0:
        return
    _last_map_print = now
 
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
    Heading estimate. Uses the gyro if available (integrating yaw rate
    over time) - this is far more reliable than the GPS-difference
    fallback, especially while turning in place or moving slowly, since
    it updates every step regardless of whether the robot is translating.
    Falls back to the old GPS-difference method if no gyro is present.
    """
    global _last_heading_pos, _last_heading_deg
 
    if gyro is not None:
        # This gyro shares the same baseline rotation as the front
        # sensors, which maps local Z to world vertical - so index [2]
        # is the yaw rate (rotation about the vertical axis).
        # Deadband: your logs show heading creeping upward (~0.1-0.2
        # deg per print) even when the robot should be sitting still -
        # that's gyro bias/noise, not real rotation. Ignoring tiny yaw
        # rates stops that drift from accumulating into a wrong heading
        # over the length of a match.
        yaw_rate = gyro.getValues()[2]
        if abs(yaw_rate) > GYRO_DEADBAND:
            _last_heading_deg = (_last_heading_deg + math.degrees(yaw_rate * TIME_STEP / 1000.0)) % 360
        return _last_heading_deg
 
    # Fallback: estimate heading from consecutive GPS positions.
    pos = gps.getValues()
    if _last_heading_pos is None:
        _last_heading_pos = pos
        return 0.0
    dx = pos[0] - _last_heading_pos[0]
    dz = pos[2] - _last_heading_pos[2]
    if math.hypot(dx, dz) < 0.01:
        # Too little movement to get a reliable heading - keep the last one
        return _last_heading_deg
    heading = math.degrees(math.atan2(dz, dx))
    _last_heading_pos = pos
    _last_heading_deg = heading
    return heading
 
 
_last_heading_pos = None
_last_heading_deg = 0.0
 
# ----------------------------------------------------------------------
# Movement helpers
# ----------------------------------------------------------------------
 
 
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
    """Returns a sensor's value, or a safe 'nothing detected' default if
    the device object is missing (None) - lets the robot keep running in
    a degraded but non-crashing way if one lookup failed, instead of
    throwing mid-run. Check the printed device list at startup to find
    and fix the real cause."""
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
 
 
if color_sensor is None:
    print("!!! CRITICAL: colour sensor device not found - pit avoidance is DISABLED. "
          "Check the device list printed above for the real name and fix get_device('colour_sensor') below. !!!")
 
_last_color_debug_print = 0.0
 
 
def debug_print_color():
    """Prints the raw colour sensor reading every 2s so you can confirm
    the sensor is actually being read (not silently defaulted) and
    check PIT_THRESHOLD is calibrated against real values, not guessed."""
    global _last_color_debug_print
    now = robot.getTime()
    if now - _last_color_debug_print < 2.0:
        return
    _last_color_debug_print = now
    print(f"[color debug] raw={get_color()} threshold={PIT_THRESHOLD}")
 
 
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
 
    forward()
 
    l = safe_read(ds6)   # true left-side sensor
    r = safe_read(ds5)   # true right-side sensor
    f = get_front()       # min across all 4 front sensors
 
    front_blocked = f < WALL_THRESHOLD
    left_blocked = l < WALL_THRESHOLD
    right_blocked = r < WALL_THRESHOLD
 
    heading = current_heading()
    debug_print_heading(heading)
 
    # Record what we can see from this cell into the persistent wall map
    mark_walls(cell, heading, front_blocked, left_blocked, right_blocked)
    print_map()
    debug_print_color()
 
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
    if get_color() < PIT_THRESHOLD:
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