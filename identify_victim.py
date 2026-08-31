import cv2
import numpy as np
from controller import Robot, Camera
from dataclasses import dataclass
import typing

# TODO: Handle fake victims

robot = Robot()
camera = typing.cast(Camera, robot.getDevice("camera1"))

# TODO: Change back
timestep = int(robot.getBasicTimeStep()) * 20
camera.enable(timestep)

IMAGE_WIDTH = camera.getWidth()
IMAGE_HEIGHT = camera.getHeight()
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT

cv2.startWindowThread()


@dataclass
class VictimCandidate:
    label: int
    center: tuple[int, int]
    area: int
    bounding_box: tuple[int, int, int, int]
    score: float


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    points = points[np.argsort(angles)]
    top_left_index = np.argmin(points[:, 0] + points[:, 1])

    points = np.roll(points, -top_left_index, axis=0)
    return points


def find_victim_candidates(
    image: np.ndarray, debug: bool = False
) -> list[VictimCandidate]:

    bgr = image[:, :, :3]
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    wall_shadow_mask = (h > 85) & (h < 105) & (l > 15) & (l < 35) & (s > 60) & (s < 80)

    BLACK_THRESHOLD = 35
    black_mask = ((l < BLACK_THRESHOLD) & ~wall_shadow_mask).astype(np.uint8)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        black_mask, connectivity=8
    )

    BLACK_SHAPE_MINIMUM_AREA = int(IMAGE_AREA * 0.003)
    BLACK_SHAPE_MAXIMUM_AREA = int(IMAGE_AREA * 0.4)
    BLACK_SHAPE_MINIMUM_ASPECT = 0.2
    BLACK_SHAPE_MAXIMUM_ASPECT = 1.4
    CENTRE_MAXIMUM_VERTICAL_LOCATION = 0.75

    candidates: list[VictimCandidate] = []

    debug_img: np.ndarray | None = None

    if debug:
        debug_img = bgr.copy()

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < BLACK_SHAPE_MINIMUM_AREA or area > BLACK_SHAPE_MAXIMUM_AREA:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx = x + h // 2
        cy = y + h // 2

        aspect = w / h
        if aspect < BLACK_SHAPE_MINIMUM_ASPECT or aspect > BLACK_SHAPE_MAXIMUM_ASPECT:
            continue

        vertical_pos = cy / IMAGE_HEIGHT
        if vertical_pos > CENTRE_MAXIMUM_VERTICAL_LOCATION:
            continue

        vertical_score = 1 - max(0, vertical_pos - 0.5)  # 0.75 - 1

        candidates.append(
            VictimCandidate(
                label=label,
                center=(cx, cy),
                area=area,
                bounding_box=(x, y, w, h),
                score=vertical_score,
            )
        )

        if debug and debug_img is not None:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    if debug:
        cv2.namedWindow("Victim Black Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Victim Black Mask", black_mask * 255)

        if debug_img is not None:
            cv2.namedWindow("Victim Black Components", cv2.WINDOW_NORMAL)
            cv2.imshow("Victim Black Components", debug_img)

            def mouse(event, x, y, _flags, _param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(hls[y, x])

            cv2.setMouseCallback("Victim Black Components", mouse)

        print(f"Victim candidates: " f"{len(candidates)}")

    return candidates


def find_plaque_quad(
    image: np.ndarray, candidate: VictimCandidate, debug_img: np.ndarray | None = None
) -> np.ndarray | None:

    bgr = image[:, :, :3]
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    l = hls[:, :, 1]
    x, y, w, h = candidate.bounding_box

    pad = int(max(w, h) * 2)

    rx1 = max(0, x - pad)
    ry1 = max(0, y - pad)

    rx2 = min(bgr.shape[1], x + w + pad)
    ry2 = min(bgr.shape[0], y + h + pad)

    roi_lightness = l[ry1:ry2, rx1:rx2]

    WHITE_THRESHOLD = 140
    white_mask = (roi_lightness > WHITE_THRESHOLD).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_quad: np.ndarray | None = None
    best_area = 0.0

    candidate_x, candidate_y = candidate.center

    MINIMUM_CONTOUR_AREA = candidate.area * 2
    print("Contours found:", len(contours))
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MINIMUM_CONTOUR_AREA:
            print("CONTOUR FAILURE: area is", area)
            continue

        if debug_img is not None:
            cv2.drawContours(debug_img, [contour], -1, (0, 0, 255), 1)

        hull = cv2.convexHull(contour)
        quad = (
            cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            .reshape(-1, 2)
            .astype(np.float32)
        )
        if debug_img is not None:
            for point in quad:
                cv2.circle(debug_img, tuple(point.astype(int)), 2, (0, 0, 255), -1)
        if len(quad) < 4:
            print("CONTOUR FAILURE: len is", len(quad))
            continue

        quad[:, 0] += rx1
        quad[:, 1] += ry1

        inside = cv2.pointPolygonTest(
            quad, (float(candidate_x), float(candidate_y)), False
        )

        if inside < 0:
            print("CONTOUR FAILURE: inside is", inside)
            continue

        if area > best_area:
            best_area = area
            best_quad = quad

    if debug_img is not None and best_quad is not None:
        plaque_center_x = int(np.mean(best_quad[:, 0]))
        plaque_center_y = int(np.mean(best_quad[:, 1]))

        cv2.circle(debug_img, (plaque_center_x, plaque_center_y), 5, (255, 0, 255), -1)

    return best_quad


def get_plaque_view_correction(plaque_quad: np.ndarray) -> str | None:
    x_min = float(np.min(plaque_quad[:, 0]))
    x_max = float(np.max(plaque_quad[:, 0]))

    left = x_min <= IMAGE_WIDTH // 64
    right = x_max >= IMAGE_WIDTH - IMAGE_WIDTH // 64
    if left and right:
        return "back"
    elif left:
        return "left"
    elif right:
        return "right"
    return None


def warp_plaque(
    image: np.ndarray,
    plaque_quad: np.ndarray,
    output_size: int = 128,
    debug_img: np.ndarray | None = None,
) -> np.ndarray:
    ordered = order_quad_points(plaque_quad)

    if debug_img is not None:
        for i, point in enumerate(ordered):
            x, y = point.astype(int)
            cv2.circle(debug_img, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(
                debug_img,
                str(i),
                (x + 5, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    destination = np.asarray(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, transform, (output_size, output_size))

    return warped


TEMPLATES = {
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
}


def classify_victim(warped: np.ndarray) -> tuple[str | None, float]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    points = cv2.findNonZero(binary)

    if points is None:
        return None, 0.0

    x, y, w, h = cv2.boundingRect(points)

    symbol = binary[y : y + h, x : x + w]

    symbol = cv2.resize(symbol, (16, 16), interpolation=cv2.INTER_NEAREST)

    symbol = (symbol > 0).astype(np.uint8)

    value = 0

    for bit in symbol.flatten():
        value = (value << 1) | int(bit)

    best_label: str | None = None
    best_distance = float("inf")

    for label, template_list in TEMPLATES.items():
        for template in template_list:
            distance = (value ^ template).bit_count()

            if distance < best_distance:
                best_distance = distance
                best_label = label

    TOTAL_BITS = 16 * 16

    confidence = (TOTAL_BITS - best_distance) / TOTAL_BITS

    return best_label, confidence


def detect_victim(
    image: np.ndarray, debug: bool = False
) -> str | tuple[str | None, float] | None:
    candidates = find_victim_candidates(image, debug=True)
    debug_img: np.ndarray | None = None
    if debug:
        print(candidates)
        debug_img = image[:, :, :3].copy()
    plaque_quad = None
    for candidate in candidates:
        plaque_quad = find_plaque_quad(image, candidate, debug_img)

    if plaque_quad is None:
        if debug:
            print("cannot find plaque quad")
            cv2.waitKey(1)
        return None

    if debug and debug_img is not None:
        cv2.polylines(debug_img, [plaque_quad.astype(np.int32)], True, (0, 255, 255), 1)
        cv2.namedWindow("Plaque Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Plaque Detection", debug_img)

    if len(plaque_quad) > 4:
        correction = get_plaque_view_correction(plaque_quad)
        if debug:
            cv2.waitKey(1)
        return correction

    warped = warp_plaque(image[:, :, :3], plaque_quad, debug_img=debug_img)

    if debug and debug_img is not None:
        cv2.namedWindow("Warped Plaque", cv2.WINDOW_NORMAL)
        cv2.imshow("Warped Plaque", warped)
        cv2.waitKey(1)

    label, confidence = classify_victim(warped)
    if confidence < 0.5:
        return None
    return label, confidence


while robot.step(timestep) != -1:
    image = camera.getImage()
    image = np.frombuffer(image, np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
    print(detect_victim(image))
