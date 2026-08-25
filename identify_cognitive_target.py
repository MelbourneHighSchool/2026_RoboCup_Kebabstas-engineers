import cv2
import numpy as np
from collections import Counter
from controller import Robot, Camera
from dataclasses import dataclass, field
import typing

robot = Robot()

camera = typing.cast(Camera, robot.getDevice("camera1"))

timestep = int(robot.getBasicTimeStep())

camera.enable(timestep)

WIDTH = camera.getWidth()
HEIGHT = camera.getHeight()


@dataclass
class ViewCorrection:
    lateral: float = 0.0
    forward: float = 0.0
    yaw: float = 0.0


type Colour = typing.Literal["red", "green", "blue", "yellow", "black", "combined"]


@dataclass
class CognitiveTargetDetection:
    found: bool = False  # valid cognitive target detected
    type: str | None = None  # F,P,C,O,None
    confidence: float = 0  # 0 - 1
    rings: list[Colour] = field(
        default_factory=list
    )  # detected colours from center to outside

    view_correction: ViewCorrection = field(default_factory=ViewCorrection)

    center: tuple[int, int] | None = None  # (x, y)
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
    agreements: list[float]
    geometry_score: float
    valid_classification: bool
    target_mask: np.ndarray | None = None


@dataclass
class EllipseCluster:
    members: list[EllipseCandidate]
    cx: float
    cy: float
    outer: EllipseCandidate


@dataclass
class RingSampleResult:
    colour: Colour | None
    # Dominant colour percentage.
    agreement: float
    # Number of valid colour samples.
    sample_count: int


def classify_hls(h: int, l: int, s: int) -> Colour | None:
    if l < 25:
        return "black"

    # All target colours are very saturated.
    # Teal wall is much less saturated.
    if s < 180:
        return None

    # Convert OpenCV hue to degrees.
    hue = h * 2

    # Red wraps around 0/360.
    if hue <= 15 or hue >= 345:
        return "red"

    if 45 <= hue <= 75:
        return "yellow"

    if 100 <= hue <= 140:
        return "green"

    if 210 <= hue <= 270:
        return "blue"

    return None


def cluster_ellipses(ellipses: list[EllipseCandidate]) -> list[EllipseCluster]:

    # Temporary list of clusters.
    #
    # Each cluster is just a list of
    # EllipseCandidate objects.
    clusters: list[list[EllipseCandidate]] = []

    max_distance = min(WIDTH, HEIGHT) * 0.08

    for ellipse in ellipses:
        for cluster in clusters:
            mean_cx = np.mean([e.cx for e in cluster])
            mean_cy = np.mean([e.cy for e in cluster])
            distance = np.hypot(
                ellipse.cx - mean_cx,
                ellipse.cy - mean_cy,
            )
            if distance <= max_distance:
                cluster.append(ellipse)
                break

        else:
            clusters.append([ellipse])

    result: list[EllipseCluster] = []

    for cluster in clusters:
        outer = max(cluster, key=lambda e: e.major * e.minor)

        result.append(
            EllipseCluster(
                members=cluster,
                cx=float(np.mean([e.cx for e in cluster])),
                cy=float(np.mean([e.cy for e in cluster])),
                outer=outer,
            )
        )

    return result


def ellipse_point(
    cx: float,
    cy: float,
    a: float,
    b: float,
    rotation_deg: float,
    frac: float,
    theta: float,
) -> tuple[int, int]:

    xr = frac * a * np.cos(theta)
    yr = frac * b * np.sin(theta)

    rot = np.deg2rad(rotation_deg)

    x = cx + xr * np.cos(rot) - yr * np.sin(rot)

    y = cy + xr * np.sin(rot) + yr * np.cos(rot)

    return int(round(x)), int(round(y))


def estimate_yaw_from_cluster(
    cluster: EllipseCluster,
) -> float:
    if len(cluster.members) < 3:
        return 0.0

    members = sorted(
        cluster.members,
        key=lambda e: e.major * e.minor,
    )

    inner = members[0]
    outer = members[-1]

    dx = outer.cx - inner.cx

    return max(-1.0, min(1.0, dx))


def sample_ring_region(
    hls_img: np.ndarray,
    cx: float,
    cy: float,
    a: float,
    b: float,
    rotation_deg: float,
    inner_frac: float,
    outer_frac: float,
) -> RingSampleResult:

    colours: list[Colour] = []

    avg_radius = (a + b) / 2.0

    angle_count = max(24, int(avg_radius))

    radius_count = 4

    radius_samples = np.linspace(inner_frac, outer_frac, radius_count)

    theta_samples = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)

    # Precompute trig values

    cos_values = np.cos(theta_samples)
    sin_values = np.sin(theta_samples)

    rot = np.deg2rad(rotation_deg)

    cos_rot = np.cos(rot)
    sin_rot = np.sin(rot)

    # Sample ring

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
        return RingSampleResult(
            colour=None,
            agreement=0.0,
            sample_count=len(colours),
        )

    dominant_colour, count = histogram.most_common(1)[0]

    agreement = count / len(colours)

    return RingSampleResult(
        colour=dominant_colour,
        agreement=agreement,
        sample_count=len(colours),
    )


COLOR_VALUES = {
    "black": -2,
    "red": -1,
    "yellow": 0,
    "green": 1,
    "blue": 2,
}

TARGET_TYPES = {
    0: "F",
    1: "P",
    2: "C",
    3: "O",
}


def populate_navigation_geometry(
    result: CognitiveTargetDetection,
    image: np.ndarray,
    cx: float,
    cy: float,
    major: float,
    minor: float,
) -> None:

    a = major / 2.0
    b = minor / 2.0

    result.center = (int(round(cx)), int(round(cy)))

    result.radius = min(a, b)

    image_height = image.shape[0]
    image_width = image.shape[1]

    # Cropping check
    outside_left = max(0.0, -(cx - a))
    outside_right = max(0.0, (cx + a) - image_width)
    outside_top = max(0.0, -(cy - b))
    outside_bottom = max(0.0, (cy + b) - image_height)
    max_outside = max(outside_left, outside_right, outside_top, outside_bottom)

    crop_fraction = max_outside / max(a, b)

    # View Correction
    correction = ViewCorrection()

    image_centre_x = image_width / 2.0

    correction.lateral = (cx - image_centre_x) / (image_width / 2.0)
    correction.lateral = max(-1.0, min(1.0, correction.lateral))

    diameter_fraction: float = max(major, minor) / min(image_width, image_height)

    MIN_DIAMETER_FRAC = 0.25
    MAX_DIAMETER_FRAC = 0.8

    if crop_fraction > 0.15:
        correction.forward = max(-1.0, -crop_fraction * 2.0)
    elif diameter_fraction < MIN_DIAMETER_FRAC:
        correction.forward = (MIN_DIAMETER_FRAC - diameter_fraction) / MIN_DIAMETER_FRAC
    elif diameter_fraction > MAX_DIAMETER_FRAC:
        correction.forward = (
            -(diameter_fraction - MAX_DIAMETER_FRAC) / MAX_DIAMETER_FRAC
        )
    else:
        correction.forward = 0.0

    result.view_correction = correction


def build_target_mask(bgr: np.ndarray, debug: bool = False) -> dict[Colour, np.ndarray]:

    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

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

    if debug:
        cv2.namedWindow("Target Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Target Mask", target_mask)
        cv2.waitKey(1)

    return {
        "red": red_mask,
        "blue": blue_mask,
        "yellow": yellow_mask,
        "green": green_mask,
        "black": black_mask,
        "combined": target_mask,
    }


def extract_ellipse_candidates(
    masks: dict[Colour, np.ndarray],
    debug: bool = False,
) -> list[EllipseCandidate]:
    candidates: list[EllipseCandidate] = []
    for source_colour, mask in masks.items():
        debug_contours: np.ndarray | None = None
        contours, _hierarchy = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < WIDTH * HEIGHT * 0.005:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter < min(WIDTH, HEIGHT) * 0.234:
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

            if debug:
                print(
                    f"ellipse: "
                    f"source colour {source_colour}"
                    f"cx={cx:.1f} "
                    f"cy={cy:.1f} "
                    f"major={major:.1f} "
                    f"minor={minor:.1f} "
                    f"circ={circularity:.2f}"
                )

            candidates.append(
                EllipseCandidate(
                    cx=cx,
                    cy=cy,
                    major=major,
                    minor=minor,
                    angle=angle,
                    area=area,
                    circularity=circularity,
                    contour_index=i,
                    source_colour=source_colour,
                )
            )
            if debug and debug_contours is not None and source_colour == "combined":
                cv2.ellipse(debug_contours, ellipse, (0, 255, 0), 1)
                cv2.circle(debug_contours, (int(cx), int(cy)), 2, (0, 0, 255), 1)
                cv2.namedWindow("CT Contours", cv2.WINDOW_NORMAL)
                cv2.imshow("CT Contours", debug_contours)
                cv2.waitKey(1)

    return candidates


def classify_candidate(
    rings: list[Colour],
) -> str | None:
    total = sum(COLOR_VALUES[colour] for colour in rings)
    return TARGET_TYPES.get(total)


def colour_evidence_score(
    cluster: EllipseCluster,
) -> float:
    colours = {
        member.source_colour
        for member in cluster.members
        if member.source_colour != "combined"
    }

    return min(len(colours) / 5.0, 1.0)


def score_candidate(
    cluster: EllipseCluster,
    rings: list[Colour],
    agreements: list[float],
    outer: EllipseCandidate,
) -> float:

    mean_agreement = sum(agreements) / len(agreements)
    ratio = min(outer.major, outer.minor) / max(outer.major, outer.minor)

    changes = sum(rings[i] != rings[i + 1] for i in range(len(rings) - 1))

    cluster_strength = min(len(cluster.members) / 4.0, 1.0)

    colour_evidence = colour_evidence_score(cluster)

    geometry_bonus = outer.circularity * ratio

    return (
        mean_agreement
        * geometry_bonus
        * (1.0 + 0.25 * changes)
        * (0.5 + 0.5 * cluster_strength)
        * (0.5 + colour_evidence)
    )


def evaluate_cluster(
    cluster: EllipseCluster,
    image: np.ndarray,
    hls: np.ndarray,
    bgr: np.ndarray,
    debug: bool = False,
) -> CandidateEvaluation:

    result = CognitiveTargetDetection()

    outer = cluster.outer

    cx = outer.cx
    cy = outer.cy

    major = outer.major
    minor = outer.minor

    a = major / 2.0
    b = minor / 2.0

    ring_bounds = [
        (0.00, 0.20),
        (0.20, 0.40),
        (0.40, 0.60),
        (0.60, 0.80),
        (0.80, 1.00),
    ]

    ring_results: list[RingSampleResult] = []

    rings: list[Colour] = []

    agreements: list[float] = []

    ellipse_area = np.pi * a * b
    occupancy = outer.area / ellipse_area

    if debug:
        print(f"Occupancy={occupancy:.3f}")

    if occupancy < 0.70:
        if debug:
            print("Rejected (low occupancy)")

        return CandidateEvaluation(
            result=result,
            score=0.0,
            geometry_score=0.0,
            valid_classification=False,
            agreements=[],
        )

    ratio = min(major, minor) / max(major, minor)
    populate_navigation_geometry(result, image, cx, cy, major, minor)
    candidate_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.ellipse(
        candidate_mask,
        ((int(cx), int(cy)), (int(major), int(minor)), outer.angle),
        255,
        -1,
    )

    image_height = image.shape[0]
    vertical_position = cy / image_height

    if debug:
        print(f"Vertical position={vertical_position:.2f}")

    vertical_penalty = 1.0
    if vertical_position > 0.80:
        vertical_penalty = 0.2
        if debug:
            print("Vertically low image, penalised")

    geometry_score = (
        outer.circularity
        * ratio
        * min(len(cluster.members) / 4.0, 1.0)
        * occupancy
        * vertical_penalty
    )

    if debug:
        print(f"Geometry Score: {geometry_score:.3f}")

    if geometry_score < 0.35:
        if debug:
            print("Rejected (weak geometry)")

        return CandidateEvaluation(
            result=result,
            score=0.0,
            geometry_score=geometry_score,
            valid_classification=False,
            agreements=[],
        )

    # Sample rings

    for inner_frac, outer_frac in ring_bounds:
        sample = sample_ring_region(
            hls_img=hls,
            cx=cx,
            cy=cy,
            a=a,
            b=b,
            rotation_deg=outer.angle,
            inner_frac=inner_frac,
            outer_frac=outer_frac,
        )

        ring_results.append(sample)

        if sample.colour is None:
            return CandidateEvaluation(
                result=result,
                score=0.0,
                valid_classification=False,
                agreements=[],
                geometry_score=geometry_score,
            )

        rings.append(sample.colour)
        agreements.append(sample.agreement)

    result.view_correction.yaw = estimate_yaw_from_cluster(cluster) * (1 - ratio)
    if debug:
        print(f"Yaw={result.view_correction.yaw:.3f}")
        print()
        print("RINGS")

        for i, sample in enumerate(ring_results):
            print(f"Ring {i}: " f"{sample.colour}")
            print(f"  agreement=" f"{sample.agreement:.3f}")

    target_type = classify_candidate(rings)

    if target_type is None:
        return CandidateEvaluation(
            result=result,
            score=0.0,
            valid_classification=False,
            agreements=agreements,
            geometry_score=geometry_score,
        )

    result.found = True
    result.type = target_type
    result.rings = rings

    mean_agreement = sum(agreements) / len(agreements)

    result.confidence = max(0.0, min(1.0, mean_agreement * geometry_score))

    if debug:
        colour_evidence = colour_evidence_score(cluster)
        colours = {member.source_colour for member in cluster.members}
        print("Cluster source colours:", colours)
        print("Colour evidence:", colour_evidence)

    score = score_candidate(cluster, rings, agreements, outer)

    return CandidateEvaluation(
        result=result,
        score=score,
        agreements=agreements,
        valid_classification=True,
        geometry_score=geometry_score,
    )


def detect_cognitive_target(
    image: np.ndarray, debug: bool = False
) -> CognitiveTargetDetection:
    result = CognitiveTargetDetection()

    bgr = image[:, :, :3]

    masks = build_target_mask(bgr, debug)

    candidates = extract_ellipse_candidates(masks)

    if len(candidates) == 0:
        return CognitiveTargetDetection()

    clusters = cluster_ellipses(candidates)

    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    best_valid: CandidateEvaluation | None = None
    best_geometry: CandidateEvaluation | None = None
    for cluster in clusters:
        candidate = evaluate_cluster(cluster, image, hls, bgr, debug)

        if (
            best_geometry is None
            or candidate.geometry_score > best_geometry.geometry_score
        ):
            best_geometry = candidate

        if candidate.valid_classification:
            if best_valid is None or candidate.score > best_valid.score:
                best_valid = candidate

    if best_valid is not None:
        result = best_valid.result

    elif best_geometry is not None:
        result = best_geometry.result
    else:
        return CognitiveTargetDetection()

    if debug:
        print()
        print("========== TARGET ==========")
        print("Type:", result.type)
        print("Rings:", result.rings)
        print("Confidence:", result.confidence)
        print("Radius:", result.radius)
        print("Centre:", result.center)
        print("== VIEW CORRECTION ==")
        print("lateral:", result.view_correction.lateral)
        print("forward:", result.view_correction.forward)
        print("yaw:", result.view_correction.yaw)
        print("============================")

    return result


while robot.step(timestep) != -1:
    image = camera.getImage()
    image = np.frombuffer(image, np.uint8).reshape((HEIGHT, WIDTH, 4))
    outcome = detect_cognitive_target(image)
    print(outcome)
