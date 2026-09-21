"""Versioned RGB-D/map observations. Coordinates follow ROS REP-103.

Images are rectified and depth is registered to the RGB optical frame. Transforms
are T_map_base and T_map_camera_optical at the image capture timestamp, in metres.
Array payloads are fixed-dtype raw bytes, never pickle or executable serialization.
"""

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

VERSION = 1
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid {name}")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4)):
        raise ValueError(f"{name} is not a rigid transform")
    return matrix


def pack_array(array, dtype):
    array = np.asarray(array, dtype=dtype)
    return {"shape": list(array.shape), "data": base64.b64encode(array.tobytes()).decode("ascii")}


def unpack_array(payload, dtype, limit):
    shape = payload["shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("expected a two-dimensional array")
    for size in shape:
        positive_int(size, "dimension")
    count = shape[0] * shape[1]
    if count > limit:
        raise ValueError("array exceeds size limit")
    raw = base64.b64decode(payload["data"], validate=True)
    if len(raw) != count * np.dtype(dtype).itemsize:
        raise ValueError("array byte length does not match dimensions")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


@dataclass
class Grid:
    cells: np.ndarray  # ROS OccupancyGrid: [row(y), column(x)], -1 unknown
    resolution: float
    origin: np.ndarray  # x, y, yaw in map

    def __post_init__(self):
        self.cells = np.asarray(self.cells)
        self.origin = np.asarray(self.origin, dtype=float)
        if (self.cells.ndim != 2 or min(self.cells.shape) < 2 or self.cells.size > 1_000_000
                or not np.isfinite(self.cells).all()
                or not np.equal(self.cells, np.floor(self.cells)).all()
                or np.any((self.cells < -1) | (self.cells > 100))):
            raise ValueError("invalid occupancy grid")
        if not np.isfinite(self.resolution) or not 0.02 <= self.resolution <= 0.5:
            raise ValueError("map resolution must be between 0.02 and 0.5 metres")
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("invalid map origin")

    @property
    def rotation(self):
        c, s = np.cos(self.origin[2]), np.sin(self.origin[2])
        return np.array([[c, -s], [s, c]])

    def world_to_cell(self, xy):
        local = (np.asarray(xy)[..., :2] - self.origin[:2]) @ self.rotation
        return (local / self.resolution - 0.5)[..., ::-1]

    def cell_to_world(self, rc):
        local = (np.asarray(rc)[..., ::-1] + 0.5) * self.resolution
        return local @ self.rotation.T + self.origin[:2]


@dataclass
class Observation:
    seq: int
    stamp_ns: int
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    map_from_camera: np.ndarray
    map_from_base: np.ndarray
    grid: Grid

    @classmethod
    def decode(cls, value):
        if value.get("version") != VERSION or value.get("frame_id") != "map":
            raise ValueError("expected protocol version 1, frame_id=map")
        seq = positive_int(value["seq"], "seq")
        stamp = positive_int(value["capture_stamp_ns"], "capture_stamp_ns")
        depth = unpack_array(value["depth_f32_m"], "<f4", 1280 * 720)
        with Image.open(io.BytesIO(base64.b64decode(value["rgb_jpeg"], validate=True))) as image:
            if image.size != (depth.shape[1], depth.shape[0]):
                raise ValueError("RGB and aligned depth sizes differ")
            rgb = np.asarray(image.convert("RGB")).copy()
        # Invalid depth is unobserved, never free space.
        depth[~np.isfinite(depth) | (depth < 0.15) | (depth > 8.)] = 0
        if np.count_nonzero(depth) < depth.size * 0.05:
            raise ValueError("insufficient valid metric depth")
        k = np.asarray(value["intrinsics"], dtype=float)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or not 0 <= k[0, 2] < depth.shape[1] or not 0 <= k[1, 2] < depth.shape[0]):
            raise ValueError("invalid calibrated camera intrinsics")
        grid = value["grid"]
        return cls(seq, stamp, rgb, depth, k,
                   rigid_matrix(value["map_from_camera"], "map_from_camera"),
                   rigid_matrix(value["map_from_base"], "map_from_base"),
                   Grid(unpack_array(grid["cells_i8"], "i1", 1_000_000),
                        float(grid["resolution"]), grid["origin"]))

    def encode(self):
        buffer = io.BytesIO()
        Image.fromarray(self.rgb).save(buffer, format="JPEG", quality=85)
        return {"version": VERSION, "frame_id": "map", "seq": self.seq,
                "capture_stamp_ns": self.stamp_ns,
                "rgb_jpeg": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "depth_f32_m": pack_array(self.depth, "<f4"),
                "intrinsics": self.intrinsics.tolist(),
                "map_from_camera": self.map_from_camera.tolist(),
                "map_from_base": self.map_from_base.tolist(),
                "grid": {"cells_i8": pack_array(self.grid.cells, "i1"),
                         "resolution": self.grid.resolution, "origin": self.grid.origin.tolist()}}


def body_waypoints(path_xy, map_from_base, spacing=0.06, max_distance=1.2):
    """Absolute poses in capture-time base_link, NOT incremental displacements."""
    path = np.asarray(path_xy, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or not len(path) or not np.isfinite(path).all():
        raise ValueError("invalid world path")
    base = rigid_matrix(map_from_base, "map_from_base")
    yaw = np.arctan2(base[1, 0], base[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    local = (path - base[:2, 3]) @ rotation
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distances) > 1e-8]
    local, distances = local[keep], distances[keep]
    if len(local) == 1:
        return [[float(local[0, 0]), float(local[0, 1]), 0.]]
    samples = np.unique(np.r_[np.arange(0, min(distances[-1], max_distance), spacing),
                              min(distances[-1], max_distance)])
    xy = np.column_stack([np.interp(samples, distances, local[:, axis]) for axis in range(2)])
    direction = np.diff(xy, axis=0)
    headings = np.arctan2(direction[:, 1], direction[:, 0])
    return np.column_stack([xy, np.r_[headings, headings[-1]]]).tolist()
