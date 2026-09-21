"""Small flat-floor RGB-D mapper for the navigation simulation harness.

Only measured depth and supplied poses are used; no simulator geometry queries.
Static obstacle evidence is conservative/sticky. This is not a replacement for
RTAB-Map, SLAM localization, dynamic costmaps, or terrain mapping on the Go2.
"""

import numpy as np

from .protocol import Grid


class DepthGridMapper:
    def __init__(self, bounds=(-2., 6., -3., 3.), resolution=.1, max_range=5.):
        xmin, xmax, ymin, ymax = bounds
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("invalid map bounds")
        self.grid = Grid(np.full((int(np.ceil((ymax-ymin)/resolution)),
                                  int(np.ceil((xmax-xmin)/resolution))), -1, np.int8),
                         resolution, [xmin, ymin, 0.])
        self.max_range = max_range

    def integrate(self, depth, k, map_from_camera, robot_xy):
        vv, uu = np.mgrid[0:depth.shape[0]:5, 0:depth.shape[1]:5]
        z = depth[::5, ::5].ravel()
        valid = np.isfinite(z) & (z > .15) & (z < self.max_range)
        rays = np.linalg.solve(k, np.vstack((uu.ravel()[valid], vv.ravel()[valid], np.ones(valid.sum()))))
        points = (map_from_camera[:3, :3] @ (rays * z[valid]) + map_from_camera[:3, 3:4]).T
        # Floor surfaces or obstacles in the robot's vertical collision envelope.
        points = points[(points[:, 2] > -.04) & (points[:, 2] < .9)]
        if not len(points):
            return
        camera_cell = self.grid.world_to_cell(map_from_camera[:2, 3])
        ends = self.grid.world_to_cell(points[:, :2])
        # Vectorized ray integration. Ends above the floor supply occupied hits.
        count = int(np.ceil(self.max_range / self.grid.resolution * 1.5))
        samples = camera_cell + (ends[:, None, :] - camera_cell) * np.linspace(0., 1., count)[None, :, None]
        cells = np.rint(samples).astype(int).reshape(-1, 2)
        inside = (cells >= 0).all(axis=1) & (cells < self.grid.cells.shape).all(axis=1)
        cells = cells[inside]
        current = self.grid.cells[cells[:, 0], cells[:, 1]]
        free = cells[current != 100]
        self.grid.cells[free[:, 0], free[:, 1]] = 0
        hits = np.rint(ends[points[:, 2] > .09]).astype(int)
        inside = (hits >= 0).all(axis=1) & (hits < self.grid.cells.shape).all(axis=1)
        hits = hits[inside]
        self.grid.cells[hits[:, 0], hits[:, 1]] = 100
        # Explicit flat-floor/occupied-footprint prior: the simulated robot is
        # known to stand here. No surrounding room geometry is prefilled.
        rows, cols = np.indices(self.grid.cells.shape)
        center = self.grid.world_to_cell(robot_xy)
        footprint = np.hypot(rows-center[0], cols-center[1]) * self.grid.resolution <= .45
        self.grid.cells[footprint & (self.grid.cells != 100)] = 0
