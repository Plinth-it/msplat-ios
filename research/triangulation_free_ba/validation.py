"""Mechanism-level validation for triangulation-free bundle adjustment.

This is a deliberately small, synthetic experiment based on arXiv:2608.21008v1.
It probes three claims that do not require the unreleased reference implementation
or the MobileBrick/ScanNet++ datasets:

1. Thresholded triangulation can commit sparse structure to a coarse pose prior.
2. Accurate-pose triangulation restores full retained observation coverage and
   permits shared-point bundle adjustment to recover the synthetic rig.
3. Re-expressing structure with one positive depth per observation permits a
   high-to-low arctan-loss schedule to retain a useful descent direction farther
   from the correct poses than the final robust loss alone.

The optimizer is intentionally test-only: dense finite-difference Levenberg-
Marquardt over a tiny rig. It is not a Ceres replacement and its success counts
must not be compared directly with the paper's real-data tables.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Iterable, Sequence

import numpy as np


NOMINAL_LOSS_SCALE = 1.0e2
GNC_LOSS_SCALES = (1.0e4, 1.0e3, 1.0e2)
DECOY_POSE_SEED = 10_000


@dataclasses.dataclass(frozen=True)
class SyntheticScene:
    focal: float
    principal: np.ndarray
    pixels: np.ndarray
    matches: np.ndarray
    true_rotations: np.ndarray
    true_centers: np.ndarray
    prior_rotations: np.ndarray
    prior_centers: np.ndarray
    points: np.ndarray
    perturbation_degrees: float

    @property
    def camera_count(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.pixels.shape[1])


@dataclasses.dataclass(frozen=True)
class SolverResult:
    rotations: np.ndarray
    centers: np.ndarray
    depths: np.ndarray | None
    cost: float
    iterations: int


@dataclasses.dataclass(frozen=True)
class TriangulationResult:
    points: np.ndarray
    observations: tuple[np.ndarray, ...]
    source_track_count: int
    camera_count: int

    @property
    def track_count(self) -> int:
        return len(self.observations)

    @property
    def observation_count(self) -> int:
        return sum(track.shape[0] for track in self.observations)

    @property
    def observed_camera_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    int(camera_index)
                    for track in self.observations
                    for camera_index in track[:, 0]
                }
            )
        )

    @property
    def observed_camera_count(self) -> int:
        return len(self.observed_camera_indices)

    @property
    def camera_component_count(self) -> int:
        observed = set(self.observed_camera_indices)
        adjacency = {camera_index: set() for camera_index in observed}
        for track in self.observations:
            cameras = [int(value) for value in track[:, 0]]
            for camera_index in cameras:
                adjacency[camera_index].update(cameras)
        components = 0
        remaining = observed.copy()
        while remaining:
            components += 1
            frontier = [remaining.pop()]
            while frontier:
                camera_index = frontier.pop()
                neighbors = adjacency[camera_index] & remaining
                remaining.difference_update(neighbors)
                frontier.extend(neighbors)
        return components

    @property
    def all_cameras_connected(self) -> bool:
        return (
            self.observed_camera_indices == tuple(range(self.camera_count))
            and self.camera_component_count == 1
        )


@dataclasses.dataclass(frozen=True)
class SharedPointBAResult:
    rotations: np.ndarray
    centers: np.ndarray
    points: np.ndarray
    cost: float
    iterations: int
    observation_count: int
    minimum_depth: float

    @property
    def runnable(self) -> bool:
        return self.observation_count > 0


@dataclasses.dataclass(frozen=True)
class SweepRow:
    perturbation_degrees: float
    seed: int
    decoy_pose_seed: int
    prior_rotation_error: float
    plain_rotation_error: float
    gnc_rotation_error: float
    prior_center_error_mm: float
    plain_center_error_mm: float
    gnc_center_error_mm: float
    prior_absolute_center_error_mm: float
    plain_absolute_center_error_mm: float
    gnc_absolute_center_error_mm: float

    @property
    def plain_success(self) -> bool:
        return self.plain_rotation_error < 1.0

    @property
    def gnc_success(self) -> bool:
        return self.gnc_rotation_error < 1.0


@dataclasses.dataclass(frozen=True)
class MechanismDiagnostic:
    alphas: np.ndarray
    prior_structure_costs: np.ndarray
    oracle_structure_costs: np.ndarray
    ray_depth_costs: np.ndarray
    prior_structure_minimum: float
    oracle_structure_minimum: float
    ray_depth_minimum: float
    nominal_normalized_forward_drop: float
    smooth_normalized_forward_drop: float
    prior_retained_observation_fraction: float
    prior_observed_camera_count: int
    prior_camera_count: int
    prior_camera_component_count: int


@dataclasses.dataclass(frozen=True)
class StructureControlRow:
    perturbation_degrees: float
    seed: int
    gate_pixels: float
    camera_count: int
    prior_track_count: int
    prior_observation_count: int
    prior_observed_camera_count: int
    prior_camera_component_count: int
    prior_all_cameras_connected: bool
    prior_rotation_error: float
    prior_center_error_mm: float
    prior_absolute_center_error_mm: float
    prior_minimum_depth: float
    accurate_track_count: int
    accurate_observation_count: int
    accurate_observed_camera_count: int
    accurate_camera_component_count: int
    accurate_all_cameras_connected: bool
    accurate_rotation_error: float
    accurate_center_error_mm: float
    accurate_absolute_center_error_mm: float
    accurate_minimum_depth: float

    @property
    def prior_graph_valid(self) -> bool:
        return self.prior_all_cameras_connected

    @property
    def accurate_graph_valid(self) -> bool:
        return self.accurate_all_cameras_connected

    @property
    def prior_solution_valid(self) -> bool:
        return (
            self.prior_graph_valid
            and np.isfinite(self.prior_rotation_error)
            and np.isfinite(self.prior_center_error_mm)
            and np.isfinite(self.prior_absolute_center_error_mm)
            and np.isfinite(self.prior_minimum_depth)
            and self.prior_minimum_depth > 0.0
        )

    @property
    def accurate_solution_valid(self) -> bool:
        return (
            self.accurate_graph_valid
            and np.isfinite(self.accurate_rotation_error)
            and np.isfinite(self.accurate_center_error_mm)
            and np.isfinite(self.accurate_absolute_center_error_mm)
            and np.isfinite(self.accurate_minimum_depth)
            and self.accurate_minimum_depth > 0.0
        )

    @property
    def prior_success(self) -> bool:
        return self.prior_solution_valid and self.prior_rotation_error < 1.0

    @property
    def accurate_success(self) -> bool:
        return self.accurate_solution_valid and self.accurate_rotation_error < 1.0


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    theta_squared = float(rotation_vector @ rotation_vector)
    omega = _skew(rotation_vector)
    if theta_squared < 1.0e-12:
        a = 1.0 - theta_squared / 6.0
        b = 0.5 - theta_squared / 24.0
    else:
        theta = math.sqrt(theta_squared)
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta_squared
    return np.eye(3) + a * omega + b * (omega @ omega)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1.0e-8:
        return 0.5 * vee
    sine = math.sin(angle)
    if abs(sine) < 1.0e-8:
        # The synthetic sweep stays far from pi, but keep this deterministic.
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(
            eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))]
        )
        return angle * axis / np.linalg.norm(axis)
    return (0.5 * angle / sine) * vee


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _look_at_rotation(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = _normalize(target - center)
    right = _normalize(np.cross(np.array([0.0, 1.0, 0.0]), forward))
    camera_up = np.cross(forward, right)
    return np.column_stack((right, camera_up, forward))


def _project(
    rotation: np.ndarray,
    center: np.ndarray,
    points: np.ndarray,
    focal: float,
    principal: np.ndarray,
) -> np.ndarray:
    camera_points = (points - center) @ rotation
    depth = camera_points[:, 2]
    safe_depth = np.where(
        np.abs(depth) < 1.0e-8,
        np.where(depth < 0.0, -1.0e-8, 1.0e-8),
        depth,
    )
    return principal + focal * camera_points[:, :2] / safe_depth[:, None]


def _world_rays(
    rotation: np.ndarray,
    pixels: np.ndarray,
    focal: float,
    principal: np.ndarray,
) -> np.ndarray:
    camera_rays = np.column_stack(
        ((pixels - principal) / focal, np.ones(pixels.shape[0]))
    )
    camera_rays /= np.linalg.norm(camera_rays, axis=1, keepdims=True)
    return camera_rays @ rotation.T


def _two_ray_midpoint(
    first_origin: np.ndarray,
    first_direction: np.ndarray,
    second_origin: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray:
    system = np.column_stack((first_direction, -second_direction))
    times, *_ = np.linalg.lstsq(
        system, second_origin - first_origin, rcond=None
    )
    first_point = first_origin + times[0] * first_direction
    second_point = second_origin + times[1] * second_direction
    return 0.5 * (first_point + second_point)


def make_scene(
    perturbation_degrees: float,
    seed: int,
    *,
    camera_count: int = 6,
    point_count: int = 12,
    pixel_noise: float = 0.15,
    decoy_fraction: float = 0.12,
    decoy_pose_seed: int = DECOY_POSE_SEED,
) -> SyntheticScene:
    scene_rng = np.random.default_rng(0)
    pose_rng = np.random.default_rng(seed)
    decoy_rng = np.random.default_rng(decoy_pose_seed)
    focal = 520.0
    principal = np.array([320.0, 240.0])
    camera_angles = np.linspace(-0.72, 0.72, camera_count)
    true_centers = np.array(
        [
            [2.0 * math.sin(angle), 0.12 * math.cos(2.0 * angle),
             2.0 * math.cos(angle)]
            for angle in camera_angles
        ]
    )
    true_rotations = np.array(
        [_look_at_rotation(center, np.zeros(3)) for center in true_centers]
    )
    points = scene_rng.uniform(
        low=np.array([-0.48, -0.34, -0.38]),
        high=np.array([0.48, 0.34, 0.38]),
        size=(point_count, 3),
    )
    pixels = np.array(
        [
            _project(rotation, center, points, focal, principal)
            for rotation, center in zip(true_rotations, true_centers, strict=True)
        ]
    )
    pixels += scene_rng.normal(scale=pixel_noise, size=pixels.shape)

    prior_rotations = true_rotations.copy()
    prior_centers = true_centers.copy()
    angle_radians = math.radians(perturbation_degrees)
    translation_magnitude = 0.005 * perturbation_degrees
    for camera_index in range(camera_count):
        axis = _normalize(pose_rng.normal(size=3))
        direction = _normalize(pose_rng.normal(size=3))
        prior_rotations[camera_index] = (
            so3_exp(axis * angle_radians) @ true_rotations[camera_index]
        )
        prior_centers[camera_index] += direction * translation_magnitude

    # A fixed, coherent false-pose lobe defines the aliasing matches below.
    # It is generated once at 16 degrees/80 mm and does not change with the
    # perturbation level or pose-perturbation seed being evaluated.
    decoy_rotations = true_rotations.copy()
    decoy_centers = true_centers.copy()
    for camera_index in range(camera_count):
        decoy_axis = _normalize(decoy_rng.normal(size=3))
        decoy_direction = _normalize(decoy_rng.normal(size=3))
        decoy_rotations[camera_index] = (
            so3_exp(decoy_axis * math.radians(16.0))
            @ true_rotations[camera_index]
        )
        decoy_centers[camera_index] += 0.08 * decoy_direction

    camera_pairs = [
        (first, second)
        for gap in (1, 2)
        for first in range(camera_count - gap)
        for second in (first + gap,)
    ]
    correct_matches = [
        (first, point_index, second, point_index)
        for first, second in camera_pairs
        for point_index in range(point_count)
    ]

    # Add a small set of wrong matches that support the fixed false-pose lobe.
    # This deliberately constructed repeated-texture stressor is identical at
    # every perturbation level for a seed; it is not a calibrated SIFT model.
    # Correct matches remain the clear majority.
    decoy_rays = np.array(
        [
            _world_rays(rotation, camera_pixels, focal, principal)
            for rotation, camera_pixels in zip(
                decoy_rotations, pixels, strict=True
            )
        ]
    )
    decoy_candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for first, second in camera_pairs:
        for first_point in range(point_count):
            best: tuple[float, tuple[int, int, int, int]] | None = None
            for second_point in range(point_count):
                if second_point == first_point:
                    continue
                point = _two_ray_midpoint(
                    decoy_centers[first],
                    decoy_rays[first, first_point],
                    decoy_centers[second],
                    decoy_rays[second, second_point],
                )
                first_error = (
                    _project(
                        decoy_rotations[first], decoy_centers[first],
                        point[None, :], focal, principal
                    )[0]
                    - pixels[first, first_point]
                )
                second_error = (
                    _project(
                        decoy_rotations[second], decoy_centers[second],
                        point[None, :], focal, principal
                    )[0]
                    - pixels[second, second_point]
                )
                score = float(first_error @ first_error + second_error @ second_error)
                candidate = (score, (first, first_point, second, second_point))
                if best is None or candidate[0] < best[0]:
                    best = candidate
            assert best is not None
            decoy_candidates.append(best)
    decoy_candidates.sort(key=lambda item: item[0])
    decoy_count = int(round(decoy_fraction * len(correct_matches)))
    matches = np.asarray(
        correct_matches
        + [candidate for _, candidate in decoy_candidates[:decoy_count]],
        dtype=np.int64,
    )
    return SyntheticScene(
        focal=focal,
        principal=principal,
        pixels=pixels,
        matches=matches,
        true_rotations=true_rotations,
        true_centers=true_centers,
        prior_rotations=prior_rotations,
        prior_centers=prior_centers,
        points=points,
        perturbation_degrees=perturbation_degrees,
    )


def _known_observation_tracks(scene: SyntheticScene) -> tuple[np.ndarray, ...]:
    """Return synthetic track identities, intentionally excluding decoy edges."""
    return tuple(
        np.asarray(
            [
                (camera_index, point_index)
                for camera_index in range(scene.camera_count)
            ],
            dtype=np.int64,
        )
        for point_index in range(scene.point_count)
    )


def _linear_triangulate_observations(
    observations: np.ndarray,
    rays: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    normal = np.zeros((3, 3), dtype=np.float64)
    right_hand_side = np.zeros(3, dtype=np.float64)
    identity = np.eye(3)
    for camera_index, point_index in observations:
        ray = rays[camera_index, point_index]
        projector = identity - np.outer(ray, ray)
        normal += projector
        right_hand_side += projector @ centers[camera_index]
    return np.linalg.solve(normal, right_hand_side)


def _triangulation_inliers(
    scene: SyntheticScene,
    observations: np.ndarray,
    point: np.ndarray,
    rotations: np.ndarray,
    centers: np.ndarray,
    gate_pixels: float,
) -> tuple[np.ndarray, np.ndarray]:
    retained = []
    errors = []
    for observation_index, (camera_index, point_index) in enumerate(observations):
        camera_point = (point - centers[camera_index]) @ rotations[camera_index]
        if camera_point[2] <= 1.0e-6:
            continue
        predicted = _project(
            rotations[camera_index],
            centers[camera_index],
            point[None, :],
            scene.focal,
            scene.principal,
        )[0]
        error = float(
            np.linalg.norm(predicted - scene.pixels[camera_index, point_index])
        )
        if error <= gate_pixels:
            retained.append(observation_index)
            errors.append(error)
    return np.asarray(retained, dtype=np.int64), np.asarray(errors)


def triangulate_known_tracks(
    scene: SyntheticScene,
    rotations: np.ndarray,
    centers: np.ndarray,
    *,
    gate_pixels: float,
) -> TriangulationResult:
    """Triangulate synthetic known tracks using deterministic pair hypotheses."""
    if gate_pixels <= 0.0:
        raise ValueError("gate_pixels must be positive")
    source_tracks = _known_observation_tracks(scene)
    rays = np.array(
        [
            _world_rays(rotation, pixels, scene.focal, scene.principal)
            for rotation, pixels in zip(rotations, scene.pixels, strict=True)
        ]
    )
    retained_points = []
    retained_tracks = []
    for observations in source_tracks:
        best_key: tuple[int, float, float, int, int] | None = None
        best_point: np.ndarray | None = None
        best_inliers: np.ndarray | None = None
        for first in range(observations.shape[0] - 1):
            first_camera, first_point = observations[first]
            for second in range(first + 1, observations.shape[0]):
                second_camera, second_point = observations[second]
                point = _two_ray_midpoint(
                    centers[first_camera],
                    rays[first_camera, first_point],
                    centers[second_camera],
                    rays[second_camera, second_point],
                )
                inliers, errors = _triangulation_inliers(
                    scene,
                    observations,
                    point,
                    rotations,
                    centers,
                    gate_pixels,
                )
                if inliers.size < 2:
                    continue
                parallax = math.acos(
                    float(
                        np.clip(
                            rays[first_camera, first_point]
                            @ rays[second_camera, second_point],
                            -1.0,
                            1.0,
                        )
                    )
                )
                key = (
                    -int(inliers.size),
                    float(np.sum(errors)),
                    -parallax,
                    first,
                    second,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_point = point
                    best_inliers = inliers
        if best_point is None or best_inliers is None:
            continue
        selected = observations[best_inliers]
        try:
            point = _linear_triangulate_observations(selected, rays, centers)
        except np.linalg.LinAlgError:
            point = best_point
        final_inliers, _ = _triangulation_inliers(
            scene,
            observations,
            point,
            rotations,
            centers,
            gate_pixels,
        )
        if final_inliers.size < 2:
            point = best_point
            final_inliers = best_inliers
        retained_points.append(point)
        retained_tracks.append(observations[final_inliers])
    points = (
        np.asarray(retained_points)
        if retained_points
        else np.empty((0, 3), dtype=np.float64)
    )
    return TriangulationResult(
        points=points,
        observations=tuple(retained_tracks),
        source_track_count=len(source_tracks),
        camera_count=scene.camera_count,
    )


def replace_known_track_points(
    point_source: TriangulationResult,
    observation_template: TriangulationResult,
) -> TriangulationResult:
    """Use one known-track structure with another structure's retained mask."""
    points_by_track = {
        int(observations[0, 1]): point
        for point, observations in zip(
            point_source.points, point_source.observations, strict=True
        )
    }
    points = []
    for observations in observation_template.observations:
        point_indices = set(int(value) for value in observations[:, 1])
        if len(point_indices) != 1:
            raise ValueError("observation template is not a known-track result")
        point_index = point_indices.pop()
        if point_index not in points_by_track:
            raise ValueError("point source is missing an observation-template track")
        points.append(points_by_track[point_index])
    return TriangulationResult(
        points=np.asarray(points, dtype=np.float64).reshape(-1, 3),
        observations=observation_template.observations,
        source_track_count=observation_template.source_track_count,
        camera_count=observation_template.camera_count,
    )


def _poses_from_parameters(
    scene: SyntheticScene, pose_parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotations = scene.prior_rotations.copy()
    centers = scene.prior_centers.copy()
    reshaped = pose_parameters.reshape(scene.camera_count, 6)
    for camera_index, update in enumerate(reshaped):
        rotations[camera_index] = (
            so3_exp(update[:3]) @ scene.prior_rotations[camera_index]
        )
        centers[camera_index] = scene.prior_centers[camera_index] + update[3:]
    return rotations, centers


def _cross_projection_residuals(
    scene: SyntheticScene,
    rotations: np.ndarray,
    centers: np.ndarray,
    log_depths: np.ndarray,
) -> np.ndarray:
    depths = np.exp(np.clip(log_depths, -20.0, 20.0)).reshape(
        scene.camera_count, scene.point_count
    )
    rays = np.array(
        [
            _world_rays(rotation, camera_pixels, scene.focal, scene.principal)
            for rotation, camera_pixels in zip(
                rotations, scene.pixels, strict=True
            )
        ]
    )
    residuals = np.empty((scene.matches.shape[0], 2, 2), dtype=np.float64)
    for match_index, (first, first_point, second, second_point) in enumerate(
        scene.matches
    ):
        point_from_first = (
            centers[first]
            + depths[first, first_point] * rays[first, first_point]
        )
        point_from_second = (
            centers[second]
            + depths[second, second_point] * rays[second, second_point]
        )
        residuals[match_index, 0] = (
            _project(
                rotations[second],
                centers[second],
                point_from_first[None, :],
                scene.focal,
                scene.principal,
            )[0]
            - scene.pixels[second, second_point]
        )
        residuals[match_index, 1] = (
            _project(
                rotations[first],
                centers[first],
                point_from_second[None, :],
                scene.focal,
                scene.principal,
            )[0]
            - scene.pixels[first, first_point]
        )
    return residuals


def arctan_loss(squared_norms: np.ndarray, scale: float) -> np.ndarray:
    return scale * np.arctan(squared_norms / scale)


def _objective(
    scene: SyntheticScene,
    parameters: np.ndarray,
    loss_scale: float,
    *,
    position_prior_weight: float = 1.0e-2,
) -> float:
    pose_count = 6 * scene.camera_count
    rotations, centers = _poses_from_parameters(scene, parameters[:pose_count])
    residuals = _cross_projection_residuals(
        scene, rotations, centers, parameters[pose_count:]
    )
    squared = np.sum(residuals * residuals, axis=-1)
    match_cost = float(np.sum(arctan_loss(squared, loss_scale)))
    match_cost /= 4.0 * scene.matches.shape[0]
    position_cost = position_prior_weight * float(
        np.sum((centers - scene.prior_centers) ** 2)
    ) / scene.camera_count
    return match_cost + position_cost


def _raw_residuals(
    scene: SyntheticScene,
    parameters: np.ndarray,
    *,
    position_prior_weight: float = 1.0e-2,
) -> tuple[np.ndarray, int]:
    pose_count = 6 * scene.camera_count
    rotations, centers = _poses_from_parameters(scene, parameters[:pose_count])
    matches = _cross_projection_residuals(
        scene, rotations, centers, parameters[pose_count:]
    ).reshape(-1)
    prior_scale = math.sqrt(position_prior_weight / scene.camera_count)
    position_residuals = (
        prior_scale * (centers - scene.prior_centers)
    ).reshape(-1)
    return np.concatenate((matches, position_residuals)), matches.size


def _finite_difference_jacobian(
    function,
    parameters: np.ndarray,
    baseline: np.ndarray,
    pose_parameter_count: int,
) -> np.ndarray:
    jacobian = np.empty((baseline.size, parameters.size), dtype=np.float64)
    for parameter_index in range(parameters.size):
        step = 2.0e-6 if parameter_index < pose_parameter_count else 2.0e-5
        displaced = parameters.copy()
        displaced[parameter_index] += step
        jacobian[:, parameter_index] = (
            function(displaced)[0] - baseline
        ) / step
    return jacobian


def _limit_step(
    step: np.ndarray,
    camera_count: int,
) -> np.ndarray:
    limited = step.copy()
    pose_count = 6 * camera_count
    pose_steps = limited[:pose_count].reshape(camera_count, 6)
    for pose_step in pose_steps:
        rotation_norm = float(np.linalg.norm(pose_step[:3]))
        if rotation_norm > 0.15:
            pose_step[:3] *= 0.15 / rotation_norm
        translation_norm = float(np.linalg.norm(pose_step[3:]))
        if translation_norm > 0.08:
            pose_step[3:] *= 0.08 / translation_norm
    np.clip(limited[pose_count:], -0.45, 0.45, out=limited[pose_count:])
    return limited


def solve_ray_depth(
    scene: SyntheticScene,
    loss_scales: Sequence[float],
    *,
    maximum_iterations_per_stage: int = 18,
    initial_depth_meters: float = 1.0,
) -> SolverResult:
    if not math.isfinite(initial_depth_meters) or initial_depth_meters <= 0.0:
        raise ValueError("initial_depth_meters must be finite and positive")
    pose_count = 6 * scene.camera_count
    depth_count = scene.camera_count * scene.point_count
    parameters = np.zeros(pose_count + depth_count, dtype=np.float64)
    parameters[pose_count:] = math.log(initial_depth_meters)
    damping = 1.0e-3
    total_iterations = 0

    for loss_scale in loss_scales:
        for _ in range(maximum_iterations_per_stage):
            total_iterations += 1
            raw, match_scalar_count = _raw_residuals(scene, parameters)
            match_vectors = raw[:match_scalar_count].reshape(-1, 2)
            squared = np.sum(match_vectors * match_vectors, axis=1)
            robust_weights = 1.0 / (1.0 + (squared / loss_scale) ** 2)
            row_weights = np.concatenate(
                (
                    np.repeat(
                        np.sqrt(robust_weights / (4.0 * scene.matches.shape[0])),
                        2,
                    ),
                    np.ones(raw.size - match_scalar_count),
                )
            )
            residual_function = lambda value: _raw_residuals(scene, value)
            jacobian = _finite_difference_jacobian(
                residual_function, parameters, raw, pose_count
            )
            weighted_jacobian = row_weights[:, None] * jacobian
            weighted_residuals = row_weights * raw
            normal = weighted_jacobian.T @ weighted_jacobian
            gradient = weighted_jacobian.T @ weighted_residuals
            diagonal = np.maximum(np.diag(normal), 1.0e-8)
            try:
                step = np.linalg.solve(
                    normal + damping * np.diag(diagonal), -gradient
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            step = _limit_step(step, scene.camera_count)
            current_cost = _objective(scene, parameters, loss_scale)
            candidate = parameters + step
            candidate_cost = _objective(scene, candidate, loss_scale)
            if np.isfinite(candidate_cost) and candidate_cost < current_cost:
                parameters = candidate
                damping = max(damping / 3.0, 1.0e-9)
                relative_drop = (current_cost - candidate_cost) / max(
                    1.0, abs(current_cost)
                )
                if relative_drop < 1.0e-9 or np.linalg.norm(step) < 1.0e-8:
                    break
            else:
                damping = min(damping * 10.0, 1.0e12)
                if damping >= 1.0e11:
                    break

    rotations, centers = _poses_from_parameters(scene, parameters[:pose_count])
    depths = np.exp(np.clip(parameters[pose_count:], -20.0, 20.0)).reshape(
        scene.camera_count, scene.point_count
    )
    return SolverResult(
        rotations=rotations,
        centers=centers,
        depths=depths,
        cost=_objective(scene, parameters, loss_scales[-1]),
        iterations=total_iterations,
    )


def _shared_point_state(
    scene: SyntheticScene,
    triangulation: TriangulationResult,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_count = 6 * scene.camera_count
    rotations, centers = _poses_from_parameters(scene, parameters[:pose_count])
    points = triangulation.points + parameters[pose_count:].reshape(-1, 3)
    return rotations, centers, points


def _shared_point_raw_residuals(
    scene: SyntheticScene,
    triangulation: TriangulationResult,
    parameters: np.ndarray,
    *,
    position_prior_weight: float,
) -> tuple[np.ndarray, int]:
    rotations, centers, points = _shared_point_state(
        scene, triangulation, parameters
    )
    reprojection_residuals = []
    for point, observations in zip(
        points, triangulation.observations, strict=True
    ):
        for camera_index, point_index in observations:
            residual = (
                _project(
                    rotations[camera_index],
                    centers[camera_index],
                    point[None, :],
                    scene.focal,
                    scene.principal,
                )[0]
                - scene.pixels[camera_index, point_index]
            )
            reprojection_residuals.extend(residual)
    reprojection = np.asarray(reprojection_residuals, dtype=np.float64)
    prior_scale = math.sqrt(position_prior_weight / scene.camera_count)
    position = (prior_scale * (centers - scene.prior_centers)).reshape(-1)
    return np.concatenate((reprojection, position)), reprojection.size


def _shared_point_objective(
    scene: SyntheticScene,
    triangulation: TriangulationResult,
    parameters: np.ndarray,
    loss_scale: float,
    *,
    position_prior_weight: float,
) -> float:
    raw, reprojection_scalar_count = _shared_point_raw_residuals(
        scene,
        triangulation,
        parameters,
        position_prior_weight=position_prior_weight,
    )
    reprojection = raw[:reprojection_scalar_count].reshape(-1, 2)
    squared = np.sum(reprojection * reprojection, axis=1)
    cost = float(np.sum(arctan_loss(squared, loss_scale)))
    cost /= 2.0 * triangulation.observation_count
    return cost + float(np.sum(raw[reprojection_scalar_count:] ** 2))


def _limit_shared_point_step(
    step: np.ndarray,
    camera_count: int,
) -> np.ndarray:
    limited = step.copy()
    pose_count = 6 * camera_count
    pose_steps = limited[:pose_count].reshape(camera_count, 6)
    for pose_step in pose_steps:
        rotation_norm = float(np.linalg.norm(pose_step[:3]))
        if rotation_norm > 0.15:
            pose_step[:3] *= 0.15 / rotation_norm
        translation_norm = float(np.linalg.norm(pose_step[3:]))
        if translation_norm > 0.08:
            pose_step[3:] *= 0.08 / translation_norm
    point_steps = limited[pose_count:].reshape(-1, 3)
    for point_step in point_steps:
        point_norm = float(np.linalg.norm(point_step))
        if point_norm > 0.15:
            point_step *= 0.15 / point_norm
    return limited


def _minimum_observed_depth(
    rotations: np.ndarray,
    centers: np.ndarray,
    points: np.ndarray,
    observations: tuple[np.ndarray, ...],
) -> float:
    depths = []
    for point, track in zip(points, observations, strict=True):
        for camera_index, _ in track:
            camera_point = (point - centers[camera_index]) @ rotations[camera_index]
            depths.append(float(camera_point[2]))
    return min(depths) if depths else math.nan


def solve_shared_point_bundle_adjustment(
    scene: SyntheticScene,
    triangulation: TriangulationResult,
    *,
    loss_scale: float = NOMINAL_LOSS_SCALE,
    position_prior_weight: float = 1.0e-2,
    maximum_iterations: int = 24,
) -> SharedPointBAResult:
    """Run a matched-loss shared-point BA control on triangulated tracks."""
    if triangulation.observation_count == 0:
        return SharedPointBAResult(
            rotations=scene.prior_rotations.copy(),
            centers=scene.prior_centers.copy(),
            points=triangulation.points.copy(),
            cost=math.inf,
            iterations=0,
            observation_count=0,
            minimum_depth=math.nan,
        )
    pose_count = 6 * scene.camera_count
    parameters = np.zeros(
        pose_count + 3 * triangulation.track_count, dtype=np.float64
    )
    damping = 1.0e-3
    total_iterations = 0
    normalization = 2.0 * triangulation.observation_count
    residual_function = lambda value: _shared_point_raw_residuals(
        scene,
        triangulation,
        value,
        position_prior_weight=position_prior_weight,
    )
    objective_function = lambda value: _shared_point_objective(
        scene,
        triangulation,
        value,
        loss_scale,
        position_prior_weight=position_prior_weight,
    )
    for _ in range(maximum_iterations):
        total_iterations += 1
        raw, reprojection_scalar_count = residual_function(parameters)
        reprojection = raw[:reprojection_scalar_count].reshape(-1, 2)
        squared = np.sum(reprojection * reprojection, axis=1)
        robust_weights = 1.0 / (1.0 + (squared / loss_scale) ** 2)
        row_weights = np.concatenate(
            (
                np.repeat(np.sqrt(robust_weights / normalization), 2),
                np.ones(raw.size - reprojection_scalar_count),
            )
        )
        jacobian = _finite_difference_jacobian(
            residual_function, parameters, raw, pose_count
        )
        weighted_jacobian = row_weights[:, None] * jacobian
        weighted_residuals = row_weights * raw
        normal = weighted_jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ weighted_residuals
        diagonal = np.maximum(np.diag(normal), 1.0e-8)
        try:
            step = np.linalg.solve(
                normal + damping * np.diag(diagonal), -gradient
            )
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue
        step = _limit_shared_point_step(step, scene.camera_count)
        current_cost = objective_function(parameters)
        candidate = parameters + step
        candidate_cost = objective_function(candidate)
        if np.isfinite(candidate_cost) and candidate_cost < current_cost:
            parameters = candidate
            damping = max(damping / 3.0, 1.0e-9)
            relative_drop = (current_cost - candidate_cost) / max(
                1.0, abs(current_cost)
            )
            if relative_drop < 1.0e-9 or np.linalg.norm(step) < 1.0e-8:
                break
        else:
            damping = min(damping * 10.0, 1.0e12)
            if damping >= 1.0e11:
                break
    rotations, centers, points = _shared_point_state(
        scene, triangulation, parameters
    )
    return SharedPointBAResult(
        rotations=rotations,
        centers=centers,
        points=points,
        cost=objective_function(parameters),
        iterations=total_iterations,
        observation_count=triangulation.observation_count,
        minimum_depth=_minimum_observed_depth(
            rotations, centers, points, triangulation.observations
        ),
    )


def mean_rotation_error_degrees(
    rotations: np.ndarray, true_rotations: np.ndarray
) -> float:
    errors = []
    for estimated, truth in zip(rotations, true_rotations, strict=True):
        relative = estimated @ truth.T
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        errors.append(math.degrees(math.acos(cosine)))
    return float(np.mean(errors))


def mean_center_error_mm(centers: np.ndarray, true_centers: np.ndarray) -> float:
    return 1000.0 * float(np.mean(np.linalg.norm(centers - true_centers, axis=1)))


def _similarity_aligned_poses(
    rotations: np.ndarray,
    centers: np.ndarray,
    true_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centered = centers - np.mean(centers, axis=0)
    true_centered = true_centers - np.mean(true_centers, axis=0)
    covariance = true_centered.T @ centered / centers.shape[0]
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    reflection = np.ones(3)
    reflection[-1] = np.sign(np.linalg.det(left @ right_transpose))
    world_rotation = left @ np.diag(reflection) @ right_transpose
    variance = float(np.mean(np.sum(centered * centered, axis=1)))
    scale = float(singular_values @ reflection) / variance
    aligned_centers = (
        scale * (world_rotation @ centered.T).T
        + np.mean(true_centers, axis=0)
    )
    aligned_rotations = np.array(
        [world_rotation @ rotation for rotation in rotations]
    )
    return aligned_rotations, aligned_centers


def mean_aligned_rotation_error_degrees(
    rotations: np.ndarray,
    centers: np.ndarray,
    true_rotations: np.ndarray,
    true_centers: np.ndarray,
) -> float:
    aligned_rotations, _ = _similarity_aligned_poses(
        rotations, centers, true_centers
    )
    return mean_rotation_error_degrees(aligned_rotations, true_rotations)


def mean_aligned_center_error_mm(
    centers: np.ndarray, true_centers: np.ndarray
) -> float:
    identity_rotations = np.repeat(np.eye(3)[None, :, :], centers.shape[0], axis=0)
    _, aligned_centers = _similarity_aligned_poses(
        identity_rotations, centers, true_centers
    )
    return mean_center_error_mm(aligned_centers, true_centers)


def run_basin_sweep(
    levels: Iterable[float],
    seeds: Iterable[int],
    *,
    maximum_iterations_per_stage: int = 18,
    initial_depth_meters: float = 1.0,
    decoy_pose_seed: int = DECOY_POSE_SEED,
    decoy_pose_seeds: Iterable[int] | None = None,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    level_values = tuple(levels)
    pose_seeds = tuple(seeds)
    graph_seeds = (
        tuple(decoy_pose_seeds)
        if decoy_pose_seeds is not None
        else (decoy_pose_seed,)
    )
    for graph_seed in graph_seeds:
        for level in level_values:
            for seed in pose_seeds:
                scene = make_scene(
                    level, seed, decoy_pose_seed=graph_seed
                )
                plain = solve_ray_depth(
                    scene,
                    (NOMINAL_LOSS_SCALE,),
                    maximum_iterations_per_stage=(
                        maximum_iterations_per_stage * len(GNC_LOSS_SCALES)
                    ),
                    initial_depth_meters=initial_depth_meters,
                )
                gnc = solve_ray_depth(
                    scene,
                    GNC_LOSS_SCALES,
                    maximum_iterations_per_stage=maximum_iterations_per_stage,
                    initial_depth_meters=initial_depth_meters,
                )
                rows.append(
                    SweepRow(
                        perturbation_degrees=level,
                        seed=seed,
                        decoy_pose_seed=graph_seed,
                        prior_rotation_error=(
                            mean_aligned_rotation_error_degrees(
                                scene.prior_rotations,
                                scene.prior_centers,
                                scene.true_rotations,
                                scene.true_centers,
                            )
                        ),
                        plain_rotation_error=(
                            mean_aligned_rotation_error_degrees(
                                plain.rotations,
                                plain.centers,
                                scene.true_rotations,
                                scene.true_centers,
                            )
                        ),
                        gnc_rotation_error=mean_aligned_rotation_error_degrees(
                            gnc.rotations,
                            gnc.centers,
                            scene.true_rotations,
                            scene.true_centers,
                        ),
                        prior_center_error_mm=mean_aligned_center_error_mm(
                            scene.prior_centers, scene.true_centers
                        ),
                        plain_center_error_mm=mean_aligned_center_error_mm(
                            plain.centers, scene.true_centers
                        ),
                        gnc_center_error_mm=mean_aligned_center_error_mm(
                            gnc.centers, scene.true_centers
                        ),
                        prior_absolute_center_error_mm=mean_center_error_mm(
                            scene.prior_centers, scene.true_centers
                        ),
                        plain_absolute_center_error_mm=mean_center_error_mm(
                            plain.centers, scene.true_centers
                        ),
                        gnc_absolute_center_error_mm=mean_center_error_mm(
                            gnc.centers, scene.true_centers
                        ),
                    )
                )
    return rows


def run_structure_control_sweep(
    levels: Iterable[float],
    seeds: Iterable[int],
    gates: Iterable[float],
    *,
    maximum_iterations: int = 24,
) -> list[StructureControlRow]:
    """Compare prior- and accurate-pose structure on clean known tracks."""
    rows = []
    for level in levels:
        for seed in seeds:
            scene = make_scene(level, seed, decoy_fraction=0.0)
            for gate_pixels in gates:
                prior_triangulation = triangulate_known_tracks(
                    scene,
                    scene.prior_rotations,
                    scene.prior_centers,
                    gate_pixels=gate_pixels,
                )
                accurate_triangulation = triangulate_known_tracks(
                    scene,
                    scene.true_rotations,
                    scene.true_centers,
                    gate_pixels=gate_pixels,
                )
                prior = solve_shared_point_bundle_adjustment(
                    scene,
                    prior_triangulation,
                    maximum_iterations=maximum_iterations,
                )
                accurate = solve_shared_point_bundle_adjustment(
                    scene,
                    accurate_triangulation,
                    maximum_iterations=maximum_iterations,
                )

                def rotation_error(result: SharedPointBAResult) -> float:
                    if not result.runnable:
                        return math.nan
                    return mean_aligned_rotation_error_degrees(
                        result.rotations,
                        result.centers,
                        scene.true_rotations,
                        scene.true_centers,
                    )

                def aligned_center_error(result: SharedPointBAResult) -> float:
                    if not result.runnable:
                        return math.nan
                    return mean_aligned_center_error_mm(
                        result.centers, scene.true_centers
                    )

                def absolute_center_error(result: SharedPointBAResult) -> float:
                    if not result.runnable:
                        return math.nan
                    return mean_center_error_mm(
                        result.centers, scene.true_centers
                    )

                rows.append(
                    StructureControlRow(
                        perturbation_degrees=level,
                        seed=seed,
                        gate_pixels=gate_pixels,
                        camera_count=scene.camera_count,
                        prior_track_count=prior_triangulation.track_count,
                        prior_observation_count=(
                            prior_triangulation.observation_count
                        ),
                        prior_observed_camera_count=(
                            prior_triangulation.observed_camera_count
                        ),
                        prior_camera_component_count=(
                            prior_triangulation.camera_component_count
                        ),
                        prior_all_cameras_connected=(
                            prior_triangulation.all_cameras_connected
                        ),
                        prior_rotation_error=rotation_error(prior),
                        prior_center_error_mm=aligned_center_error(prior),
                        prior_absolute_center_error_mm=absolute_center_error(
                            prior
                        ),
                        prior_minimum_depth=prior.minimum_depth,
                        accurate_track_count=accurate_triangulation.track_count,
                        accurate_observation_count=(
                            accurate_triangulation.observation_count
                        ),
                        accurate_observed_camera_count=(
                            accurate_triangulation.observed_camera_count
                        ),
                        accurate_camera_component_count=(
                            accurate_triangulation.camera_component_count
                        ),
                        accurate_all_cameras_connected=(
                            accurate_triangulation.all_cameras_connected
                        ),
                        accurate_rotation_error=rotation_error(accurate),
                        accurate_center_error_mm=aligned_center_error(accurate),
                        accurate_absolute_center_error_mm=absolute_center_error(
                            accurate
                        ),
                        accurate_minimum_depth=accurate.minimum_depth,
                    )
                )
    return rows


def _interpolated_poses(
    scene: SyntheticScene, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    rotations = scene.prior_rotations.copy()
    for camera_index in range(scene.camera_count):
        correction = so3_log(
            scene.true_rotations[camera_index]
            @ scene.prior_rotations[camera_index].T
        )
        rotations[camera_index] = (
            so3_exp(alpha * correction) @ scene.prior_rotations[camera_index]
        )
    centers = scene.prior_centers + alpha * (
        scene.true_centers - scene.prior_centers
    )
    return rotations, centers


def _triangulated_structure_cost(
    scene: SyntheticScene,
    rotations: np.ndarray,
    centers: np.ndarray,
    triangulation: TriangulationResult,
    loss_scale: float,
) -> float:
    cost = 0.0
    for point, observations in zip(
        triangulation.points, triangulation.observations, strict=True
    ):
        for camera_index, point_index in observations:
            residual = (
                _project(
                    rotations[camera_index],
                    centers[camera_index],
                    point[None, :],
                    scene.focal,
                    scene.principal,
                )[0]
                - scene.pixels[camera_index, point_index]
            )
            cost += float(arctan_loss(np.asarray([residual @ residual]), loss_scale)[0])
    return cost / (2.0 * triangulation.observation_count)


def _marginalized_ray_depth_cost(
    scene: SyntheticScene,
    rotations: np.ndarray,
    centers: np.ndarray,
    loss_scale: float,
    *,
    candidate_depth_count: int = 72,
) -> float:
    if candidate_depth_count < 2:
        raise ValueError("candidate_depth_count must be at least 2")
    rays = np.array(
        [
            _world_rays(rotation, pixels, scene.focal, scene.principal)
            for rotation, pixels in zip(rotations, scene.pixels, strict=True)
        ]
    )
    outgoing: list[list[tuple[int, int]]] = [
        [] for _ in range(scene.camera_count * scene.point_count)
    ]
    for first, first_point, second, second_point in scene.matches:
        outgoing[first * scene.point_count + first_point].append(
            (int(second), int(second_point))
        )
        outgoing[second * scene.point_count + second_point].append(
            (int(first), int(first_point))
        )

    candidate_depths = np.geomspace(0.2, 4.0, candidate_depth_count)
    total = 0.0
    for camera_index in range(scene.camera_count):
        for point_index in range(scene.point_count):
            targets = outgoing[camera_index * scene.point_count + point_index]
            candidate_points = (
                centers[camera_index]
                + candidate_depths[:, None] * rays[camera_index, point_index]
            )
            candidate_costs = np.zeros(candidate_depths.size)
            for target_camera, target_point in targets:
                residuals = (
                    _project(
                        rotations[target_camera],
                        centers[target_camera],
                        candidate_points,
                        scene.focal,
                        scene.principal,
                    )
                    - scene.pixels[target_camera, target_point]
                )
                candidate_costs += arctan_loss(
                    np.sum(residuals * residuals, axis=1), loss_scale
                )
            total += float(np.min(candidate_costs))
    return total / (4.0 * scene.matches.shape[0])


def run_mechanism_diagnostic(
    *,
    perturbation_degrees: float = 16.0,
    seed: int = 0,
    alpha_count: int = 21,
    candidate_depth_count: int = 72,
) -> MechanismDiagnostic:
    scene = make_scene(perturbation_degrees, seed, decoy_fraction=0.0)
    prior_structure = triangulate_known_tracks(
        scene,
        scene.prior_rotations,
        scene.prior_centers,
        gate_pixels=4.0,
    )
    oracle_structure = triangulate_known_tracks(
        scene,
        scene.true_rotations,
        scene.true_centers,
        gate_pixels=4.0,
    )
    alphas = np.linspace(0.0, 1.0, alpha_count)
    prior_costs = []
    oracle_costs = []
    ray_costs = []
    smooth_ray_costs = []
    for alpha in alphas:
        rotations, centers = _interpolated_poses(scene, float(alpha))
        prior_costs.append(
            _triangulated_structure_cost(
                scene,
                rotations,
                centers,
                prior_structure,
                NOMINAL_LOSS_SCALE,
            )
        )
        oracle_costs.append(
            _triangulated_structure_cost(
                scene,
                rotations,
                centers,
                oracle_structure,
                NOMINAL_LOSS_SCALE,
            )
        )
        ray_costs.append(
            _marginalized_ray_depth_cost(
                scene,
                rotations,
                centers,
                NOMINAL_LOSS_SCALE,
                candidate_depth_count=candidate_depth_count,
            )
        )
        smooth_ray_costs.append(
            _marginalized_ray_depth_cost(
                scene,
                rotations,
                centers,
                GNC_LOSS_SCALES[0],
                candidate_depth_count=candidate_depth_count,
            )
        )
    prior_array = np.asarray(prior_costs)
    oracle_array = np.asarray(oracle_costs)
    ray_array = np.asarray(ray_costs)
    smooth_array = np.asarray(smooth_ray_costs)

    def normalized(values: np.ndarray) -> np.ndarray:
        span = float(np.max(values) - np.min(values))
        if span <= np.finfo(np.float64).eps:
            return np.zeros_like(values)
        return (values - np.min(values)) / span

    normalized_ray = normalized(ray_array)
    normalized_smooth = normalized(smooth_array)
    forward_index = 1
    return MechanismDiagnostic(
        alphas=alphas,
        prior_structure_costs=prior_array,
        oracle_structure_costs=oracle_array,
        ray_depth_costs=ray_array,
        prior_structure_minimum=float(
            alphas[int(np.argmin(prior_array))]
        ),
        oracle_structure_minimum=float(alphas[int(np.argmin(oracle_array))]),
        ray_depth_minimum=float(alphas[int(np.argmin(ray_array))]),
        nominal_normalized_forward_drop=float(
            normalized_ray[0] - normalized_ray[forward_index]
        ),
        smooth_normalized_forward_drop=float(
            normalized_smooth[0] - normalized_smooth[forward_index]
        ),
        prior_retained_observation_fraction=(
            prior_structure.observation_count
            / (scene.camera_count * scene.point_count)
        ),
        prior_observed_camera_count=prior_structure.observed_camera_count,
        prior_camera_count=scene.camera_count,
        prior_camera_component_count=prior_structure.camera_component_count,
    )


def _format_sweep(rows: Sequence[SweepRow]) -> str:
    lines = [
        "graph_seed level pose_seed prior_rot plain_rot gnc_rot prior_aligned_mm "
        "plain_aligned_mm gnc_aligned_mm prior_abs_mm plain_abs_mm gnc_abs_mm",
    ]
    for row in rows:
        lines.append(
            f"{row.decoy_pose_seed:10d} "
            f"{row.perturbation_degrees:5.1f} {row.seed:9d} "
            f"{row.prior_rotation_error:9.3f} "
            f"{row.plain_rotation_error:9.3f} "
            f"{row.gnc_rotation_error:7.3f} "
            f"{row.prior_center_error_mm:8.2f} "
            f"{row.plain_center_error_mm:8.2f} "
            f"{row.gnc_center_error_mm:6.2f} "
            f"{row.prior_absolute_center_error_mm:12.2f} "
            f"{row.plain_absolute_center_error_mm:12.2f} "
            f"{row.gnc_absolute_center_error_mm:10.2f}"
        )
    return "\n".join(lines)


def _format_summary(rows: Sequence[SweepRow]) -> str:
    lines = ["level runs plain_success gnc_success plain_med gnc_med"]
    levels = sorted({row.perturbation_degrees for row in rows})
    for level in levels:
        selected = [row for row in rows if row.perturbation_degrees == level]
        plain_successes = sum(row.plain_success for row in selected)
        gnc_successes = sum(row.gnc_success for row in selected)
        plain_median = float(
            np.median([row.plain_rotation_error for row in selected])
        )
        gnc_median = float(
            np.median([row.gnc_rotation_error for row in selected])
        )
        lines.append(
            f"{level:5.1f} {len(selected):4d} "
            f"{plain_successes:13d} {gnc_successes:11d} "
            f"{plain_median:9.3f} {gnc_median:7.3f}"
        )
    return "\n".join(lines)


def _format_structure_control(rows: Sequence[StructureControlRow]) -> str:
    lines = [
        "gate level seed prior_tracks prior_obs prior_cams prior_cc prior_valid "
        "prior_min_z prior_rot prior_mm prior_abs_mm accurate_tracks accurate_obs "
        "accurate_cams accurate_cc accurate_valid accurate_min_z accurate_rot "
        "accurate_mm accurate_abs_mm"
    ]
    for row in rows:
        lines.append(
            f"{row.gate_pixels:4.0f} "
            f"{row.perturbation_degrees:5.1f} {row.seed:4d} "
            f"{row.prior_track_count:12d} {row.prior_observation_count:9d} "
            f"{row.prior_observed_camera_count:5d}/{row.camera_count:<2d} "
            f"{row.prior_camera_component_count:8d} "
            f"{str(row.prior_solution_valid):11s} "
            f"{row.prior_minimum_depth:11.3f} "
            f"{row.prior_rotation_error:9.3f} "
            f"{row.prior_center_error_mm:8.2f} "
            f"{row.prior_absolute_center_error_mm:12.2f} "
            f"{row.accurate_track_count:15d} "
            f"{row.accurate_observation_count:12d} "
            f"{row.accurate_observed_camera_count:8d}/{row.camera_count:<2d} "
            f"{row.accurate_camera_component_count:11d} "
            f"{str(row.accurate_solution_valid):14s} "
            f"{row.accurate_minimum_depth:14.3f} "
            f"{row.accurate_rotation_error:12.3f} "
            f"{row.accurate_center_error_mm:11.2f} "
            f"{row.accurate_absolute_center_error_mm:15.2f}"
        )
    return "\n".join(lines)


def _format_structure_summary(rows: Sequence[StructureControlRow]) -> str:
    lines = [
        "gate level runs prior_valid prior_success accurate_valid "
        "accurate_success prior_valid_med accurate_valid_med"
    ]
    cells = sorted(
        {(row.gate_pixels, row.perturbation_degrees) for row in rows}
    )
    for gate_pixels, level in cells:
        selected = [
            row
            for row in rows
            if row.gate_pixels == gate_pixels
            and row.perturbation_degrees == level
        ]
        prior_errors = [
            row.prior_rotation_error
            for row in selected
            if row.prior_solution_valid
        ]
        accurate_errors = [
            row.accurate_rotation_error
            for row in selected
            if row.accurate_solution_valid
        ]
        prior_median = float(np.median(prior_errors)) if prior_errors else math.nan
        accurate_median = (
            float(np.median(accurate_errors)) if accurate_errors else math.nan
        )
        lines.append(
            f"{gate_pixels:4.0f} {level:5.1f} {len(selected):4d} "
            f"{sum(row.prior_solution_valid for row in selected):11d} "
            f"{sum(row.prior_success for row in selected):13d} "
            f"{sum(row.accurate_solution_valid for row in selected):14d} "
            f"{sum(row.accurate_success for row in selected):16d} "
            f"{prior_median:9.3f} {accurate_median:12.3f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels", default="1,2,4,8,16,32", help="comma-separated degrees"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--iterations",
        type=int,
        default=18,
        help="iterations per GNC stage; plain gets the same total budget",
    )
    parser.add_argument(
        "--initial-depth",
        type=float,
        default=1.0,
        help="initial lifted depth in meters",
    )
    parser.add_argument(
        "--decoy-pose-seed",
        type=int,
        default=DECOY_POSE_SEED,
        help="fixed false-pose lobe seed for the outlier match graph",
    )
    parser.add_argument(
        "--decoy-pose-seeds",
        default="",
        help=(
            "optional comma-separated false-lobe seeds; overrides "
            "--decoy-pose-seed and aggregates all runs"
        ),
    )
    parser.add_argument(
        "--structure-gates",
        default="",
        help="optional comma-separated pixel gates for known-track point BA",
    )
    parser.add_argument(
        "--structure-iterations",
        type=int,
        default=24,
        help="maximum iterations for each shared-point BA control",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="skip the lifted-objective sweep and run only structure controls",
    )
    args = parser.parse_args(argv)
    levels = tuple(float(value) for value in args.levels.split(","))
    if args.structure_only and not args.structure_gates:
        parser.error("--structure-only requires --structure-gates")
    if not args.structure_only:
        diagnostic = run_mechanism_diagnostic()
        print(
            "mechanism minima (alpha 0=prior, 1=truth): "
            f"prior-structure={diagnostic.prior_structure_minimum:.2f}, "
            f"oracle-structure={diagnostic.oracle_structure_minimum:.2f}, "
            f"ray-depth={diagnostic.ray_depth_minimum:.2f}"
        )
        print(
            "first-step normalized ray-depth cost drop: "
            f"a=1e2 {diagnostic.nominal_normalized_forward_drop:.6f}, "
            f"a=1e4 {diagnostic.smooth_normalized_forward_drop:.6f}"
        )
        print(
            "best-pair prior triangulation retained observations (4 px gate): "
            f"{100.0 * diagnostic.prior_retained_observation_fraction:.1f}%; "
            "observed cameras "
            f"{diagnostic.prior_observed_camera_count}/"
            f"{diagnostic.prior_camera_count} in "
            f"{diagnostic.prior_camera_component_count} component"
        )
        graph_seeds = None
        if args.decoy_pose_seeds:
            graph_seeds = tuple(
                int(value) for value in args.decoy_pose_seeds.split(",")
            )
        rows = run_basin_sweep(
            levels,
            range(args.seeds),
            maximum_iterations_per_stage=args.iterations,
            initial_depth_meters=args.initial_depth,
            decoy_pose_seed=args.decoy_pose_seed,
            decoy_pose_seeds=graph_seeds,
        )
        print(_format_sweep(rows))
        print(_format_summary(rows))
    if args.structure_gates:
        gates = tuple(
            float(value) for value in args.structure_gates.split(",")
        )
        structure_rows = run_structure_control_sweep(
            levels,
            range(args.seeds),
            gates,
            maximum_iterations=args.structure_iterations,
        )
        print(
            "matched-loss-and-position-prior shared-point BA on clean known tracks"
        )
        print(_format_structure_control(structure_rows))
        print(_format_structure_summary(structure_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
