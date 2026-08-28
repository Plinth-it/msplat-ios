"""Explicit research regressions; intentionally outside production CI."""

import unittest

import numpy as np

from research.triangulation_free_ba.validation import (
    GNC_LOSS_SCALES,
    NOMINAL_LOSS_SCALE,
    TriangulationResult,
    _cross_projection_residuals,
    make_scene,
    mean_aligned_center_error_mm,
    mean_aligned_rotation_error_degrees,
    mean_rotation_error_degrees,
    replace_known_track_points,
    run_basin_sweep,
    run_mechanism_diagnostic,
    solve_ray_depth,
    solve_shared_point_bundle_adjustment,
    triangulate_known_tracks,
)


class TriangulationFreeBundleAdjustmentClaimsTests(unittest.TestCase):
    def test_camera_graph_requires_exact_coverage_and_connectivity(self):
        disconnected = TriangulationResult(
            points=np.zeros((2, 3)),
            observations=(
                np.array(((0, 0), (1, 0))),
                np.array(((2, 1), (3, 1))),
            ),
            source_track_count=2,
            camera_count=4,
        )
        malformed_indices = TriangulationResult(
            points=np.zeros((1, 3)),
            observations=(np.array(((0, 0), (4, 0))),),
            source_track_count=1,
            camera_count=2,
        )
        self.assertEqual(disconnected.observed_camera_count, 4)
        self.assertEqual(disconnected.camera_component_count, 2)
        self.assertFalse(disconnected.all_cameras_connected)
        self.assertFalse(malformed_indices.all_cameras_connected)

    def test_symmetric_cross_projection_is_zero_at_truth(self):
        scene = make_scene(
            0.0, seed=3, pixel_noise=0.0, decoy_fraction=0.0
        )
        depths = np.empty((scene.camera_count, scene.point_count))
        for camera_index in range(scene.camera_count):
            depths[camera_index] = np.linalg.norm(
                scene.points - scene.true_centers[camera_index], axis=1
            )
        residuals = _cross_projection_residuals(
            scene,
            scene.true_rotations,
            scene.true_centers,
            np.log(depths).reshape(-1),
        )
        self.assertLess(float(np.max(np.abs(residuals))), 1.0e-9)

    def test_match_graph_is_fixed_across_perturbations_and_seeds(self):
        first = make_scene(1.0, seed=0)
        second = make_scene(32.0, seed=1)
        alternative_decoys = make_scene(
            32.0, seed=1, decoy_pose_seed=10_001
        )
        np.testing.assert_array_equal(first.pixels, second.pixels)
        np.testing.assert_array_equal(first.matches, second.matches)
        np.testing.assert_array_equal(second.pixels, alternative_decoys.pixels)
        self.assertFalse(
            np.array_equal(second.matches, alternative_decoys.matches)
        )
        self.assertAlmostEqual(
            mean_rotation_error_degrees(
                second.prior_rotations, second.true_rotations
            ),
            32.0,
            places=9,
        )

    def test_committed_structure_and_ray_depth_landscapes_diverge(self):
        diagnostic = run_mechanism_diagnostic(
            perturbation_degrees=16.0, seed=0, alpha_count=21
        )
        self.assertLessEqual(diagnostic.prior_structure_minimum, 0.1)
        self.assertAlmostEqual(
            diagnostic.prior_retained_observation_fraction,
            16.0 / 72.0,
        )
        self.assertEqual(diagnostic.prior_observed_camera_count, 4)
        self.assertEqual(diagnostic.prior_camera_count, 6)
        self.assertEqual(diagnostic.prior_camera_component_count, 1)
        self.assertGreaterEqual(diagnostic.oracle_structure_minimum, 0.9)
        self.assertGreaterEqual(diagnostic.ray_depth_minimum, 0.9)
        self.assertGreater(
            diagnostic.nominal_normalized_forward_drop, 0.0
        )
        self.assertGreater(
            diagnostic.smooth_normalized_forward_drop,
            5.0 * diagnostic.nominal_normalized_forward_drop,
        )

    def test_fixed_outlier_graph_has_32_degree_gnc_advantage(self):
        rows = run_basin_sweep(
            (32.0,), range(3), maximum_iterations_per_stage=8
        )
        plain_successes = sum(row.plain_success for row in rows)
        gnc_successes = sum(row.gnc_success for row in rows)
        self.assertEqual(plain_successes, 0)
        self.assertGreaterEqual(gnc_successes, 2)
        self.assertLess(
            np.median([row.gnc_rotation_error for row in rows]),
            np.median([row.plain_rotation_error for row in rows]),
        )

    def test_outlier_graph_seed_axis_is_explicit(self):
        rows = run_basin_sweep(
            iter((0.0,)),
            iter((0,)),
            maximum_iterations_per_stage=0,
            decoy_pose_seeds=(10_000, 10_001),
        )
        self.assertEqual(
            [row.decoy_pose_seed for row in rows],
            [10_000, 10_001],
        )

    def test_structure_gating_removes_whole_rig_observability(self):
        scene = make_scene(16.0, seed=0, decoy_fraction=0.0)
        prior_structure = triangulate_known_tracks(
            scene,
            scene.prior_rotations,
            scene.prior_centers,
            gate_pixels=4.0,
        )
        accurate_structure = triangulate_known_tracks(
            scene,
            scene.true_rotations,
            scene.true_centers,
            gate_pixels=4.0,
        )
        ungated_prior_structure = triangulate_known_tracks(
            scene,
            scene.prior_rotations,
            scene.prior_centers,
            gate_pixels=np.inf,
        )
        accurate_points_sparse_mask = replace_known_track_points(
            accurate_structure, prior_structure
        )
        self.assertEqual(prior_structure.track_count, 8)
        self.assertEqual(prior_structure.observation_count, 16)
        self.assertEqual(prior_structure.observed_camera_indices, (0, 1, 3, 4))
        self.assertEqual(prior_structure.camera_component_count, 1)
        self.assertFalse(prior_structure.all_cameras_connected)
        self.assertEqual(accurate_structure.observation_count, 72)
        self.assertTrue(accurate_structure.all_cameras_connected)
        self.assertEqual(ungated_prior_structure.observation_count, 72)
        self.assertTrue(ungated_prior_structure.all_cameras_connected)
        self.assertEqual(accurate_points_sparse_mask.observation_count, 16)
        self.assertFalse(accurate_points_sparse_mask.all_cameras_connected)
        prior = solve_shared_point_bundle_adjustment(
            scene, prior_structure, maximum_iterations=24
        )
        accurate = solve_shared_point_bundle_adjustment(
            scene, accurate_structure, maximum_iterations=24
        )
        ungated_prior = solve_shared_point_bundle_adjustment(
            scene, ungated_prior_structure, maximum_iterations=24
        )
        accurate_sparse = solve_shared_point_bundle_adjustment(
            scene, accurate_points_sparse_mask, maximum_iterations=24
        )
        prior_rotation = mean_aligned_rotation_error_degrees(
            prior.rotations,
            prior.centers,
            scene.true_rotations,
            scene.true_centers,
        )
        accurate_rotation = mean_aligned_rotation_error_degrees(
            accurate.rotations,
            accurate.centers,
            scene.true_rotations,
            scene.true_centers,
        )
        ungated_prior_rotation = mean_aligned_rotation_error_degrees(
            ungated_prior.rotations,
            ungated_prior.centers,
            scene.true_rotations,
            scene.true_centers,
        )
        accurate_sparse_rotation = mean_aligned_rotation_error_degrees(
            accurate_sparse.rotations,
            accurate_sparse.centers,
            scene.true_rotations,
            scene.true_centers,
        )
        self.assertGreater(prior_rotation, 10.0)
        self.assertLess(accurate_rotation, 1.0)
        self.assertLess(ungated_prior_rotation, 1.0)
        self.assertGreater(accurate_sparse_rotation, 5.0)
        for result in (prior, accurate, ungated_prior, accurate_sparse):
            self.assertGreater(result.minimum_depth, 0.0)
        for camera_index in (2, 5):
            np.testing.assert_allclose(
                prior.rotations[camera_index],
                scene.prior_rotations[camera_index],
                atol=1.0e-12,
            )

    def test_known_track_retention_increases_with_explicit_gate(self):
        scene = make_scene(16.0, seed=0, decoy_fraction=0.0)
        retained = [
            triangulate_known_tracks(
                scene,
                scene.prior_rotations,
                scene.prior_centers,
                gate_pixels=gate,
            ).observation_count
            for gate in (2.0, 4.0, 8.0, 16.0)
        ]
        self.assertEqual(retained, sorted(retained))
        self.assertGreater(retained[-1], retained[0])

    def test_ray_depth_landscape_minimum_is_stable_to_grid_resolution(self):
        minima = [
            run_mechanism_diagnostic(
                perturbation_degrees=16.0,
                seed=0,
                alpha_count=21,
                candidate_depth_count=count,
            ).ray_depth_minimum
            for count in (36, 72, 144)
        ]
        self.assertTrue(all(minimum >= 0.9 for minimum in minima))

    def test_clean_accurate_prior_is_not_materially_moved(self):
        scene = make_scene(
            0.0, seed=0, pixel_noise=0.15, decoy_fraction=0.0
        )
        plain = solve_ray_depth(
            scene, (NOMINAL_LOSS_SCALE,), maximum_iterations_per_stage=45
        )
        gnc = solve_ray_depth(
            scene, GNC_LOSS_SCALES, maximum_iterations_per_stage=15
        )
        for result in (plain, gnc):
            self.assertLess(
                mean_aligned_rotation_error_degrees(
                    result.rotations,
                    result.centers,
                    scene.true_rotations,
                    scene.true_centers,
                ),
                0.2,
            )
            self.assertLess(
                mean_aligned_center_error_mm(
                    result.centers, scene.true_centers
                ),
                5.0,
            )


if __name__ == "__main__":
    unittest.main()
