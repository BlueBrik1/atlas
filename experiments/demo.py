"""Run the five-frame ATLAS method on the first seeded benchmark sequence."""
import atlas_temporal_experiment as temporal


def main():
    rng = temporal.np.random.default_rng(20261103)
    condition, direction, frames, _ = temporal.sequences(0, rng)
    points, magnitudes, tracks = temporal.persist(frames, need=3)
    result, _, _, candidates = temporal.match(points, magnitudes, direction, "ATLAS")
    print(f"Noise condition: {condition}")
    print(f"Retained tracks: {tracks['nstable']}, shortlisted fields: {len(candidates)}")
    print(f"Refined boresight error: {result['refined_error_deg']:.5f} degrees")


if __name__ == "__main__":
    main()
