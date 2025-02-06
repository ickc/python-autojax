#!/usr/bin/env python

import numpy as np

from autojax.jax import log_likelihood_function as log_likelihood_function_jax
from autojax.numba import log_likelihood_function as log_likelihood_function_numba
from autojax.original import log_likelihood_function


def external():
    from pathlib import Path

    import autolens as al

    real_space_mask = al.Mask2D.circular(
        shape_native=(100, 100),
        pixel_scales=0.2,
        radius=3.0,
    )
    dataset_type = "sma"
    dataset_path = Path(__file__).parent / ".." / ".." / "dirac_rse_interferometer" / "dataset" / dataset_type
    dataset = al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )
    del real_space_mask, dataset_type, dataset_path
    dirty_image = dataset.w_tilde.dirty_image
    w_tilde = dataset.w_tilde.w_matrix
    noise_map = dataset.noise_map

    mass = al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    )
    lens_galaxy = al.Galaxy(redshift=0.5, mass=mass)
    del mass

    pixelization = al.Pixelization(
        image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
        mesh=al.mesh.Delaunay(),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)
    del pixelization

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    del lens_galaxy, source_galaxy
    tracer_to_inversion = al.TracerToInversion(tracer=tracer, dataset=dataset)
    del dataset, tracer
    inversion = tracer_to_inversion.inversion
    del tracer_to_inversion
    mapping_matrix = inversion.mapping_matrix
    mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]
    del inversion

    neighbors = mapper.source_plane_mesh_grid.neighbors
    neighbors_sizes = mapper.source_plane_mesh_grid.neighbors.sizes
    del mapper
    return (
        dirty_image,
        w_tilde,
        noise_map,
        mapping_matrix,
        neighbors,
        neighbors_sizes,
    )


def main() -> float:
    ref = -3028.5513427463716

    dirty_image, w_tilde, noise_map, mapping_matrix, neighbors, neighbors_sizes = external()
    for name in ("dirty_image", "w_tilde", "noise_map", "mapping_matrix", "neighbors", "neighbors_sizes"):
        print("=========", name, "=========")
        var = np.array(locals()[name])
        print(np.info(var))
    log_likelihood = log_likelihood_function(
        dirty_image,
        w_tilde,
        noise_map,
        mapping_matrix,
        neighbors,
        neighbors_sizes,
    )
    print("=========", "log_likelihood", "=========")
    print(log_likelihood)
    np.testing.assert_allclose(log_likelihood, ref)
    log_likelihood_numba = log_likelihood_function_numba(
        np.array(dirty_image),
        w_tilde,
        np.array(noise_map),
        mapping_matrix,
        neighbors,
        neighbors_sizes,
    )
    print("=========", "log_likelihood_numba", "=========")
    print(log_likelihood_numba)
    np.testing.assert_allclose(log_likelihood_numba, ref)
    log_likelihood_jax = log_likelihood_function_jax(
        np.array(dirty_image),
        w_tilde,
        np.array(noise_map),
        mapping_matrix,
        neighbors,
        neighbors_sizes,
    )
    print("=========", "log_likelihood_jax", "=========")
    print(log_likelihood_jax)
    return log_likelihood


if __name__ == "__main__":
    log_likelihood = main()
