#!/usr/bin/env python

from pathlib import Path

import autolens as al
import numpy as np

from autojax.jax import log_likelihood_function as log_likelihood_function_jax
from autojax.numba import log_likelihood_function as log_likelihood_function_numba
from autojax.original import log_likelihood_function, log_likelihood_function_via_preload_method


def external():
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
    data = dataset.data
    noise_map = dataset.noise_map
    uv_wavelengths = dataset.uv_wavelengths
    grid_radians_slim = dataset.grid.in_radians
    shape_masked_pixels_2d = dataset.transformer.grid.mask.shape_native_masked_pixels
    grid_radians_2d = np.array(dataset.transformer.grid.mask.derive_grid.all_false.in_radians.native)

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
    del tracer
    inversion = tracer_to_inversion.inversion
    del tracer_to_inversion
    mapping_matrix = inversion.mapping_matrix
    mapper = inversion.cls_list_from(cls=al.AbstractMapper)[0]
    del inversion

    neighbors = mapper.source_plane_mesh_grid.neighbors
    neighbors_sizes = mapper.source_plane_mesh_grid.neighbors.sizes

    pix_indexes_for_sub_slim_index = mapper.pix_indexes_for_sub_slim_index
    pix_size_for_sub_slim_index = mapper.pix_sizes_for_sub_slim_index
    pix_weights_for_sub_slim_index = mapper.pix_weights_for_sub_slim_index
    native_index_for_slim_index = dataset.transformer.real_space_mask.derive_indexes.native_for_slim
    pix_pixels = mapper.pixels
    del dataset, mapper
    return (
        dirty_image,
        data,
        noise_map,
        uv_wavelengths,
        grid_radians_slim,
        shape_masked_pixels_2d,
        grid_radians_2d,
        mapping_matrix,
        neighbors,
        neighbors_sizes,
        pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        native_index_for_slim_index,
        pix_pixels,
    )


def main() -> float:
    ref = -13401.986947103405

    (
        dirty_image,
        data,
        noise_map,
        uv_wavelengths,
        grid_radians_slim,
        shape_masked_pixels_2d,
        grid_radians_2d,
        mapping_matrix,
        neighbors,
        neighbors_sizes,
        pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        native_index_for_slim_index,
        pix_pixels,
    ) = external()
    for name in (
        "dirty_image",
        "data",
        "noise_map",
        "uv_wavelengths",
        "grid_radians_slim",
        "shape_masked_pixels_2d",
        "grid_radians_2d",
        "mapping_matrix",
        "neighbors",
        "neighbors_sizes",
        "pix_indexes_for_sub_slim_index",
        "pix_size_for_sub_slim_index",
        "pix_weights_for_sub_slim_index",
        "native_index_for_slim_index",
        "pix_pixels",
    ):
        print("=========", name, "=========")
        var = np.array(locals()[name])
        print(np.info(var))

    for name in ("shape_masked_pixels_2d", "pix_pixels"):
        print("=========", name, "=========")
        var = locals()[name]
        print(var)

    log_likelihood = log_likelihood_function(
        dirty_image,
        data,
        np.array(noise_map),
        np.array(uv_wavelengths),
        np.array(grid_radians_slim),
        mapping_matrix,
        neighbors,
        neighbors_sizes,
    )
    print("=========", "log_likelihood", "=========")
    print(log_likelihood)
    np.testing.assert_allclose(log_likelihood, ref)

    log_likelihood_via_preload_method = log_likelihood_function_via_preload_method(
        dirty_image,
        data,
        np.array(noise_map),
        shape_masked_pixels_2d,
        grid_radians_2d,
        uv_wavelengths,
        mapping_matrix,
        neighbors,
        neighbors_sizes,
        pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
        native_index_for_slim_index,
        pix_pixels,
    )
    print("=========", "log_likelihood_function_via_preload_method", "=========")
    print(log_likelihood_via_preload_method)
    np.testing.assert_allclose(log_likelihood_via_preload_method, ref)

    log_likelihood_numba = log_likelihood_function_numba(
        np.array(dirty_image),
        np.array(data),
        np.array(noise_map),
        np.array(uv_wavelengths),
        np.array(grid_radians_slim),
        mapping_matrix,
        neighbors,
        neighbors_sizes,
    )
    print("=========", "log_likelihood_numba", "=========")
    print(log_likelihood_numba)
    np.testing.assert_allclose(log_likelihood_numba, ref)

    log_likelihood_jax = log_likelihood_function_jax(
        np.array(dirty_image),
        np.array(data),
        np.array(noise_map),
        np.array(uv_wavelengths),
        np.array(grid_radians_slim),
        mapping_matrix,
        neighbors,
        neighbors_sizes,
    )
    print("=========", "log_likelihood_jax", "=========")
    print(log_likelihood_jax)
    return log_likelihood


if __name__ == "__main__":
    log_likelihood = main()
