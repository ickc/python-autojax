#!/usr/bin/env python

from pathlib import Path

import autolens as al
import numpy as np


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
    pix_weights_for_sub_slim_index = mapper.pix_weights_for_sub_slim_index
    del dataset, mapper
    return (
        dirty_image,
        data,
        noise_map,
        uv_wavelengths,
        mapping_matrix,
        neighbors,
        neighbors_sizes,
        pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
    )


def main():
    (
        dirty_image,
        data,
        noise_map,
        uv_wavelengths,
        mapping_matrix,
        neighbors,
        neighbors_sizes,
        pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index,
    ) = external()
    np.savez(
        Path(__file__).parent / ".." / "tests" / "data.npz",
        dirty_image=dirty_image,
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        mapping_matrix=mapping_matrix,
        neighbors=neighbors,
        neighbors_sizes=neighbors_sizes,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
    )


if __name__ == "__main__":
    main()
