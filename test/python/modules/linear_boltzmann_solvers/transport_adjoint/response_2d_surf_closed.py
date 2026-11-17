#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D transport response test for closed surfaces in surface angular-flux export.

The closed surface "det" is the boundary of the cells whose centroids lie inside the box
[5.5, 8.5] x [3.5, 7.5]. It encloses the detector [6, 8] x [4, 7] and excludes the source
[0, 1] x [0, 1], and all exterior boundaries are vacuum, so the response is carried entirely
across the surface:

    QoI = int_S int_4pi (Omega . n) psi(r, Omega) psi^dagger(r, Omega) dOmega dA,

where n points from the source side into the enclosed region. That is the normal of the det_d
tag, written from the outside cells. The det_u tag, written from the inside cells, has the outward
normal, so the response measured along it must equal minus the QoI. For the upwind DFEM
discretization both identities are exact, as for the planar interior surface in
response_2d_surf.py.

The partition cuts the box at y = 5, so faces of the surface lie on both ranks and their inside or
outside cells are ghost cells on the rank that owns the other side.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    barrier = MPI.COMM_WORLD.Barrier
    global_sum = MPI.COMM_WORLD.allreduce
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.mesh import KBAGraphPartitioner, OrthogonalMeshGenerator
    from pyopensn.response import ResponseEvaluator
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.source import VolumetricSource
    from pyopensn.xs import MultiGroupXS
else:
    barrier = MPIBarrier
    global_sum = MPIAllReduce


if __name__ == "__main__":
    num_procs = 2
    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} processors but got {size}.")

    # Create mesh and assign material IDs.
    num_cells = 20
    length = 10.0
    cell_size = length / num_cells
    nodes = [i * cell_size for i in range(num_cells + 1)]
    meshgen = OrthogonalMeshGenerator(
        node_sets=[nodes, nodes],
        partitioner=KBAGraphPartitioner(nx=1, ny=2, nz=1, ycuts=[length / 2.0]),
    )
    grid = meshgen.Execute()
    grid.SetOrthogonalBoundaries()
    grid.SetUniformBlockID(0)

    source_volume = RPPLogicalVolume(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, infz=True)
    grid.SetBlockIDFromLogicalVolume(source_volume, 1, True)

    detector_volume = RPPLogicalVolume(xmin=6.0, xmax=8.0, ymin=4.0, ymax=7.0, infz=True)
    grid.SetBlockIDFromLogicalVolume(detector_volume, 2, True)

    box = (5.5, 8.5, 3.5, 7.5)
    box_center = (0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3]))
    closed_volume = RPPLogicalVolume(
        xmin=box[0], xmax=box[1], ymin=box[2], ymax=box[3], infz=True
    )
    # The box edges lie on cell faces, so the surface is the box boundary.
    expected_faces = int(round(2 * ((box[1] - box[0]) + (box[3] - box[2])) / cell_size))

    # Define cross sections.
    xs_background = MultiGroupXS()
    xs_background.CreateSimpleOneGroup(0.1, 0.9)

    xs_source = MultiGroupXS()
    xs_source.CreateSimpleOneGroup(2.0, 1.0)

    detector_absorption = 0.8
    xs_detector = MultiGroupXS()
    xs_detector.CreateSimpleOneGroup(detector_absorption, 0.0)

    xs_map = [
        {"block_ids": [0], "xs": xs_background},
        {"block_ids": [1], "xs": xs_source},
        {"block_ids": [2], "xs": xs_detector},
    ]
    source_block_id = 1

    quadrature = GLCProductQuadrature2DXY(n_polar=2, n_azimuthal=32, scattering_order=0)
    problem = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=1,
        groupsets=[
            {
                "groups_from_to": (0, 0),
                "angular_quadrature": quadrature,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-12,
                "l_max_its": 500,
                "gmres_restart_interval": 100,
            },
        ],
        xs_map=xs_map,
        boundary_conditions=[
            {"name": "xmin", "type": "vacuum"},
            {"name": "xmax", "type": "vacuum"},
            {"name": "ymin", "type": "vacuum"},
            {"name": "ymax", "type": "vacuum"},
        ],
        options={"save_angular_flux": True},
    )

    source_strength = 1.0
    source_area = 1.0
    forward_source = VolumetricSource(
        block_ids=[source_block_id], group_strength=[source_strength / source_area]
    )
    problem.SetVolumetricSources(volumetric_sources=[forward_source])

    solver = SteadyStateSourceSolver(problem=problem)
    solver.Initialize()
    solver.Execute()

    field_functions = problem.GetScalarFluxFieldFunction()
    interpolator = FieldFunctionInterpolationVolume()
    interpolator.SetOperationType("sum")
    interpolator.SetLogicalVolume(detector_volume)
    interpolator.AddFieldFunction(field_functions[0])
    interpolator.Execute()
    forward_qoi = detector_absorption * interpolator.GetValue()

    # Invalid closed-surface requests are rejected before any file is created.
    invalid_prefix = "InvalidClosed_p"
    reaches_boundary = RPPLogicalVolume(xmin=-1.0, xmax=2.0, ymin=3.0, ymax=5.0, infz=True)
    # Cell centroids lie at odd multiples of 0.25; this box contains none of them.
    no_cells = RPPLogicalVolume(xmin=3.3, xmax=3.4, ymin=3.3, ymax=3.4, infz=True)
    invalid_closed_cases = [
        ({"closed_surfaces": {"edge": reaches_boundary}}, "reaches an exterior boundary"),
        ({"closed_surfaces": {"empty": no_cells}}, "matches no mesh faces"),
        (
            {"closed_surfaces": {"det": closed_volume}, "interior_surfaces": {"x": ("x", 5.5)}},
            "shares a face",
        ),
        (
            {"closed_surfaces": {"det": closed_volume, "twin": closed_volume}},
            "shares a face",
        ),
        (
            {"boundary_surfaces": ["xmin"], "closed_surfaces": {"xmin": closed_volume}},
            "collides with a requested surface name",
        ),
        (
            {"interior_surfaces": {"det_u": ("x", 3.0)}, "closed_surfaces": {"det": closed_volume}},
            "collides with a requested surface name",
        ),
        (
            {"closed_surfaces": {"det": closed_volume} if rank == 0 else {}},
            "Every rank must request the same closed surfaces",
        ),
    ]
    for arguments, expected_error in invalid_closed_cases:
        rejected = False
        try:
            problem.WriteSurfaceAngularFluxes(invalid_prefix, **arguments)
        except ValueError as error:
            rejected = expected_error in str(error)
            if not rejected:
                raise
        if not rejected:
            raise RuntimeError(f"Invalid closed surface {arguments} was not rejected")
        if os.path.exists(f"{invalid_prefix}{rank}.h5"):
            raise RuntimeError("Rejected closed surface created an output file")

    forward_prefix = "ClosedSurf_Fwd_p"
    problem.WriteSurfaceAngularFluxes(
        forward_prefix,
        boundary_surfaces=["xmin", "xmax", "ymin", "ymax"],
        closed_surfaces={"det": closed_volume},
    )

    # Geometry of the exported surface. Read every file of the set on each rank.
    def read_all_partitions(prefix, tag):
        return [
            problem.ReadSurfaceAngularFluxes(prefix, [tag], partition=p)[0] for p in range(size)
        ]

    local_faces = {}
    for tag, outward in (("det_u", True), ("det_d", False)):
        surfaces = read_all_partitions(forward_prefix, tag)
        num_faces = 0
        for surface in surfaces:
            mapping = surface["mapping"]
            start = 0
            for face, num_nodes in enumerate(mapping["num_face_nodes"]):
                cx = sum(mapping["nodes_x"][start : start + num_nodes]) / num_nodes
                cy = sum(mapping["nodes_y"][start : start + num_nodes]) / num_nodes
                start += num_nodes
                normal = mapping["normals"][3 * face : 3 * face + 3]
                on_box = (
                    min(abs(cx - box[0]), abs(cx - box[1])) < 1e-12
                    or min(abs(cy - box[2]), abs(cy - box[3])) < 1e-12
                )
                if not on_box:
                    raise RuntimeError(f"{tag} face at ({cx}, {cy}) is not on the box boundary")
                radial = normal[0] * (cx - box_center[0]) + normal[1] * (cy - box_center[1])
                if (radial > 0.0) != outward:
                    raise RuntimeError(f"{tag} face at ({cx}, {cy}) has normal {normal}")
            num_faces += len(mapping["num_face_nodes"])
        if num_faces != expected_faces:
            raise RuntimeError(f"{tag} has {num_faces} faces, expected {expected_faces}")
        local_faces[tag] = len(surfaces[rank]["mapping"]["num_face_nodes"])
    if local_faces["det_u"] == 0 or local_faces["det_d"] == 0:
        raise RuntimeError("The partition does not split the closed surface across ranks")

    # Adjoint solve and response evaluation.
    problem.SetAdjoint(True)
    adjoint_source = VolumetricSource(
        logical_volume=detector_volume, group_strength=[detector_absorption]
    )
    problem.SetVolumetricSources(volumetric_sources=[adjoint_source])
    solver.Execute()

    adjoint_flux_prefix = "ClosedSurf_AdjFluxMoments_p"
    adjoint_prefix = "ClosedSurf_Adj_p"
    problem.WriteFluxMoments(adjoint_flux_prefix)
    problem.WriteSurfaceAngularFluxes(adjoint_prefix, closed_surfaces={"det": closed_volume})

    evaluator = ResponseEvaluator(problem=problem)
    evaluator.SetOptions(
        data=[
            {"name": "detector", "file_prefixes": {"flux_moments": adjoint_flux_prefix}},
            {
                "name": "forward_closed",
                "file_prefixes": {"surface_angular_fluxes": forward_prefix},
                "surfaces": ["det"],
            },
            {
                "name": "adjoint_closed",
                "file_prefixes": {"surface_angular_fluxes": adjoint_prefix},
                "surfaces": ["det"],
            },
        ],
        sources={
            "material": [
                {"block_id": source_block_id, "strength": [source_strength / source_area]}
            ]
        },
    )
    adjoint_qoi = evaluator.EvaluateResponse("detector")

    inward = evaluator.EvaluateSurfaceResponse(
        "forward_closed", "adjoint_closed", surfaces=["det_d"]
    )["det_d"]
    outward = evaluator.EvaluateSurfaceResponse(
        "forward_closed", "adjoint_closed", surfaces=["det"]
    )["det"]
    if (
        outward["outgoing"] != -inward["incoming"]
        or outward["incoming"] != -inward["outgoing"]
        or outward["total"] != outward["outgoing"] + outward["incoming"]
    ):
        raise RuntimeError("Inconsistent outgoing and incoming parts of det")

    if rank == 0:
        print(f"Closed faces={expected_faces}")
        print(f"Forward QoI={forward_qoi:.8e}")
        print(f"Adjoint QoI={adjoint_qoi:.8e}")
        print(f"Inward surface QoI={inward['total']:.8e}")
        print(f"Adjoint identity error={abs(adjoint_qoi / forward_qoi - 1.0):.3e}")
        # Along the inward normal the surface QoI equals the forward QoI.
        print(f"Inward identity error={abs(inward['total'] / forward_qoi - 1.0):.3e}")
        # Along the outward normal it equals minus the forward QoI.
        print(f"Outward identity error={abs(outward['total'] / forward_qoi + 1.0):.3e}")
        print("CLOSED_SURFACE_CHECKS_PASSED=1")

    barrier()
    for file_prefix in (forward_prefix, adjoint_flux_prefix, adjoint_prefix):
        os.remove(f"{file_prefix}{rank}.h5")
