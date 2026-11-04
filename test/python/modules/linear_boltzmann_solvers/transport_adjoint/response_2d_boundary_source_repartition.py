#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D adjoint response to an isotropic boundary source, from adjoint data written on another partition.

This is the problem of response_2d_boundary_source.py. By forward-adjoint duality, the detector
response to an isotropic incoming flux on xmin,

    QoI = int_V sigma_d phi dV,

equals the adjoint evaluation

    QoI = int_{xmin} int_{Omega . n < 0} |Omega . n| psi_b psi^dagger(r, Omega) dOmega dA.

The forward problem is partitioned in x and the adjoint problem in y, so the adjoint flux moments
and angular fluxes are written by ranks that own other cells. The ResponseEvaluator of the forward
problem reads them: each cell's values must reach the rank that owns it on the forward partition,
where the boundary term reads the adjoint angular flux of its boundary cells. The adjoint response
must equal the forward QoI, the response_2d_boundary_source.py gold value.
"""

import os
import sys

if "opensn_console" not in globals():
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    barrier = MPI.COMM_WORLD.Barrier
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


if __name__ == "__main__":
    if size not in (2, 4):
        sys.exit(f"Expected 2 or 4 processors but got {size}.")

    nodes = [i * 0.5 for i in range(21)]
    cuts = [5.0] if size == 2 else [2.5, 5.0, 7.5]
    detector_volume = RPPLogicalVolume(xmin=9.0, xmax=10.0, ymin=4.0, ymax=6.0, infz=True)

    def make_grid(partitioner):
        grid = OrthogonalMeshGenerator(node_sets=[nodes, nodes], partitioner=partitioner).Execute()
        grid.SetOrthogonalBoundaries()
        grid.SetUniformBlockID(0)
        grid.SetBlockIDFromLogicalVolume(detector_volume, 1, True)
        return grid

    forward_grid = make_grid(KBAGraphPartitioner(nx=size, ny=1, nz=1, xcuts=cuts))
    adjoint_grid = make_grid(KBAGraphPartitioner(nx=1, ny=size, nz=1, ycuts=cuts))

    xs_background = MultiGroupXS()
    xs_background.CreateSimpleOneGroup(0.3, 0.5)
    xs_detector = MultiGroupXS()
    xs_detector.CreateSimpleOneGroup(0.8, 0.0)
    detector_sigma = 0.8
    boundary_strength = 1.0

    quadrature = GLCProductQuadrature2DXY(n_polar=4, n_azimuthal=16, scattering_order=0)

    def boundary_conditions(with_source):
        bcs = [{"name": name, "type": "vacuum"} for name in ("xmax", "ymin", "ymax")]
        if with_source:
            bcs.append(
                {"name": "xmin", "type": "isotropic", "group_strength": [boundary_strength]}
            )
        else:
            bcs.append({"name": "xmin", "type": "vacuum"})
        return bcs

    def make_problem(grid, with_source):
        return DiscreteOrdinatesProblem(
            mesh=grid,
            num_groups=1,
            groupsets=[
                {
                    "groups_from_to": (0, 0),
                    "angular_quadrature": quadrature,
                    "inner_linear_method": "petsc_gmres",
                    "l_abs_tol": 1.0e-12,
                    "l_max_its": 500,
                }
            ],
            xs_map=[
                {"block_ids": [0], "xs": xs_background},
                {"block_ids": [1], "xs": xs_detector},
            ],
            boundary_conditions=boundary_conditions(with_source),
            options={"save_angular_flux": True},
        )

    # Forward solve and volumetric detector response on the x-cut partition.
    forward_problem = make_problem(forward_grid, True)
    forward_solver = SteadyStateSourceSolver(problem=forward_problem)
    forward_solver.Initialize()
    forward_solver.Execute()
    interpolator = FieldFunctionInterpolationVolume()
    interpolator.SetOperationType("sum")
    interpolator.SetLogicalVolume(detector_volume)
    interpolator.AddFieldFunction(forward_problem.GetScalarFluxFieldFunction()[0])
    interpolator.Execute()
    forward_qoi = detector_sigma * interpolator.GetValue()

    # Adjoint solve on the y-cut partition, written there.
    adjoint_problem = make_problem(adjoint_grid, False)
    adjoint_problem.SetAdjoint(True)
    adjoint_problem.SetVolumetricSources(
        volumetric_sources=[
            VolumetricSource(logical_volume=detector_volume, group_strength=[detector_sigma])
        ]
    )
    adjoint_solver = SteadyStateSourceSolver(problem=adjoint_problem)
    adjoint_solver.Initialize()
    adjoint_solver.Execute()
    adjoint_phi_prefix = f"BndrySrcRepartition{size}AdjPhi_p"
    adjoint_psi_prefix = f"BndrySrcRepartition{size}AdjPsi_p"
    adjoint_problem.WriteFluxMoments(adjoint_phi_prefix)
    adjoint_problem.WriteAngularFluxes(adjoint_psi_prefix)
    barrier()

    # The boundary-source response on the forward partition, from the adjoint files.
    evaluator = ResponseEvaluator(problem=forward_problem)
    evaluator.SetOptions(
        data=[
            {
                "name": "detector",
                "file_prefixes": {
                    "flux_moments": adjoint_phi_prefix,
                    "angular_fluxes": adjoint_psi_prefix,
                },
            }
        ],
        sources={
            "boundary": [
                {"name": "xmin", "type": "isotropic", "group_strength": [boundary_strength]}
            ]
        },
    )
    adjoint_qoi = evaluator.EvaluateResponse("detector")

    if rank == 0:
        print(f"Forward QoI={forward_qoi:.8e}")
        print(f"Cross-partition boundary QoI={adjoint_qoi:.8e}")
        # Relative error of the duality identity.
        print(f"Cross-partition boundary identity error={abs(adjoint_qoi / forward_qoi - 1.0):.3e}")

    barrier()
    for prefix in (adjoint_phi_prefix, adjoint_psi_prefix):
        os.remove(f"{prefix}{rank}.h5")
