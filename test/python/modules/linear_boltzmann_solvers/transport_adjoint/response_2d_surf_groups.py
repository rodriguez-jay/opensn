#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2D transport response test for interior face groups in surface angular-flux export.

The test writes a gmsh 2.2 mesh of 20 x 20 quadrilaterals on [0, 10]^2 with line groups:

* "Wall": the line x = 6 from y = 0 to y = 10, with elements running in +y. It is an open group
  that separates the source [0, 1] x [0, 1] from the detector [9.5, 10] x [9.5, 10]. Its reference
  normal, the tangent rotated clockwise, is +x, so Wall_u holds the faces of the cells left of the
  line and must equal the planar interior surface ("x", 6). The surface response across it equals
  the forward QoI, as in response_2d_surf.py.
* "Shelf" (y = 3, x = 2 to 4, elements running in -x) and "ShelfRev" (y = 7, x = 2 to 4, elements
  running in +x): open segments whose _u normals must be +y and -y.
* "Corner": the line y = 0 from x = 0 to x = 2, on the exterior boundary only.
* "xmin", "xmax", "ymin", "ymax": the exterior boundaries.

The partition cuts the mesh at y = 5, so the Wall group lies on both ranks.
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
    from pyopensn.mesh import FromFileMeshGenerator, KBAGraphPartitioner
    from pyopensn.response import ResponseEvaluator
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.source import VolumetricSource
    from pyopensn.xs import MultiGroupXS
else:
    barrier = MPIBarrier


def write_mesh(file_name, n, length):
    """Writes the gmsh 2.2 mesh described in the module docstring."""
    h = length / n

    def node(i, j):
        return j * (n + 1) + i + 1

    def segments(points):
        return [(node(*points[k]), node(*points[k + 1])) for k in range(len(points) - 1)]

    line_groups = {
        "Wall": segments([(12, j) for j in range(n + 1)]),
        "Shelf": segments([(i, 6) for i in range(8, 3, -1)]),
        "ShelfRev": segments([(i, 14) for i in range(4, 9)]),
        "Corner": segments([(i, 0) for i in range(0, 5)]),
        "xmin": segments([(0, j) for j in range(n + 1)]),
        "xmax": segments([(n, j) for j in range(n + 1)]),
        "ymin": segments([(i, 0) for i in range(n + 1)]),
        "ymax": segments([(i, n) for i in range(n + 1)]),
    }
    # The exterior boundary elements of y = 0 for x < 2 belong to Corner, not ymin.
    corner = set(line_groups["Corner"])
    line_groups["ymin"] = [s for s in line_groups["ymin"] if s not in corner]

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames"]
    lines.append(str(len(line_groups) + 1))
    for g, name in enumerate(line_groups):
        lines.append(f'1 {g + 10} "{name}"')
    lines += ['2 1 "Domain"', "$EndPhysicalNames", "$Nodes", str((n + 1) ** 2)]
    for j in range(n + 1):
        for i in range(n + 1):
            lines.append(f"{node(i, j)} {i * h} {j * h} 0")
    elements = []
    for g, name in enumerate(line_groups):
        for a, b in line_groups[name]:
            elements.append(f"1 2 {g + 10} {g + 10} {a} {b}")
    for j in range(n):
        for i in range(n):
            elements.append(
                f"3 2 1 1 {node(i, j)} {node(i + 1, j)} {node(i + 1, j + 1)} {node(i, j + 1)}"
            )
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    lines += [f"{k + 1} {element}" for k, element in enumerate(elements)]
    lines.append("$EndElements")
    with open(file_name, "w") as output:
        output.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    num_procs = 2
    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} processors but got {size}.")

    num_cells = 20
    length = 10.0
    mesh_file = "response_2d_surf_groups.msh"
    if rank == 0:
        write_mesh(mesh_file, num_cells, length)
    barrier()

    meshgen = FromFileMeshGenerator(
        filename=mesh_file,
        partitioner=KBAGraphPartitioner(nx=1, ny=2, nz=1, ycuts=[length / 2.0]),
    )
    grid = meshgen.Execute()
    barrier()
    if rank == 0:
        os.remove(mesh_file)
    grid.SetUniformBlockID(0)

    source_volume = RPPLogicalVolume(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, infz=True)
    grid.SetBlockIDFromLogicalVolume(source_volume, 1, True)
    detector_volume = RPPLogicalVolume(xmin=9.5, xmax=10.0, ymin=9.5, ymax=10.0, infz=True)
    grid.SetBlockIDFromLogicalVolume(detector_volume, 2, True)

    xs_background = MultiGroupXS()
    xs_background.CreateSimpleOneGroup(0.1, 0.9)
    xs_source = MultiGroupXS()
    xs_source.CreateSimpleOneGroup(2.0, 1.0)
    detector_absorption = 0.8
    xs_detector = MultiGroupXS()
    xs_detector.CreateSimpleOneGroup(detector_absorption, 0.0)

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
        xs_map=[
            {"block_ids": [0], "xs": xs_background},
            {"block_ids": [1], "xs": xs_source},
            {"block_ids": [2], "xs": xs_detector},
        ],
        boundary_conditions=[
            {"name": name, "type": "vacuum"} for name in ("xmin", "xmax", "ymin", "ymax", "Corner")
        ],
        options={"save_angular_flux": True},
    )

    source_strength = 1.0
    problem.SetVolumetricSources(
        volumetric_sources=[VolumetricSource(block_ids=[1], group_strength=[source_strength])]
    )
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

    # Invalid requests are rejected before any file is created.
    invalid_prefix = "InvalidGroups_p"
    invalid_cases = [
        ({"interior_face_groups": ["Missing"]}, "is not a physical group"),
        ({"interior_face_groups": ["Corner"]}, "has no faces inside the mesh"),
        (
            {"interior_face_groups": ["Wall"], "interior_surfaces": {"x6": ("x", 6.0)}},
            "shares a face",
        ),
        ({"interior_face_groups": ["Wall", "Wall"]}, "collides with a requested surface"),
        (
            {"interior_face_groups": ["Wall"], "interior_surfaces": {"Wall_u": ("y", 2.0)}},
            "collides with a requested surface",
        ),
        # A group without exterior faces is not a boundary of the grid.
        (
            {"interior_face_groups": ["Shelf"], "boundary_surfaces": ["Wall"]},
            "not found on grid",
        ),
        (
            {"interior_face_groups": ["Wall"] if rank == 0 else []},
            "Every rank must request the same",
        ),
    ]
    for arguments, expected_error in invalid_cases:
        rejected = False
        try:
            problem.WriteSurfaceAngularFluxes(invalid_prefix, **arguments)
        except ValueError as error:
            rejected = expected_error in str(error)
            if not rejected:
                raise
        if not rejected:
            raise RuntimeError(f"Invalid interior face group request {arguments} was not rejected")
        if os.path.exists(f"{invalid_prefix}{rank}.h5"):
            raise RuntimeError("Rejected interior face group request created an output file")

    groups_prefix = "Groups_Fwd_p"
    plane_prefix = "Plane_Fwd_p"
    problem.WriteSurfaceAngularFluxes(
        groups_prefix, interior_face_groups=["Wall", "Shelf", "ShelfRev"]
    )
    problem.WriteSurfaceAngularFluxes(plane_prefix, interior_surfaces={"x6": ("x", 6.0)})
    # Every rank reads every file below, so all files must be complete.
    barrier()

    def read_faces(prefix, tag):
        """Face centroids and normals of a tag from every file of a set."""
        faces = []
        for partition in range(size):
            (surface,) = problem.ReadSurfaceAngularFluxes(prefix, [tag], partition=partition)
            mapping = surface["mapping"]
            start = 0
            for face, num_nodes in enumerate(mapping["num_face_nodes"]):
                cx = sum(mapping["nodes_x"][start : start + num_nodes]) / num_nodes
                cy = sum(mapping["nodes_y"][start : start + num_nodes]) / num_nodes
                start += num_nodes
                normal = tuple(round(c, 12) for c in mapping["normals"][3 * face : 3 * face + 3])
                faces.append((round(cx, 12), round(cy, 12), normal))
        return sorted(faces)

    # The open Wall group selects the same faces, with the same normals, as the plane x = 6.
    for side in ("_u", "_d"):
        if read_faces(groups_prefix, "Wall" + side) != read_faces(plane_prefix, "x6" + side):
            raise RuntimeError(f"Wall{side} differs from the plane x = 6")
    wall_faces = len(read_faces(groups_prefix, "Wall_u"))
    local_wall = problem.ReadSurfaceAngularFluxes(groups_prefix, ["Wall_u"])[0]
    if not local_wall["mapping"]["num_face_nodes"]:
        raise RuntimeError("The partition does not split the Wall group across ranks")

    # The element order sets the reference normal of open segments.
    expected_normals = {
        "Shelf_u": (0.0, 1.0, 0.0),
        "Shelf_d": (0.0, -1.0, 0.0),
        "ShelfRev_u": (0.0, -1.0, 0.0),
        "ShelfRev_d": (0.0, 1.0, 0.0),
    }
    shelf_faces = 0
    for tag, normal in expected_normals.items():
        faces = read_faces(groups_prefix, tag)
        shelf_faces += len(faces)
        if len(faces) != 4 or any(face[2] != normal for face in faces):
            raise RuntimeError(f"{tag} has faces {faces}, expected 4 faces with normal {normal}")

    # Adjoint solve and response across the Wall group.
    problem.SetAdjoint(True)
    problem.SetVolumetricSources(
        volumetric_sources=[
            VolumetricSource(logical_volume=detector_volume, group_strength=[detector_absorption])
        ]
    )
    solver.Execute()
    adjoint_prefix = "Groups_Adj_p"
    problem.WriteSurfaceAngularFluxes(adjoint_prefix, interior_face_groups=["Wall"])

    evaluator = ResponseEvaluator(problem=problem)
    evaluator.SetOptions(
        data=[
            {
                "name": "forward",
                "file_prefixes": {"surface_angular_fluxes": groups_prefix},
                "surfaces": ["Wall"],
            },
            {
                "name": "adjoint",
                "file_prefixes": {"surface_angular_fluxes": adjoint_prefix},
                "surfaces": ["Wall"],
            },
        ]
    )
    wall = evaluator.EvaluateSurfaceResponse("forward", "adjoint", surfaces=["Wall"])["Wall"]

    if rank == 0:
        print(f"Wall faces={wall_faces}")
        print(f"Shelf faces={shelf_faces}")
        print(f"Forward QoI={forward_qoi:.8e}")
        print(f"Wall surface QoI={wall['total']:.8e}")
        print(f"Wall identity error={abs(wall['total'] / forward_qoi - 1.0):.3e}")
        print("INTERIOR_FACE_GROUP_CHECKS_PASSED=1")

    barrier()
    for prefix in (groups_prefix, plane_prefix, adjoint_prefix):
        os.remove(f"{prefix}{rank}.h5")
