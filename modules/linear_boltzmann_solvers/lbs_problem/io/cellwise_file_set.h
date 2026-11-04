// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/lbs_problem/io/lbs_problem_io.h"
#include "framework/runtime.h"
#include "hdf5.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace opensn
{

class Cell;
class Mesh;
class SpatialDiscretization;

/// A dataset of a cell-wise file holding `stride` values per file node, ordered by cell and node.
struct CellwiseDataset
{
  std::string name;
  std::uint64_t stride = 0;
};

/// Default bound on the packed bytes each rank sends per exchange while reading a cell-wise set.
const std::uint64_t cellwise_redistribution_bytes = std::uint64_t{64} << 20;

/**
 * Check the solution provenance of a flux-moment or angular-flux file: the `solution_role` and
 * `adjoint_orientation` attributes must be both present or both absent, valid, and consistent; they
 * must match `expected_role` and `expected_orientation` unless these are unspecified; and when the
 * file replaces the state of a problem, `problem_is_adjoint` is the problem's mode, a recorded role
 * must match it, and an adjoint must be in the solver orientation. Throws std::logic_error on the
 * calling rank.
 */
void CheckSolutionProvenance(hid_t file,
                             const std::string& file_name,
                             FluxSolutionRole expected_role,
                             AdjointFluxOrientation expected_orientation,
                             std::optional<bool> problem_is_adjoint);

/**
 * Read a set of cell-wise files, such as those written by LBSSolverIO::WriteFluxMoments and
 * DiscreteOrdinatesProblemIO::WriteAngularFluxes, onto the cells of `grid` that this rank owns.
 *
 * The set is `<file_base>0.h5`, ..., `<file_base>(N-1).h5`, or the single file `<file_base>.h5` if
 * `single_file` is true. Each file stores the global ids, node counts, and node locations of its
 * cells in its "mesh" group, and each dataset of `datasets` stores `stride` values per node in
 * that order. N is the `num_partitions` attribute of the files. Files written before it was
 * recorded form a set of as many files as `comm` has ranks.
 *
 * The files may have been written by any number of ranks with any partitioning of the same mesh.
 * File k is read by rank k mod P, one file at a time, and each cell's node locations and values
 * are sent to the rank that owns the cell, found with an OwnerDirectory, in exchanges of at most
 * `max_send_bytes` bytes per rank. A rank therefore holds one file, one of its datasets, and one
 * exchange at a time, in addition to its result.
 *
 * Every rank calls `validate_file` on each file it opens, before its data are read, and calls
 * `scatter(dataset, cell, node, values)` for every node of every local cell: `values` are the
 * `stride` values of dataset `dataset` at local node `node` of `cell`. File nodes are matched to
 * local nodes by location, so their order within a cell may differ.
 *
 * Collective over `comm`. Throws std::runtime_error on every rank, with `operation` and the lowest
 * failing rank in its message, if a file is missing, malformed, inconsistent with the set, contains
 * non-finite values, or fails `validate_file`; if a file cell is not a cell of `grid`, appears in
 * more than one file, or has node locations that differ from the mesh; or if a local cell is in no
 * file. A failure may leave `scatter` called for some nodes, so callers scatter into a temporary.
 */
void ReadCellwiseFileSet(
  const std::string& file_base,
  bool single_file,
  const std::vector<CellwiseDataset>& datasets,
  const Mesh& grid,
  const SpatialDiscretization& discretization,
  const std::function<void(hid_t file, const std::string& file_name)>& validate_file,
  const std::function<
    void(std::size_t dataset, const Cell& cell, std::size_t node, const double* values)>& scatter,
  const mpi::Communicator& comm,
  const std::string& operation,
  std::uint64_t max_send_bytes = cellwise_redistribution_bytes);

} // namespace opensn
