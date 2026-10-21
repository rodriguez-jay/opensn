// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/io/discrete_ordinates_problem_io.h"
#include "framework/runtime.h"
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace opensn
{

/// Default per-rank send budget, in bytes, of one surface-data redistribution round.
const std::uint64_t SURFACE_REDISTRIBUTION_BYTES = std::uint64_t{64} << 20;

/// Quadrature of one groupset of a surface angular-flux file set.
struct SurfaceQuadrature
{
  std::uint64_t num_dirs = 0;
  std::uint64_t num_groups = 0;
  /// Direction vectors, three Cartesian components per direction.
  std::vector<double> omega;
  /// Quadrature weights, one per direction.
  std::vector<double> weights;
};

/**
 * The faces of one stored surface tag that are owned by this rank.
 *
 * A face is owned by rank `SurfaceFaceOwner(key, P)`, where `key` is its centroid key, so every
 * data set read on the same communicator places a given physical face, from either side of an
 * interior surface, on the same rank. Faces are stored in arrival order; `index` gives the local
 * face index of each key in key order.
 */
struct SurfaceShard
{
  struct GroupsetValues
  {
    /// Direction cosine with the face normal, ordered by face and direction.
    std::vector<double> mu;
    /// Angular flux, ordered by face, face node, direction, and group.
    std::vector<double> psi;
  };

  /// (face key, local face index) pairs sorted by key.
  std::vector<std::pair<DiscreteOrdinatesProblemIO::FaceCentroidKey, std::uint64_t>> index;
  /// Start offset of each face's nodes. Its size is the number of faces plus one.
  std::vector<std::uint64_t> node_offset = {0};
  /// Start offset of each face's mass matrix. Its size is the number of faces plus one.
  std::vector<std::uint64_t> mass_offset = {0};
  /// Face node coordinates, three per node.
  std::vector<double> coords;
  /// Face mass matrices, row-major by face node, including the curvilinear measure scale.
  std::vector<double> mass;
  /// Values per groupset.
  std::vector<GroupsetValues> groupsets;

  std::uint64_t GetNumFaces() const { return node_offset.size() - 1; }

  std::uint64_t GetNumFaceNodes(const std::uint64_t face) const
  {
    return node_offset[face + 1] - node_offset[face];
  }

  /// Return the local index of the face with `key`, if this rank owns it.
  std::optional<std::uint64_t> Find(const DiscreteOrdinatesProblemIO::FaceCentroidKey& key) const;
};

/// Surface angular-flux data distributed by face across the ranks of a communicator.
struct DistributedSurfaceData
{
  /// True when the data were written from an adjoint problem.
  bool adjoint = false;
  /// Number of files the data were read from.
  std::uint64_t num_partitions = 0;
  std::array<double, 3> centroid_origin = {0.0, 0.0, 0.0};
  double centroid_spacing = 0.0;
  /// Quadrature per groupset. Identical on all ranks.
  std::vector<SurfaceQuadrature> groupsets;
  /// Locally owned faces keyed by stored surface tag. Every rank holds every tag.
  std::map<std::string, SurfaceShard> surfaces;
};

/// The surface response of one boundary or interior surface.
struct SurfaceResponseTerm
{
  /// Boundary name or interior surface name (without the `_u`/`_d` suffix).
  std::string surface;
  bool interior = false;
  /// Contribution of directions with positive cosine with the `_u` (boundary: outward) normal.
  double upwind = 0.0;
  /// Contribution of directions with negative cosine with the `_u` normal. Zero for a boundary.
  double downwind = 0.0;
};

/// Rank that owns the face with `key` on a communicator with `num_ranks` ranks.
int SurfaceFaceOwner(const DiscreteOrdinatesProblemIO::FaceCentroidKey& key, int num_ranks);

/**
 * Read surface angular fluxes written by `DiscreteOrdinatesProblemIO::WriteSurfaceAngularFluxes`
 * and distribute their faces across the ranks of `comm`.
 *
 * The files `<file_base>0.h5`, ..., `<file_base>(N-1).h5` are found from the `num_partitions`
 * attribute of `<file_base>0.h5`. Every file must carry the `num_partitions`, `adjoint`,
 * `centroid_origin_*`, and `centroid_spacing` attributes written by the current
 * WriteSurfaceAngularFluxes; files from earlier versions of OpenSn are rejected. N need not equal
 * the size of `comm`, and the data need not match the partitioning of any current mesh. File k is
 * read by rank k modulo the size of `comm`. A planning pass reads each file's face-node counts and
 * node coordinates (32 bytes per node at most) to size the owner storage exactly. The angular
 * fluxes, direction cosines, and mass matrices are then read and sent to their owner ranks (see
 * SurfaceFaceOwner) in rounds of at most `max_send_bytes` packed bytes per rank, with at least one
 * face per round, so no rank holds a whole file's angular flux.
 *
 * \param file_base File name base.
 * \param surfaces Stored surface tags to read. Must be non-empty and unique. Boundary tags are
 *        their boundary names; interior surface tags end in `_u` or `_d`.
 * \param comm Communicator to distribute the faces over.
 * \param max_send_bytes Per-rank send budget of one redistribution round.
 *
 * Collective over `comm`. Throws `std::invalid_argument` for invalid arguments and
 * `std::logic_error`, on all ranks, for missing, inconsistent, or malformed files, for a tag with
 * no faces in any file, and for duplicate faces.
 */
DistributedSurfaceData
ReadDistributedSurfaceAngularFluxes(const std::string& file_base,
                                    const std::vector<std::string>& surfaces,
                                    const mpi::Communicator& comm,
                                    std::uint64_t max_send_bytes = SURFACE_REDISTRIBUTION_BYTES);

/**
 * Evaluate the surface inner products of forward and adjoint angular fluxes.
 *
 * For a boundary surface with outward normal n,
 *   R = sum_{d: mu_d > 0} mu_d w_d sum_ij M_ij sum_g psi(i, d, g) psi^dagger(j, d, g),
 * and for an interior surface, with n the `_u` normal (the positive axis) and mu_d its cosine,
 *   R_u = sum_{d: mu_d > 0} mu_d w_d sum_ij M_ij sum_g psi_u(i, d, g) psi^dagger_d(j, d, g),
 *   R_d = sum_{d: mu_d < 0} mu_d w_d sum_ij M_ij sum_g psi_d(i, d, g) psi^dagger_u(j, d, g).
 * Each trace is taken from its upwind side: the forward flux from the side it leaves, the adjoint
 * flux from the side it leaves in the reversed direction. Forward and adjoint directions are
 * paired by equal direction vectors. Face nodes are paired by location.
 *
 * \param forward Forward data. Must not be adjoint.
 * \param adjoint Adjoint data. Must be adjoint and must be distributed over `comm`.
 * \param surfaces Boundary names and interior surface names to evaluate. If empty, every surface in
 *        the forward data is evaluated. Interior surfaces are identified by tag pairs `X_u` and
 *        `X_d`; any other tag is a boundary.
 * \param comm Communicator both data sets were distributed over.
 * \return One term per surface, in name order when `surfaces` is empty and in the given order
 *         otherwise. Values are global sums.
 *
 * Collective over `comm`. Throws `std::invalid_argument` for incompatible data sets or unknown
 * surfaces and `std::logic_error`, on all ranks, if any face lacks a matching face in another
 * required tag or matching faces have different node locations.
 */
std::vector<SurfaceResponseTerm>
EvaluateSurfaceResponseTerms(const DistributedSurfaceData& forward,
                             const DistributedSurfaceData& adjoint,
                             const std::vector<std::string>& surfaces,
                             const mpi::Communicator& comm);

} // namespace opensn
