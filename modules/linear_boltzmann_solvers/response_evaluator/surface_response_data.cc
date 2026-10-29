// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/response_evaluator/surface_response_data.h"
#include "framework/logging/log.h"
#include "framework/utils/error.h"
#include "framework/utils/hdf_utils.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace opensn
{

namespace
{

using FaceCentroidKey = DiscreteOrdinatesProblemIO::FaceCentroidKey;

/// Tolerance for comparing quadrature values and direction cosines written by one problem.
const double same_problem_tolerance = 1.0e-12;

/**
 * Largest forward-to-adjoint direction distance, relative to the distance from the adjoint
 * direction to its nearest neighbor, at which the two are taken as the same direction. This is the
 * angular analog of matching face nodes within 1e-6 times the face size.
 */
const double direction_match_tolerance = 1.0e-6;

/// Throw on every rank if any rank recorded an error. Collective.
void
ThrowIfAnyRankFailed(const mpi::Communicator& comm,
                     const std::string& local_error,
                     const std::string& operation)
{
  int failed = local_error.empty() ? 0 : 1;
  comm.all_reduce(failed, mpi::op::max<int>());
  if (failed == 0)
    return;
  if (local_error.empty())
    throw std::logic_error(operation + " failed on another rank.");
  throw std::logic_error(operation + " failed on rank " + std::to_string(comm.rank()) + ": " +
                         local_error);
}

template <typename T>
void
BroadcastVector(const mpi::Communicator& comm, std::vector<T>& values, const int root)
{
  std::uint64_t size = values.size();
  comm.broadcast(size, root);
  values.resize(size);
  if (size > 0)
    comm.broadcast(values.data(), static_cast<int>(size), root);
}

std::uint64_t
CheckedMultiply(const std::uint64_t a, const std::uint64_t b, const std::string& context)
{
  OpenSnLogicalErrorIf(b != 0 and a > std::numeric_limits<std::uint64_t>::max() / b,
                       "Size overflow for " + context + ".");
  return a * b;
}

std::string
PartitionFileName(const std::string& file_base, const std::uint64_t id)
{
  return file_base + std::to_string(id) + ".h5";
}

/// Attributes of one surface angular-flux file.
struct FileMetadata
{
  std::uint64_t num_partitions = 0;
  bool adjoint = false;
  std::array<double, 3> centroid_origin = {0.0, 0.0, 0.0};
  double centroid_spacing = 0.0;
  std::vector<std::uint64_t> num_dirs;
  std::vector<std::uint64_t> num_groups;
};

FileMetadata
ReadFileMetadata(const hid_t file, const std::string& file_name)
{
  FileMetadata meta_data;
  std::uint64_t num_groupsets = 0;
  OpenSnLogicalErrorIf(not H5ReadAttribute(file, "num_groupsets", num_groupsets) or
                         num_groupsets == 0,
                       "Failed to read a positive number of groupsets from " + file_name + ".");
  // Every attribute below is written by the current WriteSurfaceAngularFluxes. Files without them
  // were written by an unsupported version of OpenSn.
  OpenSnLogicalErrorIf(not H5ReadAttribute(file, "num_partitions", meta_data.num_partitions) or
                         meta_data.num_partitions == 0,
                       "Failed to read a positive number of partitions from " + file_name +
                         ". The file was written by an unsupported version of OpenSn.");
  OpenSnLogicalErrorIf(not H5ReadAttribute(file, "adjoint", meta_data.adjoint),
                       "Failed to read the adjoint flag from " + file_name + ".");
  for (size_t d = 0; d < 3; ++d)
    OpenSnLogicalErrorIf(not H5ReadAttribute(file,
                                             "centroid_origin_" + std::to_string(d),
                                             meta_data.centroid_origin[d]),
                         "Failed to read the centroid origin from " + file_name + ".");
  OpenSnLogicalErrorIf(not H5ReadAttribute(file, "centroid_spacing", meta_data.centroid_spacing),
                       "Failed to read the centroid spacing from " + file_name + ".");
  OpenSnLogicalErrorIf(not(meta_data.centroid_spacing > 0.0) or
                         not std::isfinite(meta_data.centroid_spacing),
                       "Invalid centroid spacing in " + file_name + ".");

  // Read one groupset at a time so a corrupt count fails at the first missing attribute rather
  // than in an allocation.
  for (std::uint64_t gs = 0; gs < num_groupsets; ++gs)
  {
    const std::string group_name = "groupset_" + std::to_string(gs);
    std::uint64_t num_dirs = 0;
    std::uint64_t num_groups = 0;
    OpenSnLogicalErrorIf(
      not H5ReadAttribute(file, group_name + "/num_directions", num_dirs) or num_dirs == 0,
      "Failed to read a positive direction count for " + group_name + " from " + file_name + ".");
    OpenSnLogicalErrorIf(
      not H5ReadAttribute(file, group_name + "/num_groups", num_groups) or num_groups == 0,
      "Failed to read a positive group count for " + group_name + " from " + file_name + ".");
    meta_data.num_dirs.push_back(num_dirs);
    meta_data.num_groups.push_back(num_groups);
  }
  return meta_data;
}

/// Return a description of how `meta_data` differs from `reference_data`, or an empty string.
std::string
CompareFileMetadata(const FileMetadata& reference_data, const FileMetadata& meta_data)
{
  if (meta_data.num_partitions != reference_data.num_partitions)
    return "it records " + std::to_string(meta_data.num_partitions) +
           " partitions, but partition 0 "
           "records " +
           std::to_string(reference_data.num_partitions);
  if (meta_data.adjoint != reference_data.adjoint)
    return "its adjoint flag differs from partition 0";
  if (meta_data.centroid_origin != reference_data.centroid_origin or
      meta_data.centroid_spacing != reference_data.centroid_spacing)
    return "its centroid origin or spacing differs from partition 0";
  if (meta_data.num_dirs != reference_data.num_dirs or
      meta_data.num_groups != reference_data.num_groups)
    return "its groupset, direction, or group counts differ from partition 0";
  return {};
}

/// Broadcast `meta_data` from rank 0 to every rank of `comm`. Collective.
void
BroadcastFileMetadata(const mpi::Communicator& comm, FileMetadata& meta_data)
{
  std::vector<std::uint64_t> ints;
  std::vector<double> reals;
  if (comm.rank() == 0)
  {
    ints = {meta_data.num_partitions, meta_data.adjoint ? 1u : 0u};
    ints.insert(ints.end(), meta_data.num_dirs.begin(), meta_data.num_dirs.end());
    ints.insert(ints.end(), meta_data.num_groups.begin(), meta_data.num_groups.end());
    reals = {meta_data.centroid_origin[0],
             meta_data.centroid_origin[1],
             meta_data.centroid_origin[2],
             meta_data.centroid_spacing};
  }
  BroadcastVector(comm, ints, 0);
  BroadcastVector(comm, reals, 0);

  meta_data.num_partitions = ints[0];
  meta_data.adjoint = ints[1] != 0;
  const auto num_groupsets = static_cast<std::ptrdiff_t>((ints.size() - 2) / 2);
  meta_data.num_dirs.assign(ints.begin() + 2, ints.begin() + 2 + num_groupsets);
  meta_data.num_groups.assign(ints.begin() + 2 + num_groupsets, ints.end());
  meta_data.centroid_origin = {reals[0], reals[1], reals[2]};
  meta_data.centroid_spacing = reals[3];
}

/**
 * Sizes of one surface tag in one file. Face f has nodes [node_offset[f], node_offset[f + 1]) and
 * mass-matrix entries [mass_offset[f], mass_offset[f + 1]).
 */
struct TagLayout
{
  std::vector<std::uint64_t> node_offset = {0};
  std::vector<std::uint64_t> mass_offset = {0};

  std::uint64_t GetNumFaces() const { return node_offset.size() - 1; }

  std::uint64_t GetNumNodes() const { return node_offset.back(); }

  std::uint64_t GetNumFaceNodes(const std::uint64_t face) const
  {
    return node_offset[face + 1] - node_offset[face];
  }
};

/// Read the face-node counts of `tag` and check every dataset size against them.
TagLayout
ReadTagLayout(const hid_t file,
              const std::string& file_name,
              const std::string& tag,
              const FileMetadata& meta_data)
{
  const std::string context = tag + " in " + file_name;
  const std::string mesh = "mesh/" + tag;
  OpenSnLogicalErrorIf(not H5Has(file, mesh), "Surface " + context + " was not found.");

  std::uint64_t num_cells = 0;
  std::vector<std::uint64_t> num_face_nodes;
  OpenSnLogicalErrorIf(not H5GetDataset1DSize(file, mesh + "/cell_ids", num_cells) or
                         not H5ReadDataset1D(file, mesh + "/num_face_nodes", num_face_nodes) or
                         num_face_nodes.size() != num_cells,
                       "Failed to read consistent face metadata for " + context + ".");

  TagLayout layout;
  layout.node_offset.reserve(num_face_nodes.size() + 1);
  layout.mass_offset.reserve(num_face_nodes.size() + 1);
  for (const auto n : num_face_nodes)
  {

    OpenSnLogicalErrorIf(n == 0, "Encountered a surface face without nodes for " + context + ".");
    // An oversized n makes the node total disagree with the coordinate datasets checked below.
    const auto n_sq = CheckedMultiply(n, n, context);
    OpenSnLogicalErrorIf(
      layout.node_offset.back() > std::numeric_limits<std::uint64_t>::max() - n or
        layout.mass_offset.back() > std::numeric_limits<std::uint64_t>::max() - n_sq,
      "Size overflow for " + context + ".");
    layout.node_offset.push_back(layout.node_offset.back() + n);
    layout.mass_offset.push_back(layout.mass_offset.back() + n_sq);
  }
  const auto num_nodes = layout.GetNumNodes();

  const auto CheckSize = [&](const std::string& name, const std::uint64_t expected)
  {
    std::uint64_t size = 0;
    OpenSnLogicalErrorIf(not H5GetDataset1DSize(file, name, size) or size != expected,
                         "Dataset " + name + " in " + file_name + " has an inconsistent size.");
  };
  for (const auto* axis : {"/nodes_x", "/nodes_y", "/nodes_z"})
    CheckSize(mesh + axis, num_nodes);
  CheckSize("groupset_0/" + tag + "/M_ij", layout.mass_offset.back());
  for (size_t gs = 0; gs < meta_data.num_dirs.size(); ++gs)
  {
    const std::string group = "groupset_" + std::to_string(gs) + "/" + tag;
    const auto num_node_dirs = CheckedMultiply(num_nodes, meta_data.num_dirs[gs], context);
    CheckSize(group + "/mu", num_node_dirs);
    CheckSize(group + "/wt_d", num_node_dirs);
    CheckSize(group + "/omega", CheckedMultiply(num_node_dirs, 3, context));
    CheckSize(group + "/surf_flux",
              CheckedMultiply(num_node_dirs, meta_data.num_groups[gs], context));
  }
  return layout;
}

/**
 * Read and key faces [face, face + count) of `tag`. If `coords` is given, append their node
 * coordinates to it, three per node.
 */
std::vector<FaceCentroidKey>
ReadFaceKeys(const hid_t file,
             const std::string& file_name,
             const std::string& tag,
             const TagLayout& layout,
             const std::uint64_t face,
             const std::uint64_t count,
             const FileMetadata& meta_data,
             std::vector<double>* coords)
{
  const auto node = layout.node_offset[face];
  const auto num_nodes = layout.node_offset[face + count] - node;
  const std::string mesh = "mesh/" + tag;
  std::array<std::vector<double>, 3> xyz;
  const std::array<std::string, 3> axes = {"/nodes_x", "/nodes_y", "/nodes_z"};
  for (size_t d = 0; d < 3; ++d)
    OpenSnLogicalErrorIf(not H5ReadDataset1DSlab(file, mesh + axes[d], node, num_nodes, xyz[d]),
                         "Failed to read node coordinates for " + tag + " in " + file_name + ".");

  std::vector<FaceCentroidKey> face_keys;
  face_keys.reserve(count);
  std::uint64_t i = 0;
  for (std::uint64_t f = face; f < face + count; ++f)
  {
    const auto n = layout.GetNumFaceNodes(f);
    std::vector<Vector3> nodes;
    nodes.reserve(n);
    for (std::uint64_t k = 0; k < n; ++k, ++i)
    {
      nodes.emplace_back(xyz[0][i], xyz[1][i], xyz[2][i]);
      if (coords)
        coords->insert(coords->end(), {xyz[0][i], xyz[1][i], xyz[2][i]});
    }
    face_keys.push_back(DiscreteOrdinatesProblemIO::MakeFaceCentroidKey(std::move(nodes),
                                                                        meta_data.centroid_origin,
                                                                        meta_data.centroid_spacing,
                                                                        tag + " in " + file_name));
  }
  return face_keys;
}

/// Read the quadrature of every groupset from the first node of `tag`, which must have faces.
std::vector<SurfaceQuadrature>
ReadQuadrature(const hid_t file,
               const std::string& file_name,
               const std::string& tag,
               const FileMetadata& meta_data)
{
  std::vector<SurfaceQuadrature> quadratures(meta_data.num_dirs.size());
  for (size_t gs = 0; gs < quadratures.size(); ++gs)
  {
    auto& quadrature = quadratures[gs];
    quadrature.num_dirs = meta_data.num_dirs[gs];
    quadrature.num_groups = meta_data.num_groups[gs];
    const std::string group = "groupset_" + std::to_string(gs) + "/" + tag;
    OpenSnLogicalErrorIf(
      not H5ReadDataset1DSlab(
        file, group + "/omega", 0, 3 * quadrature.num_dirs, quadrature.omegas) or
        not H5ReadDataset1DSlab(file, group + "/wt_d", 0, quadrature.num_dirs, quadrature.weights),
      "Failed to read the quadrature of " + group + " in " + file_name + ".");
  }
  return quadratures;
}

bool
SameQuadrature(const std::vector<SurfaceQuadrature>& a, const std::vector<SurfaceQuadrature>& b)
{
  const auto Close = [](const std::vector<double>& x, const std::vector<double>& y)
  {
    if (x.size() != y.size())
      return false;
    for (size_t i = 0; i < x.size(); ++i)
      if (std::abs(x[i] - y[i]) > same_problem_tolerance)
        return false;
    return true;
  };
  if (a.size() != b.size())
    return false;
  for (size_t gs = 0; gs < a.size(); ++gs)
    if (a[gs].num_dirs != b[gs].num_dirs or a[gs].num_groups != b[gs].num_groups or
        not Close(a[gs].omegas, b[gs].omegas) or not Close(a[gs].weights, b[gs].weights))
      return false;
  return true;
}

/// Faces read from this rank's files, in read order, before they are sent to their owners.
struct FaceChunk
{
  std::vector<FaceCentroidKey> keys;
  std::vector<std::uint64_t> num_nodes;
  std::vector<double> coords;
  std::vector<double> mass;
  /// Per groupset: one row of direction cosines per face.
  std::vector<std::vector<double>> mu;
  /// Per groupset: angular flux by face, node, direction, and group.
  std::vector<std::vector<double>> psi;
  std::uint64_t bytes = 0;
};

/// Number of int64 header values per face: three key components and the node count.
const size_t face_header_size = 4;

/// Number of payload doubles of a face with `n` nodes.
std::uint64_t
FacePayloadSize(const std::uint64_t n, const std::vector<SurfaceQuadrature>& quadratures)
{
  std::uint64_t size = 3 * n + n * n;
  for (const auto& quadrature : quadratures)
    size += quadrature.num_dirs + n * quadrature.num_dirs * quadrature.num_groups;
  return size;
}

/// Reads one surface tag from this rank's files in bounded chunks.
class TagReader
{
public:
  TagReader(const std::string& file_base,
            const std::vector<std::uint64_t>& file_ids,
            const std::string& tag,
            const FileMetadata& meta,
            const std::vector<SurfaceQuadrature>& quadratures)
    : file_base_(file_base), file_ids_(file_ids), tag_(tag), meta_(meta), quadratures_(quadratures)
  {
  }

  bool IsDone() const { return next_file_ >= file_ids_.size() and not file_.has_value(); }

  /// Append faces to `chunk` until `budget` packed bytes are reached, with at least one face.
  void Read(const std::uint64_t budget, FaceChunk& chunk)
  {
    while (not IsDone())
    {
      if (not file_.has_value())
      {
        Open();
        continue;
      }
      if (face_ == layout_.GetNumFaces())
      {
        file_.reset();
        continue;
      }

      std::uint64_t end = face_;
      while (end < layout_.GetNumFaces())
      {
        const auto bytes =
          sizeof(std::int64_t) * face_header_size +
          sizeof(double) * FacePayloadSize(layout_.GetNumFaceNodes(end), quadratures_);
        if ((not chunk.keys.empty() or end > face_) and chunk.bytes + bytes > budget)
          break;
        chunk.bytes += bytes;
        ++end;
      }
      if (end == face_)
        return;
      ReadSegment(end - face_, chunk);
      face_ = end;
    }
  }

private:
  void Open()
  {
    file_name_ = PartitionFileName(file_base_, file_ids_[next_file_++]);
    file_.emplace(H5Fopen(file_name_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    OpenSnLogicalErrorIf(file_->Id() < 0, "Failed to open " + file_name_ + ".");
    layout_ = ReadTagLayout(file_->Id(), file_name_, tag_, meta_);
    face_ = 0;
  }

  template <typename T>
  static void Append(std::vector<T>& destination, std::vector<T>&& source)
  {
    if (destination.empty())
      destination = std::move(source);
    else
      destination.insert(destination.end(), source.begin(), source.end());
  }

  /// Read faces [face_, face_ + count) into `chunk`.
  void ReadSegment(const std::uint64_t count, FaceChunk& chunk)
  {
    const auto id = file_->Id();
    const auto node = layout_.node_offset[face_];
    const auto num_nodes = layout_.node_offset[face_ + count] - node;
    const auto mass_start = layout_.mass_offset[face_];
    const auto mass_size = layout_.mass_offset[face_ + count] - mass_start;

    auto keys = ReadFaceKeys(id, file_name_, tag_, layout_, face_, count, meta_, &chunk.coords);
    chunk.keys.insert(chunk.keys.end(), keys.begin(), keys.end());
    for (std::uint64_t f = face_; f < face_ + count; ++f)
      chunk.num_nodes.push_back(layout_.GetNumFaceNodes(f));

    std::vector<double> mass;
    OpenSnLogicalErrorIf(
      not H5ReadDataset1DSlab(id, "groupset_0/" + tag_ + "/M_ij", mass_start, mass_size, mass),
      "Failed to read the mass matrix of " + tag_ + " in " + file_name_ + ".");
    Append(chunk.mass, std::move(mass));

    chunk.mu.resize(quadratures_.size());
    chunk.psi.resize(quadratures_.size());
    for (size_t gs = 0; gs < quadratures_.size(); ++gs)
    {
      const auto num_dirs = quadratures_[gs].num_dirs;
      const auto num_groups = quadratures_[gs].num_groups;
      const std::string group = "groupset_" + std::to_string(gs) + "/" + tag_;

      std::vector<double> mu;
      OpenSnLogicalErrorIf(
        not H5ReadDataset1DSlab(id, group + "/mu", node * num_dirs, num_nodes * num_dirs, mu),
        "Failed to read direction cosines of " + group + " in " + file_name_ + ".");
      // The cosine depends only on the face normal; keep the first node's row of each face.
      std::uint64_t row = 0;
      for (std::uint64_t f = face_; f < face_ + count; ++f)
      {
        const auto n = layout_.GetNumFaceNodes(f);
        for (std::uint64_t k = 1; k < n; ++k)
          for (std::uint64_t d = 0; d < num_dirs; ++d)
            OpenSnLogicalErrorIf(std::abs(mu[(row + k) * num_dirs + d] - mu[row * num_dirs + d]) >
                                   same_problem_tolerance,
                                 "Direction cosines vary across the nodes of face " +
                                   std::to_string(f) + " of " + group + " in " + file_name_ + ".");
        chunk.mu[gs].insert(chunk.mu[gs].end(),
                            mu.begin() + static_cast<std::ptrdiff_t>(row * num_dirs),
                            mu.begin() + static_cast<std::ptrdiff_t>((row + 1) * num_dirs));
        row += n;
      }

      std::vector<double> psi;
      const auto stride = num_dirs * num_groups;
      OpenSnLogicalErrorIf(
        not H5ReadDataset1DSlab(id, group + "/surf_flux", node * stride, num_nodes * stride, psi),
        "Failed to read the angular flux of " + group + " in " + file_name_ + ".");
      Append(chunk.psi[gs], std::move(psi));
    }
  }

  const std::string& file_base_;
  const std::vector<std::uint64_t>& file_ids_;
  const std::string& tag_;
  const FileMetadata& meta_;
  const std::vector<SurfaceQuadrature>& quadratures_;

  size_t next_file_ = 0;
  std::optional<H5FileHandle> file_;
  std::string file_name_;
  TagLayout layout_;
  std::uint64_t face_ = 0;
};

std::vector<int>
MakeOffsets(const std::vector<int>& counts)
{
  std::vector<int> offsets(counts.size(), 0);
  for (size_t p = 1; p < counts.size(); ++p)
    offsets[p] = offsets[p - 1] + counts[p - 1];
  return offsets;
}

/**
 * Send the faces of `chunk` to their owners and append the received faces to `shard`. Collective.
 * Returns a description of a local unpacking error, or an empty string.
 */
std::string
ExchangeChunk(const mpi::Communicator& comm,
              const std::vector<SurfaceQuadrature>& quadratures,
              FaceChunk& chunk,
              SurfaceShard& shard)
{
  const auto num_ranks = comm.size();
  const auto num_faces = chunk.keys.size();

  // Count per destination. A chunk is bounded by the send budget, so its counts fit in an int.
  std::vector<int> owner(num_faces);
  std::vector<std::uint64_t> send_counts(2 * num_ranks, 0);
  for (size_t f = 0; f < num_faces; ++f)
  {
    owner[f] = SurfaceFaceOwner(chunk.keys[f], num_ranks);
    send_counts[2 * owner[f]] += face_header_size;
    send_counts[2 * owner[f] + 1] += FacePayloadSize(chunk.num_nodes[f], quadratures);
  }
  std::vector<std::uint64_t> recv_counts(2 * num_ranks, 0);
  comm.all_to_all(send_counts.data(), 2, recv_counts.data(), 2);

  // The receive volume depends on the senders, so check it collectively before using int counts.
  std::uint64_t recv_headers = 0;
  std::uint64_t recv_payload = 0;
  for (int p = 0; p < num_ranks; ++p)
  {
    recv_headers += recv_counts[2 * p];
    recv_payload += recv_counts[2 * p + 1];
  }
  std::uint64_t send_payload = 0;
  for (int p = 0; p < num_ranks; ++p)
    send_payload += send_counts[2 * p + 1];
  const auto int_max = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  ThrowIfAnyRankFailed(comm,
                       std::max({recv_headers, recv_payload, send_payload}) > int_max
                         ? "the redistribution round exceeds the MPI count range; use a smaller "
                           "send budget"
                         : "",
                       "Surface angular-flux redistribution");

  std::vector<int> header_counts(num_ranks), payload_counts(num_ranks);
  std::vector<int> recv_header_counts(num_ranks), recv_payload_counts(num_ranks);
  for (int p = 0; p < num_ranks; ++p)
  {
    header_counts[p] = static_cast<int>(send_counts[2 * p]);
    payload_counts[p] = static_cast<int>(send_counts[2 * p + 1]);
    recv_header_counts[p] = static_cast<int>(recv_counts[2 * p]);
    recv_payload_counts[p] = static_cast<int>(recv_counts[2 * p + 1]);
  }
  const auto header_offsets = MakeOffsets(header_counts);
  const auto payload_offsets = MakeOffsets(payload_counts);

  // Pack in destination order.
  std::vector<std::int64_t> headers(num_faces * face_header_size);
  std::vector<double> payload(send_payload);
  {
    auto header_cursor = header_offsets;
    auto payload_cursor = payload_offsets;
    std::uint64_t node = 0;
    std::uint64_t mass = 0;
    std::vector<std::uint64_t> psi(quadratures.size(), 0);
    for (size_t f = 0; f < num_faces; ++f)
    {
      const auto n = chunk.num_nodes[f];
      auto* h = &headers[header_cursor[owner[f]]];
      h[0] = std::get<0>(chunk.keys[f]);
      h[1] = std::get<1>(chunk.keys[f]);
      h[2] = std::get<2>(chunk.keys[f]);
      h[3] = static_cast<std::int64_t>(n);
      header_cursor[owner[f]] += static_cast<int>(face_header_size);

      auto* out = &payload[payload_cursor[owner[f]]];
      out = std::copy_n(&chunk.coords[3 * node], 3 * n, out);
      out = std::copy_n(&chunk.mass[mass], n * n, out);
      for (size_t gs = 0; gs < quadratures.size(); ++gs)
      {
        const auto num_dirs = quadratures[gs].num_dirs;
        const auto size = n * num_dirs * quadratures[gs].num_groups;
        out = std::copy_n(&chunk.mu[gs][f * num_dirs], num_dirs, out);
        out = std::copy_n(&chunk.psi[gs][psi[gs]], size, out);
        psi[gs] += size;
      }
      payload_cursor[owner[f]] += static_cast<int>(FacePayloadSize(n, quadratures));
      node += n;
      mass += n * n;
    }
  }
  chunk = FaceChunk();

  std::vector<std::int64_t> recv_header_values;
  comm.all_to_all(headers,
                  header_counts,
                  header_offsets,
                  recv_header_values,
                  recv_header_counts,
                  MakeOffsets(recv_header_counts));
  headers = {};
  std::vector<double> recv_payload_values;
  comm.all_to_all(payload,
                  payload_counts,
                  payload_offsets,
                  recv_payload_values,
                  recv_payload_counts,
                  MakeOffsets(recv_payload_counts));
  payload = {};

  // Unpack in arrival order.
  const double* in = recv_payload_values.data();
  const double* const in_end = in + recv_payload_values.size();
  for (size_t h = 0; h < recv_header_values.size(); h += face_header_size)
  {
    const auto n = static_cast<std::uint64_t>(recv_header_values[h + 3]);
    if (n == 0 or static_cast<std::uint64_t>(in_end - in) < FacePayloadSize(n, quadratures))
      return "received an inconsistent surface face";
    const FaceCentroidKey key = {
      recv_header_values[h], recv_header_values[h + 1], recv_header_values[h + 2]};
    shard.index.emplace_back(key, shard.GetNumFaces());
    shard.coords.insert(shard.coords.end(), in, in + 3 * n);
    in += 3 * n;
    shard.mass.insert(shard.mass.end(), in, in + n * n);
    in += n * n;
    for (size_t gs = 0; gs < quadratures.size(); ++gs)
    {
      const auto num_dirs = quadratures[gs].num_dirs;
      const auto size = n * num_dirs * quadratures[gs].num_groups;
      auto& values = shard.groupsets[gs];
      values.mu.insert(values.mu.end(), in, in + num_dirs);
      in += num_dirs;
      values.psi.insert(values.psi.end(), in, in + size);
      in += size;
    }
    shard.node_offset.push_back(shard.node_offset.back() + n);
    shard.mass_offset.push_back(shard.mass_offset.back() + n * n);
  }
  if (in != in_end)
    return "received surface payload that does not match its face headers";
  return {};
}

} // namespace

std::optional<std::uint64_t>
SurfaceShard::Find(const DiscreteOrdinatesProblemIO::FaceCentroidKey& key) const
{
  const auto it =
    std::lower_bound(index.begin(),
                     index.end(),
                     key,
                     [](const auto& entry, const auto& value) { return entry.first < value; });
  if (it == index.end() or it->first != key)
    return std::nullopt;
  return it->second;
}

int
SurfaceFaceOwner(const DiscreteOrdinatesProblemIO::FaceCentroidKey& key, const int num_ranks)
{
  // splitmix64 finalizer: deterministic across ranks, builds, and platforms.
  const auto Mix = [](std::uint64_t x)
  {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  };
  auto hash = Mix(static_cast<std::uint64_t>(std::get<0>(key)));
  hash = Mix(hash ^ static_cast<std::uint64_t>(std::get<1>(key)));
  hash = Mix(hash ^ static_cast<std::uint64_t>(std::get<2>(key)));
  return static_cast<int>(hash % static_cast<std::uint64_t>(num_ranks));
}

DistributedSurfaceData
ReadDistributedSurfaceAngularFluxes(const std::string& file_base,
                                    const std::vector<std::string>& surfaces,
                                    const mpi::Communicator& comm,
                                    const std::uint64_t max_send_bytes)
{
  OpenSnInvalidArgumentIf(surfaces.empty(), "At least one surface tag is required.");
  OpenSnInvalidArgumentIf(std::set<std::string>(surfaces.begin(), surfaces.end()).size() !=
                            surfaces.size(),
                          "Surface tags must be unique.");
  OpenSnInvalidArgumentIf(max_send_bytes == 0, "The send budget must be positive.");
  const std::string operation = "Reading surface angular fluxes from " + file_base;
  const auto rank = comm.rank();
  const auto num_ranks = comm.size();

  // Rank 0 discovers the file set and its reference metadata.
  FileMetadata reference_data;
  std::vector<char> root_error;
  if (rank == 0)
  {
    try
    {
      const auto file_name = PartitionFileName(file_base, 0);
      const H5FileHandle file(H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
      OpenSnLogicalErrorIf(file.Id() < 0, "Failed to open " + file_name + ".");
      reference_data = ReadFileMetadata(file.Id(), file_name);
    }
    catch (const std::exception& error)
    {
      const std::string message = error.what();
      root_error.assign(message.begin(), message.end());
    }
  }
  BroadcastVector(comm, root_error, 0);
  if (not root_error.empty())
    throw std::logic_error(operation +
                           " failed: " + std::string(root_error.begin(), root_error.end()));

  BroadcastFileMetadata(comm, reference_data);
  const auto num_partitions = reference_data.num_partitions;
  const auto num_groupsets = reference_data.num_dirs.size();

  DistributedSurfaceData data;
  data.adjoint = reference_data.adjoint;
  data.num_partitions = num_partitions;
  data.centroid_origin = reference_data.centroid_origin;
  data.centroid_spacing = reference_data.centroid_spacing;

  // File k is read by rank k mod P.
  std::vector<std::uint64_t> file_ids;
  for (auto k = static_cast<std::uint64_t>(rank); k < num_partitions;
       k += static_cast<std::uint64_t>(num_ranks))
    file_ids.push_back(k);

  // Planning pass: validate every file, read the quadrature from a non-empty file, and count the
  // faces, nodes, and mass-matrix entries each rank will own, so owner storage is sized exactly.
  std::string error;
  std::optional<std::vector<SurfaceQuadrature>> local_quadrature;
  std::vector<std::uint64_t> send_totals(3 * surfaces.size() * num_ranks, 0);
  for (const auto file_id : file_ids)
  {
    const auto file_name = PartitionFileName(file_base, file_id);
    try
    {
      const H5FileHandle file(H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
      OpenSnLogicalErrorIf(file.Id() < 0, "Failed to open " + file_name + ".");
      const auto mismatch =
        CompareFileMetadata(reference_data, ReadFileMetadata(file.Id(), file_name));
      OpenSnLogicalErrorIf(not mismatch.empty(),
                           file_name + " does not belong to the file set: " + mismatch + ".");

      for (size_t s = 0; s < surfaces.size(); ++s)
      {
        const auto& surface_tag = surfaces[s];
        const auto layout = ReadTagLayout(file.Id(), file_name, surface_tag, reference_data);
        if (layout.GetNumFaces() == 0)
          continue;

        // Every non-empty tag of every file must use the same quadrature per groupset. The
        // quadrature is read from the first node of each tag only; the writer stores identical
        // copies at every node.
        auto quadrature = ReadQuadrature(file.Id(), file_name, surface_tag, reference_data);
        if (not local_quadrature.has_value())
          local_quadrature = std::move(quadrature);
        else
          OpenSnLogicalErrorIf(not SameQuadrature(*local_quadrature, quadrature),
                               "The quadrature of " + surface_tag + " in " + file_name +
                                 " differs from the rest of the file set.");

        const auto face_keys = ReadFaceKeys(file.Id(),
                                            file_name,
                                            surface_tag,
                                            layout,
                                            0,
                                            layout.GetNumFaces(),
                                            reference_data,
                                            nullptr);
        for (size_t f = 0; f < face_keys.size(); ++f)
        {
          const auto n = layout.GetNumFaceNodes(f);
          auto* totals =
            &send_totals[3 * (s * num_ranks + SurfaceFaceOwner(face_keys[f], num_ranks))];
          totals[0] += 1;
          totals[1] += n;
          totals[2] += n * n;
        }
      }
    }
    catch (const std::exception& exception)
    {
      error = exception.what();
      break;
    }
  }
  ThrowIfAnyRankFailed(comm, error, operation);

  // Owner totals: entry p of each (surface, rank) block arrives from rank p.
  std::vector<std::uint64_t> recv_totals;
  {
    // Reorder to rank-major blocks for the all-to-all.
    const auto block = 3 * surfaces.size();
    std::vector<std::uint64_t> send(send_totals.size());
    for (size_t s = 0; s < surfaces.size(); ++s)
      for (int p = 0; p < num_ranks; ++p)
        for (size_t c = 0; c < 3; ++c)
          send[p * block + 3 * s + c] = send_totals[3 * (s * num_ranks + p) + c];
    std::vector<std::uint64_t> recv(send.size(), 0);
    comm.all_to_all(send.data(), static_cast<int>(block), recv.data(), static_cast<int>(block));
    recv_totals.assign(block, 0);
    for (int p = 0; p < num_ranks; ++p)
      for (size_t i = 0; i < block; ++i)
        recv_totals[i] += recv[p * block + i];
  }

  // Every tag must have faces somewhere.
  std::vector<std::uint64_t> local_faces(surfaces.size()), global_faces(surfaces.size());
  for (size_t s = 0; s < surfaces.size(); ++s)
    local_faces[s] = recv_totals[3 * s];
  comm.all_reduce(local_faces.data(),
                  static_cast<int>(surfaces.size()),
                  global_faces.data(),
                  mpi::op::sum<std::uint64_t>());
  for (size_t s = 0; s < surfaces.size(); ++s)
    OpenSnLogicalErrorIf(global_faces[s] == 0,
                         operation + " failed: surface " + surfaces[s] +
                           " has no faces in any of " + std::to_string(num_partitions) + " files.");

  // All ranks adopt the quadrature of the lowest rank that read one and check their own against it.
  {
    int source = local_quadrature.has_value() ? rank : num_ranks;
    comm.all_reduce(source, mpi::op::min<int>());
    data.groupsets.resize(num_groupsets);
    for (size_t gs = 0; gs < num_groupsets; ++gs)
    {
      auto& quadrature = data.groupsets[gs];
      if (rank == source)
        quadrature = (*local_quadrature)[gs];
      else
      {
        quadrature.num_dirs = reference_data.num_dirs[gs];
        quadrature.num_groups = reference_data.num_groups[gs];
        quadrature.omegas.resize(3 * quadrature.num_dirs);
        quadrature.weights.resize(quadrature.num_dirs);
      }
      comm.broadcast(quadrature.omegas.data(), static_cast<int>(quadrature.omegas.size()), source);
      comm.broadcast(
        quadrature.weights.data(), static_cast<int>(quadrature.weights.size()), source);
    }
    ThrowIfAnyRankFailed(comm,
                         local_quadrature.has_value() and
                             not SameQuadrature(*local_quadrature, data.groupsets)
                           ? "the quadrature differs across the file set"
                           : "",
                         operation);
  }

  // Streaming pass: send bounded chunks of faces to their owners.
  for (size_t s = 0; s < surfaces.size(); ++s)
  {
    const auto& surface_tag = surfaces[s];
    auto& shard = data.surfaces[surface_tag];
    const auto num_faces = recv_totals[3 * s];
    const auto num_nodes = recv_totals[3 * s + 1];
    shard.index.reserve(num_faces);
    shard.node_offset.reserve(num_faces + 1);
    shard.mass_offset.reserve(num_faces + 1);
    shard.coords.reserve(3 * num_nodes);
    shard.mass.reserve(recv_totals[3 * s + 2]);
    shard.groupsets.resize(num_groupsets);
    for (size_t gs = 0; gs < num_groupsets; ++gs)
    {
      const auto& quadrature = data.groupsets[gs];
      shard.groupsets[gs].mu.reserve(num_faces * quadrature.num_dirs);
      shard.groupsets[gs].psi.reserve(num_nodes * quadrature.num_dirs * quadrature.num_groups);
    }

    TagReader reader(file_base, file_ids, surface_tag, reference_data, data.groupsets);
    while (true)
    {
      FaceChunk chunk;
      if (error.empty())
      {
        try
        {
          reader.Read(max_send_bytes, chunk);
        }
        catch (const std::exception& exception)
        {
          error = exception.what();
          chunk = FaceChunk();
        }
      }
      // Every rank takes part in every round until no rank has faces left.
      std::array<int, 2> state = {error.empty() ? 0 : 1, chunk.keys.empty() ? 0 : 1};
      std::array<int, 2> global_state = {0, 0};
      comm.all_reduce(state.data(), 2, global_state.data(), mpi::op::max<int>());
      if (global_state[0] != 0)
        ThrowIfAnyRankFailed(comm, error, operation);
      if (global_state[1] == 0)
        break;
      error = ExchangeChunk(comm, data.groupsets, chunk, shard);
    }

    if (shard.GetNumFaces() != num_faces)
      error = "received " + std::to_string(shard.GetNumFaces()) + " faces of " + surface_tag +
              ", expected " + std::to_string(num_faces);
    std::sort(shard.index.begin(), shard.index.end());
    for (size_t i = 1; i < shard.index.size() and error.empty(); ++i)
      if (shard.index[i].first == shard.index[i - 1].first)
        error = "surface " + surface_tag +
                " contains two faces with the same centroid key. Face centroids must differ by "
                "more than the centroid spacing " +
                std::to_string(data.centroid_spacing) + ".";
    ThrowIfAnyRankFailed(comm, error, operation);
  }

  std::uint64_t total_faces = 0;
  for (const auto faces : global_faces)
    total_faces += faces;
  log.Log0Verbose1() << "Read " << total_faces << " surface faces of " << surfaces.size()
                     << " surface(s) from " << num_partitions << " file(s) " << file_base
                     << "<rank>.h5 onto " << num_ranks << " rank(s).";
  return data;
}

namespace
{

/// Indices of the directions of `omegas`, three components per direction, sorted by z component.
std::vector<std::uint64_t>
SortDirectionsByZ(const std::vector<double>& omegas)
{
  std::vector<std::uint64_t> order(omegas.size() / 3);
  for (std::uint64_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::sort(order.begin(),
            order.end(),
            [&omegas](const std::uint64_t a, const std::uint64_t b)
            { return omegas[3 * a + 2] < omegas[3 * b + 2]; });
  return order;
}

/**
 * Return the direction of `omegas` nearest to `query` in Euclidean distance, skipping `exclude`,
 * and that distance. `order` is `SortDirectionsByZ(omegas)`; the search scans outward from the z
 * component of `query` and stops once the z difference alone exceeds the best distance.
 */
std::pair<std::uint64_t, double>
NearestDirection(const std::vector<double>& omegas,
                 const std::vector<std::uint64_t>& order,
                 const double* query,
                 const std::uint64_t exclude)
{
  std::pair<std::uint64_t, double> best = {exclude, std::numeric_limits<double>::infinity()};
  const auto Visit = [&](const std::uint64_t j)
  {
    if (j == exclude)
      return;
    const double* w = &omegas[3 * j];
    const double distance =
      std::sqrt((w[0] - query[0]) * (w[0] - query[0]) + (w[1] - query[1]) * (w[1] - query[1]) +
                (w[2] - query[2]) * (w[2] - query[2]));
    if (distance < best.second)
      best = {j, distance};
  };
  const auto start =
    static_cast<size_t>(std::lower_bound(order.begin(),
                                         order.end(),
                                         query[2],
                                         [&omegas](const std::uint64_t j, const double z)
                                         { return omegas[3 * j + 2] < z; }) -
                        order.begin());
  for (size_t k = start; k < order.size() and omegas[3 * order[k] + 2] - query[2] <= best.second;
       ++k)
    Visit(order[k]);
  for (size_t k = start; k > 0 and query[2] - omegas[3 * order[k - 1] + 2] <= best.second; --k)
    Visit(order[k - 1]);
  return best;
}

/**
 * Distance from each direction of `omegas` to its nearest other direction, capped at 2, the largest
 * distance between unit vectors. Throws if two directions coincide.
 */
std::vector<double>
DirectionSpacing(const std::vector<double>& omegas,
                 const std::vector<std::uint64_t>& order,
                 const std::string& description)
{
  std::vector<double> spacing(order.size());
  for (std::uint64_t j = 0; j < spacing.size(); ++j)
  {
    spacing[j] = std::min(2.0, NearestDirection(omegas, order, &omegas[3 * j], j).second);
    OpenSnInvalidArgumentIf(not(spacing[j] > 0.0),
                            "The " + description + " contains duplicate directions.");
  }
  return spacing;
}

/**
 * Map each direction of `from` to the direction of `to` with the same direction vector.
 *
 * Forward direction `d` maps to its nearest adjoint direction `j` when |omega_d - omega_j| is at
 * most direction_match_tolerance times the distance from `j` to its nearest other adjoint
 * direction. Every other adjoint direction is then nearly that full spacing from `omega_d`, so the
 * match is unambiguous however closely the quadrature places its directions.
 */
std::vector<std::uint64_t>
BuildSameDirectionMap(const SurfaceQuadrature& from,
                      const SurfaceQuadrature& to,
                      const std::string& context)
{
  OpenSnInvalidArgumentIf(from.num_dirs != to.num_dirs or from.num_groups != to.num_groups,
                          "The forward and adjoint data have different direction or group counts "
                          "in " +
                            context + ".");
  for (const auto* omegas : {&from.omegas, &to.omegas})
    OpenSnInvalidArgumentIf(
      omegas->size() != 3 * from.num_dirs or
        not std::all_of(
          omegas->begin(), omegas->end(), [](const double value) { return std::isfinite(value); }),
      "The forward or adjoint directions of " + context + " are missing or not finite.");

  const auto from_order = SortDirectionsByZ(from.omegas);
  const auto to_order = SortDirectionsByZ(to.omegas);
  DirectionSpacing(from.omegas, from_order, "forward quadrature of " + context);
  const auto to_spacing = DirectionSpacing(to.omegas, to_order, "adjoint quadrature of " + context);

  std::vector<std::uint64_t> map(from.num_dirs);
  std::vector<bool> used(to.num_dirs, false);
  for (std::uint64_t d = 0; d < from.num_dirs; ++d)
  {
    const auto [match, distance] =
      NearestDirection(to.omegas, to_order, &from.omegas[3 * d], to.num_dirs);
    if (not(distance <= direction_match_tolerance * to_spacing[match]))
    {
      std::ostringstream message;
      message << std::setprecision(3) << "Forward direction " << d << " of " << context
              << " has no matching adjoint direction. The nearest, adjoint direction " << match
              << ", is " << distance << " away, more than " << direction_match_tolerance
              << " times its spacing " << to_spacing[match] << " to the next adjoint direction.";
      OpenSnInvalidArgument(message.str());
    }
    OpenSnInvalidArgumentIf(used[match],
                            "The direction mapping of " + context + " is not one-to-one.");
    used[match] = true;
    map[d] = match;
  }
  return map;
}

/**
 * Pair the nodes of face `fa` of `anchor` with those of face `fp` of `partner` by location, within
 * 1e-6 times the face size. `permutation[i]` is the partner node of anchor node `i`.
 */
bool
MatchFaceNodes(const SurfaceShard& anchor,
               const std::uint64_t fa,
               const SurfaceShard& partner,
               const std::uint64_t fp,
               std::vector<std::uint64_t>& permutation)
{
  const auto n = anchor.GetNumFaceNodes(fa);
  if (partner.GetNumFaceNodes(fp) != n)
    return false;
  permutation.assign(n, 0);
  // A single-node face is identified by its key alone.
  if (n == 1)
    return true;

  const double* a = &anchor.coords[3 * anchor.node_offset[fa]];
  const double* b = &partner.coords[3 * partner.node_offset[fp]];
  const auto DistanceSquared = [](const double* x, const double* y)
  {
    return (x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]) +
           (x[2] - y[2]) * (x[2] - y[2]);
  };
  double face_size_sq = 0.0;
  for (std::uint64_t i = 0; i < n; ++i)
    for (std::uint64_t j = i + 1; j < n; ++j)
      face_size_sq = std::max(face_size_sq, DistanceSquared(&a[3 * i], &a[3 * j]));
  const double tolerance_sq = 1.0e-12 * face_size_sq;

  std::vector<bool> used(n, false);
  for (std::uint64_t i = 0; i < n; ++i)
  {
    bool found = false;
    for (std::uint64_t j = 0; j < n and not found; ++j)
      if (not used[j] and DistanceSquared(&a[3 * i], &b[3 * j]) <= tolerance_sq)
      {
        permutation[i] = j;
        used[j] = true;
        found = true;
      }
    if (not found)
      return false;
  }
  return true;
}

/**
 * Return sum_ij M_ij sum_g a(r_i, da, g) b(c_j, db, g) for one face, where `a` and `b` hold the
 * face's angular flux ordered by node, direction, and group, `r_i` is `row_nodes[i]` (or `i` when
 * `row_nodes` is null), and `c_j` is `col_nodes[j]`.
 */
double
FaceProduct(const double* mass,
            const std::uint64_t n,
            const double* a,
            const std::vector<std::uint64_t>* row_nodes,
            const std::uint64_t da,
            const double* b,
            const std::vector<std::uint64_t>& col_nodes,
            const std::uint64_t db,
            const std::uint64_t num_dirs,
            const std::uint64_t num_groups)
{
  double sum = 0.0;
  for (std::uint64_t i = 0; i < n; ++i)
  {
    const auto row = row_nodes ? (*row_nodes)[i] : i;
    const double* a_i = a + (row * num_dirs + da) * num_groups;
    for (std::uint64_t j = 0; j < n; ++j)
    {
      const double* b_j = b + (col_nodes[j] * num_dirs + db) * num_groups;
      double dot = 0.0;
      for (std::uint64_t g = 0; g < num_groups; ++g)
        dot += a_i[g] * b_j[g];
      sum += mass[i * n + j] * dot;
    }
  }
  return sum;
}

/// Pointer to the angular flux of face `f` of `shard` in groupset `gs`.
const double*
FacePsi(const SurfaceShard& shard,
        const std::uint64_t f,
        const size_t gs,
        const SurfaceQuadrature& quadrature)
{
  return &shard.groupsets[gs]
            .psi[shard.node_offset[f] * quadrature.num_dirs * quadrature.num_groups];
}

bool
EndsWith(const std::string& value, const std::string& suffix)
{
  return value.size() > suffix.size() and
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

std::vector<SurfaceResponse>
ComputeSurfaceResponses(const DistributedSurfaceData& forward,
                        const DistributedSurfaceData& adjoint,
                        const std::vector<std::string>& surfaces,
                        const mpi::Communicator& comm)
{
  OpenSnInvalidArgumentIf(forward.surfaces.empty() or adjoint.surfaces.empty(),
                          "Both data sets must contain surface angular fluxes.");
  OpenSnInvalidArgumentIf(forward.adjoint,
                          "The forward data were written from a problem in adjoint mode.");
  OpenSnInvalidArgumentIf(not adjoint.adjoint,
                          "The adjoint data were written from a problem in forward mode.");
  OpenSnInvalidArgumentIf(forward.centroid_origin != adjoint.centroid_origin or
                            forward.centroid_spacing != adjoint.centroid_spacing,
                          "The forward and adjoint data have different face-centroid origins or "
                          "spacings. Both must be written for the same mesh.");
  OpenSnInvalidArgumentIf(forward.groupsets.size() != adjoint.groupsets.size(),
                          "The forward and adjoint data have different numbers of groupsets.");

  const auto num_groupsets = forward.groupsets.size();
  std::vector<std::vector<std::uint64_t>> direction_maps;
  direction_maps.reserve(num_groupsets);
  for (size_t gs = 0; gs < num_groupsets; ++gs)
    direction_maps.push_back(BuildSameDirectionMap(
      forward.groupsets[gs], adjoint.groupsets[gs], "groupset " + std::to_string(gs)));

  // Map the available surfaces by name from the forward data.
  // Interior surfaces are stored as <name>_u and <name>_d; any other tag is a boundary.
  // ex; available_surfs = { "inter_x" -> true, "xmax" -> false, "xmin" -> false }
  std::map<std::string, bool> available_surfs;
  for (const auto& surface : forward.surfaces)
  {
    const auto& surface_tag = surface.first;
    std::string name = surface_tag;
    bool interior = false;
    if (EndsWith(surface_tag, "_u") or EndsWith(surface_tag, "_d"))
    {
      name = surface_tag.substr(0, surface_tag.size() - 2);
      const auto other = name + (EndsWith(surface_tag, "_u") ? "_d" : "_u");
      OpenSnInvalidArgumentIf(forward.surfaces.count(other) == 0,
                              "Interior surface tag " + surface_tag + " was read without " + other +
                                ". Both sides of an interior surface are required.");
      interior = true;
    }
    const auto [it, inserted] = available_surfs.emplace(name, interior);
    OpenSnInvalidArgumentIf(not inserted and it->second != interior,
                            "Surface name " + name +
                              " refers to both a boundary and an interior surface.");
  }

  std::vector<std::pair<std::string, bool>> selected_surfs;
  if (surfaces.empty())
    // If no surfaces are specified all are used.
    selected_surfs.assign(available_surfs.begin(), available_surfs.end());
  else
  {
    // Guard for when a surface is requested multiple times.
    std::set<std::string> parsed_surfs;
    for (const auto& surface : surfaces)
    {
      const auto it = available_surfs.find(surface);
      OpenSnInvalidArgumentIf(it == available_surfs.end(),
                              "Surface " + surface + " is not in the forward data.");
      OpenSnInvalidArgumentIf(not parsed_surfs.insert(surface).second,
                              "Surface " + surface + " was requested more than once.");
      selected_surfs.emplace_back(*it);
    }
  }

  for (const auto& [name, interior] : selected_surfs)
    for (const auto& surface_tag : interior ? std::vector<std::string>{name + "_u", name + "_d"}
                                            : std::vector<std::string>{name})
      OpenSnInvalidArgumentIf(adjoint.surfaces.count(surface_tag) == 0,
                              "Surface " + name + " requires tag " + surface_tag +
                                " in the adjoint data.");

  // Per surface: outgoing and incoming contributions, and counts of anchor faces without a partner,
  // partner faces without an anchor, and matched faces with different node locations.
  const auto num_surfaces = selected_surfs.size();
  std::vector<double> local_responses(2 * num_surfaces, 0.0);
  std::vector<std::uint64_t> local_counts(3 * num_surfaces, 0);
  std::vector<std::uint64_t> perm_fd, perm_au, perm_ad;

  for (size_t s = 0; s < num_surfaces; ++s)
  {
    const auto& [name, interior] = selected_surfs[s];
    double& outgoing = local_responses[2 * s];
    double& incoming = local_responses[2 * s + 1];
    std::uint64_t& missing = local_counts[3 * s];
    std::uint64_t& extra = local_counts[3 * s + 1];
    std::uint64_t& mismatched = local_counts[3 * s + 2];

    if (not interior)
    {
      const auto& anchor = forward.surfaces.at(name);
      const auto& partner = adjoint.surfaces.at(name);
      std::uint64_t matched = 0;
      for (const auto& [key, fa] : anchor.index)
      {
        const auto fp = partner.Find(key);
        if (not fp.has_value())
        {
          ++missing;
          continue;
        }
        ++matched;
        if (not MatchFaceNodes(anchor, fa, partner, *fp, perm_ad))
        {
          ++mismatched;
          continue;
        }

        const auto n = anchor.GetNumFaceNodes(fa);
        const double* mass = &anchor.mass[anchor.mass_offset[fa]];
        for (size_t gs = 0; gs < num_groupsets; ++gs)
        {
          const auto& quadrature = forward.groupsets[gs];
          const auto num_dirs = quadrature.num_dirs;
          const double* mu = &anchor.groupsets[gs].mu[fa * num_dirs];
          const double* psi = FacePsi(anchor, fa, gs, quadrature);
          const double* psi_adj = FacePsi(partner, *fp, gs, adjoint.groupsets[gs]);
          for (std::uint64_t d = 0; d < num_dirs; ++d)
            if (mu[d] > 0.0)
              outgoing += mu[d] * quadrature.weights[d] *
                          FaceProduct(mass,
                                      n,
                                      psi,
                                      nullptr,
                                      d,
                                      psi_adj,
                                      perm_ad,
                                      direction_maps[gs][d],
                                      num_dirs,
                                      quadrature.num_groups);
        }
      }
      extra += partner.GetNumFaces() - matched;
      continue;
    }

    // The anchor is the forward `_u` face; its normal is the positive axis.
    const auto& fwd_u = forward.surfaces.at(name + "_u");
    const auto& fwd_d = forward.surfaces.at(name + "_d");
    const auto& adj_u = adjoint.surfaces.at(name + "_u");
    const auto& adj_d = adjoint.surfaces.at(name + "_d");
    std::array<std::uint64_t, 3> matched = {0, 0, 0};
    for (const auto& [key, fa] : fwd_u.index)
    {
      const auto fd = fwd_d.Find(key);
      const auto au = adj_u.Find(key);
      const auto ad = adj_d.Find(key);
      matched[0] += fd.has_value() ? 1 : 0;
      matched[1] += au.has_value() ? 1 : 0;
      matched[2] += ad.has_value() ? 1 : 0;
      if (not(fd.has_value() and au.has_value() and ad.has_value()))
      {
        ++missing;
        continue;
      }
      if (not MatchFaceNodes(fwd_u, fa, fwd_d, *fd, perm_fd) or
          not MatchFaceNodes(fwd_u, fa, adj_u, *au, perm_au) or
          not MatchFaceNodes(fwd_u, fa, adj_d, *ad, perm_ad))
      {
        ++mismatched;
        continue;
      }

      const auto n = fwd_u.GetNumFaceNodes(fa);
      const double* mass = &fwd_u.mass[fwd_u.mass_offset[fa]];
      for (size_t gs = 0; gs < num_groupsets; ++gs)
      {
        const auto& quadrature = forward.groupsets[gs];
        const auto& adj_quadrature = adjoint.groupsets[gs];
        const auto num_dirs = quadrature.num_dirs;
        const auto num_groups = quadrature.num_groups;
        const double* mu = &fwd_u.groupsets[gs].mu[fa * num_dirs];
        const double* psi_u = FacePsi(fwd_u, fa, gs, quadrature);
        const double* psi_d = FacePsi(fwd_d, *fd, gs, quadrature);
        const double* psi_adj_u = FacePsi(adj_u, *au, gs, adj_quadrature);
        const double* psi_adj_d = FacePsi(adj_d, *ad, gs, adj_quadrature);
        for (std::uint64_t d = 0; d < num_dirs; ++d)
        {
          const auto weight = mu[d] * quadrature.weights[d];
          const auto da = direction_maps[gs][d];
          // Forward leaves the `_u` cell and the adjoint leaves the `_d` cell for mu > 0, and the
          // reverse for mu < 0.
          if (mu[d] > 0.0)
            outgoing +=
              weight *
              FaceProduct(mass, n, psi_u, nullptr, d, psi_adj_d, perm_ad, da, num_dirs, num_groups);
          else if (mu[d] < 0.0)
            incoming +=
              weight * FaceProduct(
                         mass, n, psi_d, &perm_fd, d, psi_adj_u, perm_au, da, num_dirs, num_groups);
        }
      }
    }
    extra += (fwd_d.GetNumFaces() - matched[0]) + (adj_u.GetNumFaces() - matched[1]) +
             (adj_d.GetNumFaces() - matched[2]);
  }

  std::vector<double> global_responses(local_responses.size(), 0.0);
  std::vector<std::uint64_t> global_counts(local_counts.size(), 0);
  comm.all_reduce(local_responses.data(),
                  static_cast<int>(local_responses.size()),
                  global_responses.data(),
                  mpi::op::sum<double>());
  comm.all_reduce(local_counts.data(),
                  static_cast<int>(local_counts.size()),
                  global_counts.data(),
                  mpi::op::sum<std::uint64_t>());

  // The counts are global, so every rank reaches the same decision.
  std::string message;
  for (size_t s = 0; s < num_surfaces; ++s)
  {
    const auto* counts = &global_counts[3 * s];
    if (counts[0] == 0 and counts[1] == 0 and counts[2] == 0)
      continue;
    message += " Surface " + selected_surfs[s].first + ": " + std::to_string(counts[0]) +
               " forward faces without a matching face, " + std::to_string(counts[1]) +
               " faces without a matching forward face, and " + std::to_string(counts[2]) +
               " matched faces with different node locations.";
  }
  OpenSnLogicalErrorIf(not message.empty(),
                       "The forward and adjoint surface data do not cover the same faces." +
                         message);

  std::vector<SurfaceResponse> surface_responses;
  surface_responses.reserve(num_surfaces);
  for (size_t s = 0; s < num_surfaces; ++s)
    surface_responses.push_back({selected_surfs[s].first,
                                 selected_surfs[s].second,
                                 global_responses[2 * s],
                                 global_responses[2 * s + 1]});
  return surface_responses;
}

} // namespace opensn
