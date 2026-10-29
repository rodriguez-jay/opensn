// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/response_evaluator/surface_response_data.h"
#include "framework/runtime.h"
#include "framework/utils/hdf_utils.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace opensn;

namespace
{

/*
 * Synthetic surface angular-flux files in the WriteSurfaceAngularFluxes format.
 *
 * Each surface tag lies on a plane x = const and consists of num_faces two-node segments of length
 * face_length along y, so the surface spans y in [0, surface_length]. The angular flux is linear in
 * y, so the face mass matrix face_length/6 [[2, 1], [1, 2]] integrates the product of two fluxes
 * exactly and the response has the closed form of ExpectedResponse.
 */
const int num_faces = 7;
const double face_length = 0.2;
const double surface_length = num_faces * face_length;
const std::vector<std::array<double, 3>> directions = {
  {0.6, 0.8, 0.0}, {-0.6, 0.8, 0.0}, {-0.6, -0.8, 0.0}, {0.6, -0.8, 0.0}};
const std::vector<double> weights = {0.1, 0.2, 0.3, 0.4};
/// Groups per groupset.
const std::vector<std::uint64_t> groups_per_groupset = {2, 1};

/// Angular flux a + b y of a data set (`adjoint`) on one side ('u', 'd', or 'b') of a surface.
std::array<double, 2>
FluxCoefficients(const bool adjoint,
                 const char side,
                 const std::array<double, 3>& omega,
                 const size_t gs,
                 const std::uint64_t g)
{
  const double s = side == 'u' ? 1.0 : (side == 'd' ? 2.0 : 3.0);
  const double a = (adjoint ? 0.5 : 1.0) + 0.3 * s + omega[0] + 0.7 * omega[1] +
                   0.1 * static_cast<double>(g) + 0.05 * static_cast<double>(gs);
  const double b = (adjoint ? -0.4 : 0.6) + 0.2 * s * omega[1] + 0.1 * static_cast<double>(g);
  return {a, b};
}

/// Integral over [0, surface_length] of (a1 + b1 y)(a2 + b2 y).
double
IntegrateProduct(const std::array<double, 2>& f, const std::array<double, 2>& g)
{
  return f[0] * g[0] * surface_length +
         (f[0] * g[1] + f[1] * g[0]) * surface_length * surface_length / 2.0 +
         f[1] * g[1] * surface_length * surface_length * surface_length / 3.0;
}

/**
 * Closed-form response of an interior surface with normal +x (`interior`) or a boundary with
 * outward normal +x.
 */
double
ExpectedResponse(const bool interior)
{
  double response = 0.0;
  for (size_t gs = 0; gs < groups_per_groupset.size(); ++gs)
    for (size_t d = 0; d < directions.size(); ++d)
    {
      const double mu = directions[d][0];
      for (std::uint64_t g = 0; g < groups_per_groupset[gs]; ++g)
      {
        if (not interior)
        {
          if (mu > 0.0)
            response += mu * weights[d] *
                        IntegrateProduct(FluxCoefficients(false, 'b', directions[d], gs, g),
                                         FluxCoefficients(true, 'b', directions[d], gs, g));
        }
        else if (mu > 0.0)
          response += mu * weights[d] *
                      IntegrateProduct(FluxCoefficients(false, 'u', directions[d], gs, g),
                                       FluxCoefficients(true, 'd', directions[d], gs, g));
        else
          response += mu * weights[d] *
                      IntegrateProduct(FluxCoefficients(false, 'd', directions[d], gs, g),
                                       FluxCoefficients(true, 'u', directions[d], gs, g));
      }
    }
  return response;
}

struct TagSpec
{
  std::string name;
  double x = 0.0;
  /// x-component of the stored face normal.
  double normal_x = 1.0;
  char side = 'u';
  /// Faces written to each file.
  std::vector<std::vector<int>> faces_per_file;
};

struct SetSpec
{
  bool adjoint = false;
  /// Omit attributes that earlier versions of OpenSn did not write.
  bool write_num_partitions = true;
  bool write_adjoint = true;
  double centroid_spacing = 1.0e-9;
  /// Store each face's nodes in reverse order.
  bool reverse_nodes = false;
  /// Store directions in reverse order.
  bool reverse_directions = false;
  std::vector<TagSpec> tags;
};

void
WriteSet(const std::string& file_base, const SetSpec& spec)
{
  const auto num_files = spec.tags.front().faces_per_file.size();
  for (size_t file_id = 0; file_id < num_files; ++file_id)
  {
    const auto file_name = file_base + std::to_string(file_id) + ".h5";
    const H5FileHandle file(H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    ASSERT_GE(file.Id(), 0);
    const auto id = file.Id();
    ASSERT_TRUE(H5CreateAttribute(id, "num_groupsets", std::uint64_t{groups_per_groupset.size()}));
    if (spec.write_adjoint)
      ASSERT_TRUE(H5CreateAttribute<bool>(id, "adjoint", spec.adjoint));
    if (spec.write_num_partitions)
      ASSERT_TRUE(H5CreateAttribute(id, "num_partitions", std::uint64_t{num_files}));
    for (size_t d = 0; d < 3; ++d)
      ASSERT_TRUE(H5CreateAttribute(id, "centroid_origin_" + std::to_string(d), 0.0));
    ASSERT_TRUE(H5CreateAttribute(id, "centroid_spacing", spec.centroid_spacing));
    ASSERT_TRUE(H5CreateGroup(id, "mesh"));

    for (const auto& tag : spec.tags)
    {
      const auto& faces = tag.faces_per_file[file_id];
      std::vector<std::uint64_t> cell_ids, num_face_nodes;
      std::vector<double> x, y, z;
      for (const auto face : faces)
      {
        cell_ids.push_back(static_cast<std::uint64_t>(face));
        num_face_nodes.push_back(2);
        std::array<double, 2> ys = {face * face_length, (face + 1) * face_length};
        if (spec.reverse_nodes)
          std::swap(ys[0], ys[1]);
        for (const auto value : ys)
        {
          x.push_back(tag.x);
          y.push_back(value);
          z.push_back(0.0);
        }
      }
      const auto mesh = "mesh/" + tag.name;
      ASSERT_TRUE(H5CreateGroup(id, mesh));
      ASSERT_TRUE(H5WriteDataset1D(id, mesh + "/cell_ids", cell_ids));
      ASSERT_TRUE(H5WriteDataset1D(id, mesh + "/num_face_nodes", num_face_nodes));
      ASSERT_TRUE(H5WriteDataset1D(id, mesh + "/nodes_x", x));
      ASSERT_TRUE(H5WriteDataset1D(id, mesh + "/nodes_y", y));
      ASSERT_TRUE(H5WriteDataset1D(id, mesh + "/nodes_z", z));
    }

    for (size_t gs = 0; gs < groups_per_groupset.size(); ++gs)
    {
      const auto group = "groupset_" + std::to_string(gs);
      ASSERT_TRUE(H5CreateGroup(id, group));
      ASSERT_TRUE(
        H5CreateAttribute(id, group + "/num_directions", std::uint64_t{directions.size()}));
      ASSERT_TRUE(H5CreateAttribute(id, group + "/num_groups", groups_per_groupset[gs]));
      for (const auto& tag : spec.tags)
      {
        std::vector<double> omega, mu, wt, fe_shape, psi, mass;
        for (const auto face : tag.faces_per_file[file_id])
        {
          std::array<double, 2> ys = {face * face_length, (face + 1) * face_length};
          if (spec.reverse_nodes)
            std::swap(ys[0], ys[1]);
          for (const auto node_y : ys)
            for (size_t k = 0; k < directions.size(); ++k)
            {
              const auto d = spec.reverse_directions ? directions.size() - 1 - k : k;
              omega.insert(omega.end(), directions[d].begin(), directions[d].end());
              mu.push_back(directions[d][0] * tag.normal_x);
              wt.push_back(weights[d]);
              fe_shape.push_back(face_length / 2.0);
              for (std::uint64_t g = 0; g < groups_per_groupset[gs]; ++g)
              {
                const auto c = FluxCoefficients(spec.adjoint, tag.side, directions[d], gs, g);
                psi.push_back(c[0] + c[1] * node_y);
              }
            }
          mass.insert(mass.end(),
                      {face_length / 3.0, face_length / 6.0, face_length / 6.0, face_length / 3.0});
        }
        const auto surf = group + "/" + tag.name;
        ASSERT_TRUE(H5CreateGroup(id, surf));
        ASSERT_TRUE(H5WriteDataset1D(id, surf + "/omega", omega));
        ASSERT_TRUE(H5WriteDataset1D(id, surf + "/mu", mu));
        ASSERT_TRUE(H5WriteDataset1D(id, surf + "/wt_d", wt));
        ASSERT_TRUE(H5WriteDataset1D(id, surf + "/fe_shape", fe_shape));
        ASSERT_TRUE(H5WriteDataset1D(id, surf + "/surf_flux", psi));
        ASSERT_TRUE(H5WriteDataset1D(id, surf + "/M_ij", mass));
      }
    }
  }
}

/// Forward set: five files, faces split unevenly, the two sides of the plane in different files.
SetSpec
ForwardSpec()
{
  SetSpec spec;
  spec.tags = {
    {"plane_u", 0.5, 1.0, 'u', {{0, 1, 2, 3, 4}, {}, {5}, {}, {6}}},
    {"plane_d", 0.5, -1.0, 'd', {{6}, {0, 1}, {}, {2, 3, 4, 5}, {}}},
    {"right", 1.0, 1.0, 'b', {{}, {0, 1, 2, 3, 4, 5, 6}, {}, {}, {}}},
  };
  return spec;
}

/// Adjoint set: three files with another split, reversed node order, and reordered directions.
SetSpec
AdjointSpec()
{
  SetSpec spec;
  spec.adjoint = true;
  spec.reverse_nodes = true;
  spec.reverse_directions = true;
  spec.tags = {
    {"plane_u", 0.5, 1.0, 'u', {{3, 4, 5, 6}, {0}, {1, 2}}},
    {"plane_d", 0.5, -1.0, 'd', {{}, {0, 1, 2, 3, 4, 5, 6}, {}}},
    {"right", 1.0, 1.0, 'b', {{5, 6}, {0, 1, 2}, {3, 4}}},
  };
  return spec;
}

class SurfaceResponseDataTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // All ranks use rank 0's process id so concurrent test runs do not share files.
    auto pid = static_cast<std::uint64_t>(getpid());
    mpi_comm.broadcast(pid, 0);
    directory_ =
      std::filesystem::temp_directory_path() / ("opensn_surface_response_" + std::to_string(pid));
    if (mpi_comm.rank() == 0)
      std::filesystem::create_directories(directory_);
    mpi_comm.barrier();
  }

  void TearDown() override
  {
    mpi_comm.barrier();
    if (mpi_comm.rank() == 0)
      std::filesystem::remove_all(directory_);
  }

  /// Write a set on rank 0 and return its file base.
  std::string Write(const std::string& name, const SetSpec& spec)
  {
    const auto file_base = (directory_ / (name + "_p")).string();
    if (mpi_comm.rank() == 0)
      WriteSet(file_base, spec);
    mpi_comm.barrier();
    return file_base;
  }

  static std::uint64_t GlobalFaces(const DistributedSurfaceData& data, const std::string& tag)
  {
    auto faces = data.surfaces.at(tag).GetNumFaces();
    mpi_comm.all_reduce(faces, mpi::op::sum<std::uint64_t>());
    return faces;
  }

  std::filesystem::path directory_;
};

const std::vector<std::string> all_tags = {"plane_u", "plane_d", "right"};

} // namespace

TEST_F(SurfaceResponseDataTest, MatchesClosedFormAcrossPartitions)
{
  const auto forward_base = Write("forward", ForwardSpec());
  const auto adjoint_base = Write("adjoint", AdjointSpec());

  const auto forward = ReadDistributedSurfaceAngularFluxes(forward_base, all_tags, mpi_comm);
  const auto adjoint = ReadDistributedSurfaceAngularFluxes(adjoint_base, all_tags, mpi_comm);
  EXPECT_EQ(forward.num_partitions, 5u);
  EXPECT_EQ(adjoint.num_partitions, 3u);
  EXPECT_FALSE(forward.adjoint);
  EXPECT_TRUE(adjoint.adjoint);
  for (const auto& tag : all_tags)
  {
    EXPECT_EQ(GlobalFaces(forward, tag), static_cast<std::uint64_t>(num_faces));
    EXPECT_EQ(GlobalFaces(adjoint, tag), static_cast<std::uint64_t>(num_faces));
  }

  const auto surface_responses = ComputeSurfaceResponses(forward, adjoint, {}, mpi_comm);
  ASSERT_EQ(surface_responses.size(), 2u);
  // Name order: "plane" (interior) before "right" (boundary).
  EXPECT_EQ(surface_responses[0].name, "plane");
  EXPECT_TRUE(surface_responses[0].interior);
  EXPECT_EQ(surface_responses[1].name, "right");
  EXPECT_FALSE(surface_responses[1].interior);
  EXPECT_EQ(surface_responses[1].incoming, 0.0);

  // Sums of a few hundred O(1) products: allow a small multiple of machine precision.
  const double interior = ExpectedResponse(true);
  const double boundary = ExpectedResponse(false);
  EXPECT_NEAR(surface_responses[0].outgoing + surface_responses[0].incoming,
              interior,
              1.0e-13 * std::abs(interior));
  EXPECT_NEAR(surface_responses[1].outgoing, boundary, 1.0e-13 * std::abs(boundary));
  // Both interior directions contribute, with opposite signs of mu.
  EXPECT_GT(surface_responses[0].outgoing, 0.0);
  EXPECT_LT(surface_responses[0].incoming, 0.0);

  // Selecting one surface gives the same surface response.
  const auto selected = ComputeSurfaceResponses(forward, adjoint, {"right"}, mpi_comm);
  ASSERT_EQ(selected.size(), 1u);
  EXPECT_DOUBLE_EQ(selected[0].outgoing, surface_responses[1].outgoing);
}

TEST_F(SurfaceResponseDataTest, SmallSendBudgetGivesSameResult)
{
  const auto forward_base = Write("forward", ForwardSpec());
  const auto adjoint_base = Write("adjoint", AdjointSpec());

  // A one-byte budget sends one face per rank per round.
  const auto forward = ReadDistributedSurfaceAngularFluxes(forward_base, all_tags, mpi_comm, 1);
  const auto adjoint = ReadDistributedSurfaceAngularFluxes(adjoint_base, all_tags, mpi_comm, 1);
  const auto small = ComputeSurfaceResponses(forward, adjoint, {}, mpi_comm);

  const auto forward_default =
    ReadDistributedSurfaceAngularFluxes(forward_base, all_tags, mpi_comm);
  const auto adjoint_default =
    ReadDistributedSurfaceAngularFluxes(adjoint_base, all_tags, mpi_comm);
  const auto reference = ComputeSurfaceResponses(forward_default, adjoint_default, {}, mpi_comm);

  // Owned faces are evaluated in key order, so the chunking does not change the sum.
  ASSERT_EQ(small.size(), reference.size());
  for (size_t s = 0; s < small.size(); ++s)
  {
    EXPECT_EQ(small[s].outgoing, reference[s].outgoing);
    EXPECT_EQ(small[s].incoming, reference[s].incoming);
  }
}

TEST_F(SurfaceResponseDataTest, RejectsFilesFromEarlierVersions)
{
  auto without_partitions = ForwardSpec();
  without_partitions.write_num_partitions = false;
  EXPECT_THROW(ReadDistributedSurfaceAngularFluxes(
                 Write("no_partitions", without_partitions), all_tags, mpi_comm),
               std::logic_error);

  auto without_adjoint = AdjointSpec();
  without_adjoint.write_adjoint = false;
  EXPECT_THROW(
    ReadDistributedSurfaceAngularFluxes(Write("no_adjoint", without_adjoint), all_tags, mpi_comm),
    std::logic_error);
}

TEST_F(SurfaceResponseDataTest, RejectsMissingFace)
{
  const auto forward_base = Write("forward", ForwardSpec());
  auto spec = AdjointSpec();
  spec.tags[1].faces_per_file = {{}, {0, 1, 2, 3, 5, 6}, {}};
  const auto adjoint_base = Write("adjoint", spec);

  const auto forward = ReadDistributedSurfaceAngularFluxes(forward_base, all_tags, mpi_comm);
  const auto adjoint = ReadDistributedSurfaceAngularFluxes(adjoint_base, all_tags, mpi_comm);
  EXPECT_THROW(ComputeSurfaceResponses(forward, adjoint, {"plane"}, mpi_comm), std::logic_error);
  // The boundary is complete.
  EXPECT_NO_THROW(ComputeSurfaceResponses(forward, adjoint, {"right"}, mpi_comm));
}

TEST_F(SurfaceResponseDataTest, RejectsDuplicateFace)
{
  auto spec = ForwardSpec();
  spec.tags[0].faces_per_file = {{0, 1, 2, 3, 4}, {2}, {5}, {}, {6}};
  const auto forward_base = Write("forward", spec);
  EXPECT_THROW(ReadDistributedSurfaceAngularFluxes(forward_base, all_tags, mpi_comm),
               std::logic_error);
}

TEST_F(SurfaceResponseDataTest, RejectsMissingTagAndFiles)
{
  const auto forward_base = Write("forward", ForwardSpec());
  EXPECT_THROW(ReadDistributedSurfaceAngularFluxes(forward_base, {"absent"}, mpi_comm),
               std::logic_error);
  EXPECT_THROW(
    ReadDistributedSurfaceAngularFluxes((directory_ / "absent_p").string(), all_tags, mpi_comm),
    std::logic_error);
  EXPECT_THROW(ReadDistributedSurfaceAngularFluxes(forward_base, {}, mpi_comm),
               std::invalid_argument);
}

TEST_F(SurfaceResponseDataTest, RejectsIncompatibleDataSets)
{
  const auto forward_base = Write("forward", ForwardSpec());
  auto spec = AdjointSpec();
  spec.centroid_spacing = 2.0e-9;
  const auto adjoint_base = Write("adjoint", spec);

  const auto forward = ReadDistributedSurfaceAngularFluxes(forward_base, all_tags, mpi_comm);
  const auto adjoint = ReadDistributedSurfaceAngularFluxes(adjoint_base, all_tags, mpi_comm);
  // Keys of different spacings cannot be matched.
  EXPECT_THROW(ComputeSurfaceResponses(forward, adjoint, {}, mpi_comm), std::invalid_argument);
  // Forward data in the adjoint role.
  EXPECT_THROW(ComputeSurfaceResponses(forward, forward, {}, mpi_comm), std::invalid_argument);
  // Unknown surface.
  EXPECT_THROW(ComputeSurfaceResponses(forward, adjoint, {"left"}, mpi_comm),
               std::invalid_argument);

  // One side of an interior surface alone is rejected rather than treated as a boundary.
  const auto lone_forward =
    ReadDistributedSurfaceAngularFluxes(forward_base, {"plane_u"}, mpi_comm);
  const auto lone_adjoint =
    ReadDistributedSurfaceAngularFluxes(Write("adjoint_ok", AdjointSpec()), {"plane_u"}, mpi_comm);
  EXPECT_THROW(ComputeSurfaceResponses(lone_forward, lone_adjoint, {}, mpi_comm),
               std::invalid_argument);
}

TEST_F(SurfaceResponseDataTest, MatchesDirectionsRelativeToQuadratureSpacing)
{
  const auto forward =
    ReadDistributedSurfaceAngularFluxes(Write("forward", ForwardSpec()), all_tags, mpi_comm);
  const auto adjoint =
    ReadDistributedSurfaceAngularFluxes(Write("adjoint", AdjointSpec()), all_tags, mpi_comm);
  // The nearest two fixture directions are 1.2 apart, so directions match within 1.2e-6.
  const auto Shift = [](DistributedSurfaceData data, const std::uint64_t d, const double delta)
  {
    for (auto& quadrature : data.groupsets)
      quadrature.omegas[3 * d] += delta;
    return data;
  };
  const double interior = ExpectedResponse(true);
  const double boundary = ExpectedResponse(false);

  // Within the tolerance: every direction, stored in reverse order, still finds its match.
  auto close = adjoint;
  for (auto& quadrature : close.groupsets)
    for (auto& component : quadrature.omegas)
      component += 1.0e-9;
  const auto surface_responses = ComputeSurfaceResponses(forward, close, {}, mpi_comm);
  ASSERT_EQ(surface_responses.size(), 2u);
  EXPECT_NEAR(surface_responses[0].outgoing + surface_responses[0].incoming,
              interior,
              1.0e-13 * std::abs(interior));
  EXPECT_NEAR(surface_responses[1].outgoing, boundary, 1.0e-13 * std::abs(boundary));

  // Beyond the tolerance.
  EXPECT_THROW(ComputeSurfaceResponses(forward, Shift(adjoint, 0, 1.2e-4), {}, mpi_comm),
               std::invalid_argument);

  // Duplicate adjoint directions.
  auto duplicate = adjoint;
  for (auto& quadrature : duplicate.groupsets)
    std::copy_n(quadrature.omegas.begin(), 3, quadrature.omegas.begin() + 3);
  EXPECT_THROW(ComputeSurfaceResponses(forward, duplicate, {}, mpi_comm), std::invalid_argument);
}

TEST_F(SurfaceResponseDataTest, MatchesCloselySpacedDirections)
{
  auto forward =
    ReadDistributedSurfaceAngularFluxes(Write("forward", ForwardSpec()), all_tags, mpi_comm);
  auto adjoint =
    ReadDistributedSurfaceAngularFluxes(Write("adjoint", AdjointSpec()), all_tags, mpi_comm);
  // Place fixture direction 1 1e-12 from direction 0 in both sets. The adjoint stores directions in
  // reverse order, so there they are adjoint directions 2 and 3. Only the direction map reads the
  // vectors, so the closed-form response is unchanged.
  const auto Place = [](DistributedSurfaceData& data, const std::uint64_t d, const std::uint64_t to)
  {
    for (auto& quadrature : data.groupsets)
    {
      std::copy_n(quadrature.omegas.begin() + 3 * to, 3, quadrature.omegas.begin() + 3 * d);
      quadrature.omegas[3 * d + 2] += 1.0e-12;
    }
  };
  Place(forward, 1, 0);
  Place(adjoint, 2, 3);

  const auto surface_responses = ComputeSurfaceResponses(forward, adjoint, {}, mpi_comm);
  ASSERT_EQ(surface_responses.size(), 2u);
  const double interior = ExpectedResponse(true);
  const double boundary = ExpectedResponse(false);
  EXPECT_NEAR(surface_responses[0].outgoing + surface_responses[0].incoming,
              interior,
              1.0e-13 * std::abs(interior));
  EXPECT_NEAR(surface_responses[1].outgoing, boundary, 1.0e-13 * std::abs(boundary));

  // At this spacing, a 1e-15 difference is 1e-3 of the spacing and no longer a match.
  for (auto& quadrature : adjoint.groupsets)
    quadrature.omegas[3 * 2] += 1.0e-15;
  EXPECT_THROW(ComputeSurfaceResponses(forward, adjoint, {}, mpi_comm), std::invalid_argument);
}

TEST(SurfaceFaceOwnerTest, IsInRangeAndDeterministic)
{
  for (int num_ranks : {1, 2, 3, 7})
    for (std::int64_t k = -5; k < 5; ++k)
    {
      const DiscreteOrdinatesProblemIO::FaceCentroidKey key = {k, 3 * k, -k};
      const auto owner = SurfaceFaceOwner(key, num_ranks);
      EXPECT_GE(owner, 0);
      EXPECT_LT(owner, num_ranks);
      EXPECT_EQ(owner, SurfaceFaceOwner(key, num_ranks));
    }
}
