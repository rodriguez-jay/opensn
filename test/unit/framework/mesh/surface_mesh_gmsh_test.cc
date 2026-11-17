// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/surface_mesh/surface_mesh.h"
#include "framework/mesh/logical_volume/surface_mesh_logical_volume.h"
#include "framework/parameters/parameter_block.h"
#include "framework/runtime.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace opensn;

namespace
{

struct NamedElements
{
  std::string name;
  /// Node ids (1-based) of each element of the group.
  std::vector<std::vector<int>> elements;
};

/// A gmsh 2.2 file in the temporary directory, removed on destruction.
class TempMesh
{
public:
  explicit TempMesh(const std::string& contents)
  {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("opensn_surface_mesh_gmsh_test_r" + std::to_string(mpi_comm.rank()) + "_" +
             std::to_string(now) + ".msh");
    std::ofstream(path_) << contents;
  }
  ~TempMesh() { std::filesystem::remove(path_); }
  std::string Path() const { return path_.string(); }

private:
  std::filesystem::path path_;
};

/// Writes a gmsh 2.2 mesh with the given nodes, cells of one physical group "Domain", and named
/// groups of boundary elements (lines in 2D, quadrilaterals in 3D).
std::string
WriteMsh(const std::vector<std::array<double, 3>>& nodes,
         const int cell_type,
         const std::vector<std::vector<int>>& cells,
         const int boundary_type,
         const std::vector<NamedElements>& groups)
{
  const int dim = cell_type == 5 ? 3 : 2;
  std::ostringstream out;
  out << "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n" << groups.size() + 1 << "\n";
  for (size_t g = 0; g < groups.size(); ++g)
    out << dim - 1 << " " << g + 10 << " \"" << groups[g].name << "\"\n";
  out << dim << " 1 \"Domain\"\n$EndPhysicalNames\n$Nodes\n" << nodes.size() << "\n";
  for (size_t n = 0; n < nodes.size(); ++n)
    out << n + 1 << " " << nodes[n][0] << " " << nodes[n][1] << " " << nodes[n][2] << "\n";
  size_t num_elements = cells.size();
  for (const auto& group : groups)
    num_elements += group.elements.size();
  out << "$EndNodes\n$Elements\n" << num_elements << "\n";
  size_t id = 0;
  for (size_t g = 0; g < groups.size(); ++g)
    for (const auto& element : groups[g].elements)
    {
      out << ++id << " " << boundary_type << " 2 " << g + 10 << " " << g + 10;
      for (const auto v : element)
        out << " " << v;
      out << "\n";
    }
  for (const auto& cell : cells)
  {
    out << ++id << " " << cell_type << " 2 1 1";
    for (const auto v : cell)
      out << " " << v;
    out << "\n";
  }
  out << "$EndElements\n";
  return out.str();
}

/// A 2D mesh of n x n unit quadrilaterals on [0, n]^2 with the given named line groups.
std::string
QuadMesh(const int n, const std::vector<NamedElements>& groups)
{
  std::vector<std::array<double, 3>> nodes;
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i)
      nodes.push_back({double(i), double(j), 0.0});
  const auto id = [n](const int i, const int j) { return j * (n + 1) + i + 1; };
  std::vector<std::vector<int>> cells;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      cells.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
  return WriteMsh(nodes, 3, cells, 1, groups);
}

/// The unit line segments of a closed polygon on the integer lattice of QuadMesh(n), with every
/// other segment reversed so that the stored orientation is inconsistent.
NamedElements
Loop(const std::string& name, const int n, const std::vector<std::pair<int, int>>& corners)
{
  const auto id = [n](const int i, const int j) { return j * (n + 1) + i + 1; };
  NamedElements group{name, {}};
  for (size_t c = 0; c < corners.size(); ++c)
  {
    auto [i, j] = corners[c];
    const auto [i1, j1] = corners[(c + 1) % corners.size()];
    while (i != i1 or j != j1)
    {
      const int ni = i + (i1 > i) - (i1 < i);
      const int nj = j + (j1 > j) - (j1 < j);
      if (group.elements.size() % 2 == 0)
        group.elements.push_back({id(i, j), id(ni, nj)});
      else
        group.elements.push_back({id(ni, nj), id(i, j)});
      i = ni;
      j = nj;
    }
  }
  return group;
}

/// A 3D mesh of n x n x n unit hexahedra on [0, n]^3 with the given named quadrilateral groups.
std::string
HexMesh(const int n, const std::vector<NamedElements>& groups)
{
  std::vector<std::array<double, 3>> nodes;
  for (int k = 0; k <= n; ++k)
    for (int j = 0; j <= n; ++j)
      for (int i = 0; i <= n; ++i)
        nodes.push_back({double(i), double(j), double(k)});
  const auto id = [n](const int i, const int j, const int k)
  { return (k * (n + 1) + j) * (n + 1) + i + 1; };
  std::vector<std::vector<int>> cells;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        cells.push_back({id(i, j, k),
                         id(i + 1, j, k),
                         id(i + 1, j + 1, k),
                         id(i, j + 1, k),
                         id(i, j, k + 1),
                         id(i + 1, j, k + 1),
                         id(i + 1, j + 1, k + 1),
                         id(i, j + 1, k + 1)});
  return WriteMsh(nodes, 5, cells, 3, groups);
}

/// The unit quadrilaterals of the faces of the box [lo, hi] on the integer lattice of HexMesh(n),
/// with every other quadrilateral reversed so that the stored orientation is inconsistent.
NamedElements
Box(const std::string& name,
    const int n,
    const std::array<int, 3>& lo,
    const std::array<int, 3>& hi)
{
  const auto id = [n](const std::array<int, 3>& p)
  { return (p[2] * (n + 1) + p[1]) * (n + 1) + p[0] + 1; };
  NamedElements group{name, {}};
  for (int axis = 0; axis < 3; ++axis)
  {
    const int u = (axis + 1) % 3;
    const int v = (axis + 2) % 3;
    for (const int side : {lo[axis], hi[axis]})
      for (int a = lo[u]; a < hi[u]; ++a)
        for (int b = lo[v]; b < hi[v]; ++b)
        {
          std::array<int, 3> p{};
          p[axis] = side;
          std::vector<int> quad;
          constexpr std::array<std::pair<int, int>, 4> corners{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
          for (const auto& [du, dv] : corners)
          {
            p[u] = a + du;
            p[v] = b + dv;
            quad.push_back(id(p));
          }
          if (group.elements.size() % 2 == 1)
            std::reverse(quad.begin(), quad.end());
          group.elements.push_back(quad);
        }
  }
  return group;
}

std::shared_ptr<SurfaceMeshLogicalVolume>
VolumeFromGroups(const std::string& file_name, const std::vector<std::string>& names)
{
  auto surface_mesh = std::make_shared<SurfaceMesh>();
  surface_mesh->ImportFromGmshPhysicalGroups(file_name, names);
  ParameterBlock block("");
  block.AddParameter(ParameterBlock("surface_mesh", surface_mesh));
  auto params = SurfaceMeshLogicalVolume::GetInputParameters();
  params.AssignParameters(block);
  return std::make_shared<SurfaceMeshLogicalVolume>(params);
}

/// Checks Inside against the expected region at the points of an offset lattice that avoids the
/// mesh lines, in the plane z = 0 (2D) or throughout [0, n]^3 (3D).
void
ExpectInside(const LogicalVolume& volume,
             const int n,
             const bool three_d,
             const std::function<bool(double, double, double)>& expected)
{
  constexpr int per_cell = 4;
  const int num_points = n * per_cell;
  const auto coordinate = [](const int m) { return (m + 0.37) / per_cell; };
  int num_inside = 0;
  for (int i = 0; i < num_points; ++i)
    for (int j = 0; j < num_points; ++j)
      for (int k = 0; k < (three_d ? num_points : 1); ++k)
      {
        const Vector3 p(coordinate(i), coordinate(j), three_d ? coordinate(k) : 0.0);
        const bool inside = volume.Inside(p);
        num_inside += inside;
        EXPECT_EQ(inside, expected(p.x, p.y, p.z))
          << "at (" << p.x << ", " << p.y << ", " << p.z << ")";
      }
  EXPECT_GT(num_inside, 0);
}

bool
InBox(const double x, const double y, const double x0, const double x1, const double y0, double y1)
{
  return x > x0 and x < x1 and y > y0 and y < y1;
}

} // namespace

TEST(SurfaceMeshGmshTest, Loop2DWithMixedOrientation)
{
  constexpr int n = 6;
  const TempMesh mesh(QuadMesh(n, {Loop("Box", n, {{1, 1}, {3, 1}, {3, 4}, {1, 4}})}));
  const auto volume = VolumeFromGroups(mesh.Path(), {"Box"});
  ExpectInside(
    *volume, n, false, [](double x, double y, double) { return InBox(x, y, 1, 3, 1, 4); });
}

TEST(SurfaceMeshGmshTest, Loops2DWithHole)
{
  constexpr int n = 8;
  const TempMesh mesh(QuadMesh(n,
                               {Loop("Outer", n, {{1, 1}, {7, 1}, {7, 7}, {1, 7}}),
                                Loop("Hole", n, {{3, 3}, {3, 5}, {5, 5}, {5, 3}})}));
  // The groups may be listed in any order.
  const auto volume = VolumeFromGroups(mesh.Path(), {"Hole", "Outer"});
  ExpectInside(*volume,
               n,
               false,
               [](double x, double y, double)
               { return InBox(x, y, 1, 7, 1, 7) and not InBox(x, y, 3, 5, 3, 5); });
}

TEST(SurfaceMeshGmshTest, NonConvexLoop2D)
{
  constexpr int n = 6;
  const TempMesh mesh(
    QuadMesh(n, {Loop("L", n, {{1, 1}, {5, 1}, {5, 3}, {3, 3}, {3, 5}, {1, 5}})}));
  const auto volume = VolumeFromGroups(mesh.Path(), {"L"});
  ExpectInside(*volume,
               n,
               false,
               [](double x, double y, double)
               { return InBox(x, y, 1, 5, 1, 3) or InBox(x, y, 1, 3, 1, 5); });
}

TEST(SurfaceMeshGmshTest, SurfacesFromSeveralGroups2D)
{
  constexpr int n = 6;
  const auto loop = Loop("Box", n, {{1, 1}, {4, 1}, {4, 4}, {1, 4}});
  // Split the loop into two groups, as a detector's sides would be.
  NamedElements first{"First", {}}, second{"Second", {}};
  for (size_t s = 0; s < loop.elements.size(); ++s)
    (s < 5 ? first : second).elements.push_back(loop.elements[s]);
  const TempMesh mesh(QuadMesh(n, {first, second}));
  const auto volume = VolumeFromGroups(mesh.Path(), {"First", "Second"});
  ExpectInside(
    *volume, n, false, [](double x, double y, double) { return InBox(x, y, 1, 4, 1, 4); });
  // One group alone is not closed.
  SurfaceMesh surface_mesh;
  EXPECT_THROW(surface_mesh.ImportFromGmshPhysicalGroups(mesh.Path(), {"First"}),
               std::invalid_argument);
}

TEST(SurfaceMeshGmshTest, Box3DWithMixedOrientation)
{
  constexpr int n = 6;
  const TempMesh mesh(HexMesh(n, {Box("Box", n, {1, 1, 2}, {4, 3, 5})}));
  const auto volume = VolumeFromGroups(mesh.Path(), {"Box"});
  ExpectInside(*volume,
               n,
               true,
               [](double x, double y, double z)
               { return InBox(x, y, 1, 4, 1, 3) and z > 2 and z < 5; });
}

TEST(SurfaceMeshGmshTest, Shell3DWithHole)
{
  constexpr int n = 6;
  const TempMesh mesh(
    HexMesh(n, {Box("Outer", n, {1, 1, 1}, {5, 5, 5}), Box("Hole", n, {2, 2, 2}, {4, 4, 4})}));
  const auto volume = VolumeFromGroups(mesh.Path(), {"Outer", "Hole"});
  const auto InCube = [](double x, double y, double z, double lo, double hi)
  { return x > lo and x < hi and y > lo and y < hi and z > lo and z < hi; };
  ExpectInside(*volume,
               n,
               true,
               [&](double x, double y, double z)
               { return InCube(x, y, z, 1, 5) and not InCube(x, y, z, 2, 4); });
}

TEST(SurfaceMeshGmshTest, InvalidRequests)
{
  constexpr int n = 6;
  auto open = Loop("Open", n, {{1, 1}, {3, 1}, {3, 3}, {1, 3}});
  open.elements.pop_back();
  const TempMesh mesh(QuadMesh(n, {Loop("Box", n, {{1, 1}, {3, 1}, {3, 3}, {1, 3}}), open}));

  // The 2 x 2 box has 8 segments, each extruded into 2 triangles over 2 new vertices per node.
  SurfaceMesh surface_mesh;
  surface_mesh.ImportFromGmshPhysicalGroups(mesh.Path(), {"Box"});
  ASSERT_EQ(surface_mesh.GetTriangles().size(), 2 * 8);
  ASSERT_EQ(surface_mesh.GetVertices().size(), 2 * 8);

  EXPECT_THROW(surface_mesh.ImportFromGmshPhysicalGroups(mesh.Path(), {}), std::invalid_argument);
  EXPECT_THROW(surface_mesh.ImportFromGmshPhysicalGroups(mesh.Path(), {"Missing"}),
               std::invalid_argument);
  EXPECT_THROW(surface_mesh.ImportFromGmshPhysicalGroups(mesh.Path(), {"Domain"}),
               std::invalid_argument);
  EXPECT_THROW(surface_mesh.ImportFromGmshPhysicalGroups(mesh.Path(), {"Open"}),
               std::invalid_argument);
  // Two loops that share their edges do not form closed loops.
  EXPECT_THROW(surface_mesh.ImportFromGmshPhysicalGroups(mesh.Path(), {"Box", "Open"}),
               std::invalid_argument);
  EXPECT_THROW(surface_mesh.ImportFromGmshPhysicalGroups("does_not_exist.msh", {"Box"}),
               std::runtime_error);

  // Failed imports leave the surface mesh unchanged.
  EXPECT_EQ(surface_mesh.GetTriangles().size(), 2 * 8);
  EXPECT_EQ(surface_mesh.GetVertices().size(), 2 * 8);
}
