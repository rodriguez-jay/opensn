// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/surface_mesh/surface_mesh.h"
#include "framework/mesh/io/mesh_io.h"
#include "framework/logging/log.h"
#include "framework/runtime.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace opensn
{

namespace
{

/// Signed area of a closed 2D loop given as oriented segments, in the xy-plane.
double
SignedArea(const std::vector<std::array<uint64_t, 2>>& loop, const std::vector<Vector3>& vertices)
{
  double area = 0.0;
  for (const auto& [a, b] : loop)
    area += vertices[a].x * vertices[b].y - vertices[b].x * vertices[a].y;
  return 0.5 * area;
}

/// Whether a point lies inside a closed 2D loop, by the even-odd crossing rule in the xy-plane.
bool
InsideLoop(const Vector3& point,
           const std::vector<std::array<uint64_t, 2>>& loop,
           const std::vector<Vector3>& vertices)
{
  bool inside = false;
  for (const auto& [a, b] : loop)
  {
    const auto& p = vertices[a];
    const auto& q = vertices[b];
    if ((p.y > point.y) != (q.y > point.y))
    {
      const double x_cross = p.x + (point.y - p.y) * (q.x - p.x) / (q.y - p.y);
      if (point.x < x_cross)
        inside = not inside;
    }
  }
  return inside;
}

/// Signed volume enclosed by a closed, consistently oriented triangulated surface.
double
SignedVolume(const std::vector<std::array<uint64_t, 3>>& triangles,
             const std::vector<Vector3>& vertices)
{
  double volume = 0.0;
  for (const auto& [a, b, c] : triangles)
    volume += vertices[a].Dot(vertices[b].Cross(vertices[c]));
  return volume / 6.0;
}

/// Whether a point lies inside a closed triangulated surface, by the parity of the crossings of a
/// ray with a fixed oblique direction (Moller-Trumbore intersection).
bool
InsideShell(const Vector3& point,
            const std::vector<std::array<uint64_t, 3>>& triangles,
            const std::vector<Vector3>& vertices)
{
  const Vector3 direction = Vector3(0.5773, 0.6142, 0.5381).Normalized();
  bool inside = false;
  for (const auto& [a, b, c] : triangles)
  {
    const Vector3 e1 = vertices[b] - vertices[a];
    const Vector3 e2 = vertices[c] - vertices[a];
    const Vector3 p = direction.Cross(e2);
    const double det = e1.Dot(p);
    if (std::abs(det) < 1.0e-14 * e1.Norm() * e2.Norm())
      continue;
    const Vector3 s = point - vertices[a];
    const double u = s.Dot(p) / det;
    if (u < 0.0 or u > 1.0)
      continue;
    const Vector3 q = s.Cross(e1);
    const double v = direction.Dot(q) / det;
    if (v < 0.0 or u + v > 1.0)
      continue;
    if (e2.Dot(q) / det > 0.0)
      inside = not inside;
  }
  return inside;
}

/// Number of other components that contain a component. A component nested in an even number of
/// others bounds a region from outside; one nested in an odd number bounds a hole.
template <typename Contains>
std::vector<size_t>
NestingDepths(const size_t num_components, const Contains& contains)
{
  std::vector<size_t> depths(num_components, 0);
  for (size_t c = 0; c < num_components; ++c)
    for (size_t other = 0; other < num_components; ++other)
      if (other != c and contains(other, c))
        ++depths[c];
  return depths;
}

} // namespace

void
SurfaceMesh::ImportFromGmshPhysicalGroups(const std::string& file_name,
                                          const std::vector<std::string>& physical_names)
{
  const std::string fname = "SurfaceMesh::ImportFromGmshPhysicalGroups";
  if (physical_names.empty())
    throw std::invalid_argument(fname + ": No physical group names were given.");

  UnpartitionedMesh::Options options;
  options.file_name = file_name;
  const auto umesh = MeshIO::FromGmsh(options);
  const auto dimension = umesh->GetDimension();
  if (dimension != 2 and dimension != 3)
    throw std::logic_error(fname + ": " + file_name + " is not a 2D or 3D mesh.");

  // Physical groups of dimension one less than the mesh are in the boundary-name map, and their
  // elements are the raw boundary cells, whether they lie on the mesh boundary or inside it.
  const auto& name_map = umesh->GetBoundaryNameMap();
  std::set<uint64_t> group_ids;
  for (const auto& name : physical_names)
  {
    const auto it = name_map.find(name);
    if (it == name_map.end())
      throw std::invalid_argument(fname + ": " + file_name + " has no physical group named \"" +
                                  name + "\" of dimension " + std::to_string(dimension - 1) + ".");
    group_ids.insert(it->second);
  }

  std::vector<std::vector<uint64_t>> elements;
  for (const auto& cell : umesh->GetRawBoundaryCells())
    if (group_ids.count(cell->block_id) > 0)
      elements.push_back(cell->vertex_ids);
  if (elements.empty())
    throw std::invalid_argument(fname + ": The requested physical groups of " + file_name +
                                " contain no elements.");

  const auto& mesh_vertices = umesh->GetVertices();
  std::vector<Vector3> new_vertices;
  std::vector<std::array<uint64_t, 3>> triangles;

  if (dimension == 2)
  {
    // Each vertex of a closed loop belongs to exactly two segments.
    std::map<uint64_t, std::vector<size_t>> vertex_segments;
    for (size_t s = 0; s < elements.size(); ++s)
    {
      const auto& segment = elements[s];
      if (segment.size() != 2 or segment[0] == segment[1])
        throw std::logic_error(fname + ": The requested physical groups contain an element that "
                                       "is not a line segment.");
      for (const auto v : segment)
        vertex_segments[v].push_back(s);
    }
    for (const auto& [v, segments] : vertex_segments)
      if (segments.size() != 2)
        throw std::invalid_argument(
          fname + ": The requested physical groups do not form closed loops: a vertex belongs to " +
          std::to_string(segments.size()) + " segments instead of 2.");

    // Walk each loop, orienting every segment to start where the previous one ends.
    std::vector<std::vector<std::array<uint64_t, 2>>> loops;
    std::vector<bool> visited(elements.size(), false);
    for (size_t start = 0; start < elements.size(); ++start)
    {
      if (visited[start])
        continue;
      std::vector<std::array<uint64_t, 2>> loop;
      size_t s = start;
      uint64_t from = elements[start][0];
      while (not visited[s])
      {
        visited[s] = true;
        const uint64_t to = elements[s][0] == from ? elements[s][1] : elements[s][0];
        loop.push_back({from, to});
        const auto& next = vertex_segments.at(to);
        s = next[0] == s ? next[1] : next[0];
        from = to;
      }
      loops.push_back(std::move(loop));
    }

    // Outer loops run counterclockwise and holes clockwise, so that the outward normal, the
    // tangent rotated clockwise, points out of the enclosed region.
    const auto depths = NestingDepths(
      loops.size(),
      [&](const size_t outer, const size_t inner)
      { return InsideLoop(mesh_vertices[loops[inner][0][0]], loops[outer], mesh_vertices); });
    for (size_t l = 0; l < loops.size(); ++l)
    {
      const double area = SignedArea(loops[l], mesh_vertices);
      if (area == 0.0)
        throw std::invalid_argument(fname + ": A loop of the requested physical groups encloses "
                                            "no area.");
      const bool counterclockwise = area > 0.0;
      const bool hole = depths[l] % 2 == 1;
      if (counterclockwise == hole)
      {
        std::reverse(loops[l].begin(), loops[l].end());
        for (auto& segment : loops[l])
          std::swap(segment[0], segment[1]);
      }
    }

    // Extrude each segment into a wall of two triangles spanning z0 +/- h, so that the surface
    // encloses a prism around the loops in the mesh plane z0. Points tested by
    // SurfaceMeshLogicalVolume::Inside lie in that plane.
    Vector3 lo(mesh_vertices[loops[0][0][0]]), hi(lo);
    double z0 = 0.0;
    size_t num_loop_vertices = 0;
    for (const auto& loop : loops)
      for (const auto& segment : loop)
      {
        const auto& p = mesh_vertices[segment[0]];
        lo = Vector3(std::min(lo.x, p.x), std::min(lo.y, p.y), 0.0);
        hi = Vector3(std::max(hi.x, p.x), std::max(hi.y, p.y), 0.0);
        z0 += p.z;
        ++num_loop_vertices;
      }
    z0 /= static_cast<double>(num_loop_vertices);
    const double h = 0.5 * std::max(hi.x - lo.x, hi.y - lo.y);

    std::map<uint64_t, uint64_t> bottom_index;
    const auto WallVertices = [&](const uint64_t v)
    {
      const auto [it, inserted] = bottom_index.try_emplace(v, new_vertices.size());
      if (inserted)
      {
        const auto& p = mesh_vertices[v];
        new_vertices.emplace_back(p.x, p.y, z0 - h);
        new_vertices.emplace_back(p.x, p.y, z0 + h);
      }
      return std::make_pair(it->second, it->second + 1);
    };
    for (const auto& loop : loops)
      for (const auto& [a, b] : loop)
      {
        const auto [a_lo, a_hi] = WallVertices(a);
        const auto [b_lo, b_hi] = WallVertices(b);
        // (b - a) x z points to the right of the segment, which is outward.
        triangles.push_back({a_lo, b_lo, b_hi});
        triangles.push_back({a_lo, b_hi, a_hi});
      }
  }
  else
  {
    // Each edge of a closed surface belongs to exactly two faces.
    std::map<std::pair<uint64_t, uint64_t>, std::vector<size_t>> edge_faces;
    for (size_t f = 0; f < elements.size(); ++f)
    {
      const auto& face = elements[f];
      if (face.size() != 3 and face.size() != 4)
        throw std::logic_error(fname + ": The requested physical groups contain an element that "
                                       "is not a triangle or quadrilateral.");
      for (size_t e = 0; e < face.size(); ++e)
      {
        const auto a = face[e];
        const auto b = face[(e + 1) % face.size()];
        edge_faces[{std::min(a, b), std::max(a, b)}].push_back(f);
      }
    }
    for (const auto& [edge, faces] : edge_faces)
      if (faces.size() != 2)
        throw std::invalid_argument(
          fname +
          ": The requested physical groups do not form closed surfaces: an edge belongs "
          "to " +
          std::to_string(faces.size()) + " faces instead of 2.");

    // Whether a face, in its stored vertex order, traverses the directed edge a -> b.
    const auto Traverses = [&](const size_t f, const uint64_t a, const uint64_t b)
    {
      const auto& face = elements[f];
      for (size_t e = 0; e < face.size(); ++e)
        if (face[e] == a and face[(e + 1) % face.size()] == b)
          return true;
      return false;
    };

    // Orient the faces of each connected component consistently: neighbors traverse their shared
    // edge in opposite directions.
    std::vector<int> flipped(elements.size(), -1);
    std::vector<std::vector<size_t>> components;
    for (size_t start = 0; start < elements.size(); ++start)
    {
      if (flipped[start] != -1)
        continue;
      flipped[start] = 0;
      std::vector<size_t> component = {start};
      for (size_t i = 0; i < component.size(); ++i)
      {
        const auto f = component[i];
        auto face = elements[f];
        if (flipped[f])
          std::reverse(face.begin(), face.end());
        for (size_t e = 0; e < face.size(); ++e)
        {
          const auto a = face[e];
          const auto b = face[(e + 1) % face.size()];
          const auto& shared = edge_faces.at({std::min(a, b), std::max(a, b)});
          const auto g = shared[0] == f ? shared[1] : shared[0];
          const int g_flipped = Traverses(g, a, b) ? 1 : 0;
          if (flipped[g] == -1)
          {
            flipped[g] = g_flipped;
            component.push_back(g);
          }
          else if (flipped[g] != g_flipped)
            throw std::invalid_argument(fname + ": The requested physical groups form a "
                                                "non-orientable surface.");
        }
      }
      components.push_back(std::move(component));
    }

    // Triangulate each component in its consistent orientation.
    std::map<uint64_t, uint64_t> new_index;
    const auto Index = [&](const uint64_t v)
    {
      const auto [it, inserted] = new_index.try_emplace(v, new_vertices.size());
      if (inserted)
        new_vertices.push_back(mesh_vertices[v]);
      return it->second;
    };
    std::vector<std::vector<std::array<uint64_t, 3>>> shells;
    for (const auto& component : components)
    {
      std::vector<std::array<uint64_t, 3>> shell;
      for (const auto f : component)
      {
        auto face = elements[f];
        if (flipped[f])
          std::reverse(face.begin(), face.end());
        shell.push_back({Index(face[0]), Index(face[1]), Index(face[2])});
        if (face.size() == 4)
          shell.push_back({Index(face[0]), Index(face[2]), Index(face[3])});
      }
      shells.push_back(std::move(shell));
    }

    // Outer shells enclose positive volume and holes negative volume, so that the outward normal
    // points out of the enclosed region.
    const auto depths = NestingDepths(
      shells.size(),
      [&](const size_t outer, const size_t inner)
      { return InsideShell(new_vertices[shells[inner][0][0]], shells[outer], new_vertices); });
    for (size_t s = 0; s < shells.size(); ++s)
    {
      const double volume = SignedVolume(shells[s], new_vertices);
      if (volume == 0.0)
        throw std::invalid_argument(fname + ": A surface of the requested physical groups "
                                            "encloses no volume.");
      const bool positive = volume > 0.0;
      const bool hole = depths[s] % 2 == 1;
      if (positive == hole)
        for (auto& triangle : shells[s])
          std::swap(triangle[1], triangle[2]);
      triangles.insert(triangles.end(), shells[s].begin(), shells[s].end());
    }
  }

  std::vector<Face> new_faces;
  new_faces.reserve(triangles.size());
  for (const auto& [a, b, c] : triangles)
  {
    Face face;
    face.v_index = {static_cast<int>(a), static_cast<int>(b), static_cast<int>(c)};
    face.e_index[0][0] = face.v_index[0];
    face.e_index[0][1] = face.v_index[1];
    face.e_index[1][0] = face.v_index[1];
    face.e_index[1][1] = face.v_index[2];
    face.e_index[2][0] = face.v_index[2];
    face.e_index[2][1] = face.v_index[0];
    const auto& vA = new_vertices[a];
    const auto& vB = new_vertices[b];
    const auto& vC = new_vertices[c];
    const Vector3 normal = (vB - vA).Cross(vC - vB);
    if (normal.Norm() == 0.0)
      throw std::invalid_argument(fname + ": The requested physical groups contain a degenerate "
                                          "element.");
    face.geometric_normal = normal / normal.Norm();
    face.assigned_normal = face.geometric_normal;
    face.face_centroid = (vA + vB + vC) / 3.0;
    new_faces.push_back(face);
  }

  // Replace the contents of this surface mesh only after the import has succeeded.
  vertices_ = std::move(new_vertices);
  faces_ = std::move(new_faces);
  tex_vertices_.clear();
  normals_.clear();
  lines_.clear();
  poly_faces_.clear();
  physical_region_map_.clear();

  log.Log0Verbose1() << fname << ": Surface mesh loaded from " << file_name << " with "
                     << faces_.size() << " triangles.";
}

} // namespace opensn
