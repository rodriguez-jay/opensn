// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace opensn
{

class DiscreteOrdinatesProblem;

class DiscreteOrdinatesProblemIO
{
public:
  /**
   * Write an angular flux vector to a file.
   *
   * \param do_problem Discrete ordinates problem
   * \param file_base File name base
   * \param opt_src Optional angular flux source vector
   */
  static void WriteAngularFluxes(
    DiscreteOrdinatesProblem& do_problem,
    const std::string& file_base,
    std::optional<const std::reference_wrapper<std::vector<std::vector<double>>>> opt_src =
      std::nullopt);

  /**
   * Read an angular flux vector from a file.
   *
   * \param do_problem Discrete ordinates problem
   * \param file_base File name base
   * \param opt_dest Optional angular flux destination vector
   */
  static void ReadAngularFluxes(
    DiscreteOrdinatesProblem& do_problem,
    const std::string& file_base,
    std::optional<std::reference_wrapper<std::vector<std::vector<double>>>> opt_dest =
      std::nullopt);
    
  // Can this work with Vector3?? 
  typedef std::tuple<int64_t, int64_t, int64_t> QuantizedCoordinate;
  typedef std::map<QuantizedCoordinate, uint64_t> FaceMap;
  struct SurfaceMap {
    std::vector<double> cell_ids;
    std::vector<double> num_face_nodes;
    std::map<QuantizedCoordinate, uint64_t> cell_map;
    std::vector<uint64_t> cell_stride;
    std::vector<FaceMap> faces;
    std::vector<double> nodes_x;
    std::vector<double> nodes_y;
    std::vector<double> nodes_z;
  };
  
  struct SurfaceData {
    std::vector<double> omega;
    std::vector<double> mu;
    std::vector<double> wt_d;
    std::vector<double> M_ij;
    std::vector<double> psi;
    std::vector<double> node_index;
    std::vector<double> dir_index;
  };

  struct SurfaceAngularFlux {
    SurfaceMap mapping;
    SurfaceData data;
  };

  /**
   * Write surface angular flux vector(s) to a file.
   *
   * \param do_problem Discrete ordinates problem
   * \param file_base File name base
   * \param bndrys Map of boundary names and ids
   */
  static void WriteSurfaceAngularFluxes(
    DiscreteOrdinatesProblem& do_problem,
    const std::string& file_base,
    std::vector<std::string>& bndry_surfs,
    std::vector<std::pair<std::string, std::pair<std::string, double>>>& int_surfs);

  /**
   * Read a surface angular flux vector from a file.
   *
   * \param do_problem Discrete ordinates problem
   * \param file_base File name base
   * \param per_material Optional angular flux destination vector
   */
  static std::vector<SurfaceAngularFlux> 
  ReadSurfaceAngularFluxes(
    DiscreteOrdinatesProblem& do_problem,
    const std::string& file_base,
    std::vector<std::string>& surfaces);
};

} // namespace opensn
