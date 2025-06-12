// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <string>
#include <optional>
#include <vector>
#include <cstdint>
#include <functional>

namespace opensn
{

class LBSProblem;
class DiscreteOrdinatesProblem;

class LBSSolverIO
{
public:
  /**
   * Write an angular flux vector to a file.
   *
   * \param lbs_problem LBS problem
   * \param file_base File name base
   * \param per_material Optional angular flux source vector
   */
  static void WriteAngularFluxes(
    DiscreteOrdinatesProblem& do_problem,
    const std::string& file_base,
    std::optional<const std::reference_wrapper<std::vector<std::vector<double>>>> opt_src =
      std::nullopt);

  /**
   * Read an angular flux vector from a file.
   *
   * \param lbs_problem LBS problem
   * \param file_base File name base
   * \param per_material Optional angular flux destination vector
   */
  static void ReadAngularFluxes(
    DiscreteOrdinatesProblem& do_problem,
    const std::string& file_base,
    std::optional<std::reference_wrapper<std::vector<std::vector<double>>>> opt_dest =
      std::nullopt);

  /**
   * Write surface angular flux vector(s) to a file.
   *
   * \param lbs_solver LBS solver
   * \param file_base File name stem
   * \param bndry_map Map of boundary names and ids
   */
  static void WriteSurfaceAngularFluxes(
    LBSProblem& lbs_solver,
    const std::string& file_stem,
    std::map<std::string, uint64_t>& bndry_map);

  /**
   * Read a surface angular flux vector from a file.
   *
   * \param lbs_problem LBS problem
   * \param file_base File name stem
   * \param per_material Optional angular flux destination vector
   */
  static void ReadSurfaceAngularFluxes(
    LBSProblem& lbs_solver,
    const std::string& file_stem,
    std::map<std::string, uint64_t>& bndry_map);


  /**
   * Write a flux moments vector to a file.
   *
   * \param lbs_problem LBS problem
   * \param file_base File name base
   * \param per_material Optional flux moments source vector
   */
  static void WriteFluxMoments(
    LBSProblem& lbs_problem,
    const std::string& file_base,
    std::optional<const std::reference_wrapper<std::vector<double>>> opt_src = std::nullopt);

  /**
   * Read a flux moments vector from a file.
   *
   * \param lbs_problem LBS problem
   * \param file_base File name base
   * \param single_file Single data file or data file per rank?
   * \param per_material Optional flux moments destination vector
   */
  static void ReadFluxMoments(
    LBSProblem& lbs_problem,
    const std::string& file_base,
    bool single_file,
    std::optional<std::reference_wrapper<std::vector<double>>> opt_dest = std::nullopt);
};

} // namespace opensn
