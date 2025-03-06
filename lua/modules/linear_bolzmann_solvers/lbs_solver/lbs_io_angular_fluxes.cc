// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "lua/modules/linear_bolzmann_solvers/lbs_solver/lbs_common_lua_functions.h"
#include "lua/framework/lua.h"
#include "lua/framework/console/console.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/lbs_solver.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/groupset/lbs_groupset.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/io/lbs_solver_io.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_solver/lbs_discrete_ordinates_solver.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"

using namespace opensn;

namespace opensnlua
{

RegisterLuaFunctionInNamespace(LBSWriteGroupsetAngularFlux, lbs, WriteGroupsetAngularFlux);
RegisterLuaFunctionInNamespace(LBSReadGroupsetAngularFlux, lbs, ReadGroupsetAngularFlux);

RegisterLuaFunctionInNamespace(LBSWriteAngularFluxes, lbs, WriteAngularFluxes);
RegisterLuaFunctionInNamespace(LBSWriteSurfaceAngularFluxes, lbs, WriteSurfaceAngularFluxes);
RegisterLuaFunctionInNamespace(LBSReadAngularFluxes, lbs, ReadAngularFluxes);
RegisterLuaFunctionInNamespace(LBSReadSurfaceAngularFluxes, lbs, ReadSurfaceAngularFluxes);

int
LBSWriteGroupsetAngularFlux(lua_State* L)
{
  const std::string fname = "lbs.WriteGroupsetAngularFlux";
  LuaCheckArgs<size_t, int, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto groupset_index = LuaArg<int>(L, 2);
  const auto file_base = LuaArg<std::string>(L, 3);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);
  LBSSolverIO::WriteGroupsetAngularFluxes(lbs_solver, groupset_index, file_base);

  return LuaReturn(L);
}

int
LBSReadGroupsetAngularFlux(lua_State* L)
{
  const std::string fname = "lbs.ReadGroupsetAngularFlux";
  LuaCheckArgs<size_t, int, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto groupset_index = LuaArg<int>(L, 2);
  const auto file_base = LuaArg<std::string>(L, 3);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);
  LBSSolverIO::ReadGroupsetAngularFluxes(lbs_solver, groupset_index, file_base);

  return LuaReturn(L);
}

int
LBSWriteAngularFluxes(lua_State* L)
{
  const std::string fname = "lbs.WriteAngularFluxes";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);

  // Get boundary IDs given boundary names are provided
  std::vector<uint64_t> bndry_ids;
  if (LuaNumArgs(L) > 2)
  {
    // Get the supported boundaries
    const auto supported_boundary_names = opensn::DiscreteOrdinatesSolver::supported_boundary_names;
    const auto supported_boundary_ids = opensn::DiscreteOrdinatesSolver::supported_boundary_ids;
    
    auto bnd_names = LuaArg<std::vector<std::string>>(L, 3);
    for (auto& name : bnd_names)
      bndry_ids.push_back(supported_boundary_names.at(name));

    for (auto& bid : bndry_ids)
    {
      auto bnd_name = supported_boundary_ids.at(bid);
    }
    LBSSolverIO::WriteAngularFluxes(lbs_solver, file_base, bndry_ids);
  }
  else
    LBSSolverIO::WriteAngularFluxes(lbs_solver, file_base);

  return LuaReturn(L);
}

int
LBSWriteSurfaceAngularFluxes(lua_State* L)
{
  const std::string fname = "lbs.WriteSurfaceAngularFluxes";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);
  auto bnd_names = LuaArg<std::vector<std::string>>(L, 3);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);

  // Get the supported boundaries
  const auto supported_boundary_names = opensn::DiscreteOrdinatesSolver::supported_boundary_names;
  const auto supported_boundary_ids = opensn::DiscreteOrdinatesSolver::supported_boundary_ids;
  
  std::map<std::string, uint64_t> bndry_map;
  for (auto& name : bnd_names)
  {
    uint64_t bid = supported_boundary_names.at(name);
    bndry_map[name] = bid;
  }
  
  // std::vector<uint64_t> bndry_ids;
  // for (auto& name : bnd_names)
  //   bndry_ids.push_back(supported_boundary_names.at(name));

  // for (auto& bid : bndry_ids)
  // {
  //   auto bnd_name = supported_boundary_ids.at(bid);
  // }

  LBSSolverIO::WriteSurfaceAngularFluxes(lbs_solver, file_base, bndry_map);

  return LuaReturn(L);
}

int
LBSReadAngularFluxes(lua_State* L)
{
  const std::string fname = "lbs.ReadAngularFluxes";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);
  LBSSolverIO::ReadAngularFluxes(lbs_solver, file_base);

  return LuaReturn(L);
}
int
LBSReadSurfaceAngularFluxes(lua_State* L)
{
  const std::string fname = "lbs.ReadSurfaceAngularFluxes";
  LuaCheckArgs<size_t, std::string>(L, fname);

  const auto solver_handle = LuaArg<size_t>(L, 1);
  const auto file_base = LuaArg<std::string>(L, 2);

  // Get pointer to solver
  auto& lbs_solver =
    opensn::GetStackItem<opensn::LBSSolver>(opensn::object_stack, solver_handle, fname);
  LBSSolverIO::ReadSurfaceAngularFluxes(lbs_solver, file_base);

  return LuaReturn(L);
}

} // namespace opensnlua
