#include "modules/LinearBoltzmannSolvers/A_LBSSolver/lbs_solver.h"

#include "framework/chi_runtime.h"
#include "framework/logging/chi_log.h"
#include "modules/LinearBoltzmannSolvers/A_LBSSolver/Groupset/lbs_groupset.h"

namespace lbs::common_lua_utils
{

int
chiLBSWriteGroupsetAngularFlux(lua_State* L)
{
  const std::string fname = "chiLBSWriteGroupsetAngularFlux";
  // Get arguments
  const int num_args = lua_gettop(L);
  if (num_args != 3) LuaPostArgAmountError(fname, 3, num_args);

  LuaCheckNilValue(fname, L, 1);
  LuaCheckNilValue(fname, L, 2);
  LuaCheckNilValue(fname, L, 3);

  const int solver_handle = lua_tonumber(L, 1);
  const int grpset_index = lua_tonumber(L, 2);
  const std::string file_base = lua_tostring(L, 3);

  // Get pointer to solver
  auto& lbs_solver = Chi::GetStackItem<lbs::LBSSolver>(Chi::object_stack, solver_handle, fname);

  // Obtain pointer to groupset
  lbs::LBSGroupset* groupset = nullptr;
  try
  {
    groupset = &lbs_solver.Groupsets().at(grpset_index);
  }
  catch (const std::out_of_range& o)
  {
    Chi::log.LogAllError() << "Invalid handle to groupset "
                           << "in call to " << fname;
    Chi::Exit(EXIT_FAILURE);
  }

  lbs_solver.WriteGroupsetAngularFluxes(*groupset, file_base);

  return 0;
}

int
chiLBSReadGroupsetAngularFlux(lua_State* L)
{
  const std::string fname = "chiLBSReadGroupsetAngularFlux";
  // Get arguments
  const int num_args = lua_gettop(L);
  if (num_args != 3) LuaPostArgAmountError(fname, 3, num_args);

  LuaCheckNilValue(fname, L, 1);
  LuaCheckNilValue(fname, L, 2);
  LuaCheckNilValue(fname, L, 3);

  const int solver_handle = lua_tonumber(L, 1);
  const int grpset_index = lua_tonumber(L, 2);
  const std::string file_base = lua_tostring(L, 3);

  // Get pointer to solver
  auto& lbs_solver = Chi::GetStackItem<lbs::LBSSolver>(Chi::object_stack, solver_handle, fname);

  // Obtain pointer to groupset
  lbs::LBSGroupset* groupset = nullptr;
  try
  {
    groupset = &lbs_solver.Groupsets().at(grpset_index);
  }
  catch (const std::out_of_range& o)
  {
    Chi::log.LogAllError() << "Invalid handle to groupset "
                           << "in call to " << fname;
    Chi::Exit(EXIT_FAILURE);
  }

  lbs_solver.ReadGroupsetAngularFluxes(*groupset, file_base);

  return 0;
}

} // namespace lbs::common_lua_utils