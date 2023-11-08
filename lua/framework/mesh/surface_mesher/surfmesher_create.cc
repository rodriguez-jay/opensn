#include "framework/chi_lua.h"
#include <iostream>
#include "framework/mesh/SurfaceMesher/Predefined/surfmesher_predefined.h"

#include "framework/mesh/MeshHandler/chi_meshhandler.h"

#include "framework/chi_runtime.h"
#include "framework/logging/chi_log.h"
#include "surfmesher_lua.h"
#include "framework/console/chi_console.h"

RegisterLuaFunctionAsIs(chiSurfaceMesherCreate);
RegisterLuaConstantAsIs(SURFACEMESHER_PREDEFINED, chi_data_types::Varying(1));

int
chiSurfaceMesherCreate(lua_State* L)
{
  auto& cur_hndlr = chi_mesh::GetCurrentHandler();

  // Get argument
  LuaCheckNilValue("chiSurfaceMesherCreate", L, 1);
  int type = lua_tonumber(L, 1);

  // Create the surface mesher
  std::shared_ptr<chi_mesh::SurfaceMesher> new_mesher = nullptr;
  if (type == (int)chi_mesh::SurfaceMesherType::Predefined)
  {
    new_mesher = std::make_shared<chi_mesh::SurfaceMesherPredefined>();
  }
  else
  {
    std::cerr << "ERROR: Illegal surface mesher specified"
                 "in chiSurfaceMesherCreate"
              << std::endl;
    Chi::Exit(EXIT_FAILURE);
  }

  cur_hndlr.SetSurfaceMesher(new_mesher);

  Chi::log.LogAllVerbose2() << "chiSurfaceMesherCreate: Surface remesher created." << std::endl;

  return 0;
}