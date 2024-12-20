// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"

namespace opensn
{

void
GlobalCellHandler::push_back(std::unique_ptr<Cell> new_cell)
{
  if (new_cell->partition_id == static_cast<uint64_t>(opensn::mpi_comm.rank()))
  {
    new_cell->local_id = local_cells_ref_.size();

    local_cells_ref_.push_back(std::move(new_cell));

    const auto& cell = local_cells_ref_.back();

    global_cell_id_to_native_id_map_.insert(
      std::make_pair(cell->global_id, local_cells_ref_.size() - 1));
  }
  else
  {
    ghost_cells_ref_.push_back(std::move(new_cell));

    const auto& cell = ghost_cells_ref_.back();

    global_cell_id_to_foreign_id_map_.insert(
      std::make_pair(cell->global_id, ghost_cells_ref_.size() - 1));
  }
}

Cell&
GlobalCellHandler::operator[](uint64_t cell_global_index)
{
  auto native_location = global_cell_id_to_native_id_map_.find(cell_global_index);

  if (native_location != global_cell_id_to_native_id_map_.end())
    return *local_cells_ref_[native_location->second];
  else
  {
    auto foreign_location = global_cell_id_to_foreign_id_map_.find(cell_global_index);
    if (foreign_location != global_cell_id_to_foreign_id_map_.end())
      return *ghost_cells_ref_[foreign_location->second];
  }

  std::stringstream ostr;
  ostr << "MeshContinuum::cells. Mapping error."
       << "\n"
       << cell_global_index;

  throw std::invalid_argument(ostr.str());
}

const Cell&
GlobalCellHandler::operator[](uint64_t cell_global_index) const
{
  auto native_location = global_cell_id_to_native_id_map_.find(cell_global_index);

  if (native_location != global_cell_id_to_native_id_map_.end())
    return *local_cells_ref_[native_location->second];
  else
  {
    auto foreign_location = global_cell_id_to_foreign_id_map_.find(cell_global_index);
    if (foreign_location != global_cell_id_to_foreign_id_map_.end())
      return *ghost_cells_ref_[foreign_location->second];
  }

  std::stringstream ostr;
  ostr << "MeshContinuum::cells. Mapping error."
       << "\n"
       << cell_global_index;

  throw std::invalid_argument(ostr.str());
}

std::vector<uint64_t>
GlobalCellHandler::GetGhostGlobalIDs() const
{
  std::vector<uint64_t> ids;
  ids.reserve(GetNumGhosts());

  for (auto& cell : ghost_cells_ref_)
    ids.push_back(cell->global_id);

  return ids;
}

uint64_t
GlobalCellHandler::GetGhostLocalID(uint64_t cell_global_index) const
{
  auto foreign_location = global_cell_id_to_foreign_id_map_.find(cell_global_index);

  if (foreign_location != global_cell_id_to_foreign_id_map_.end())
    return foreign_location->second;

  std::stringstream ostr;
  ostr << "Grid GetGhostLocalID failed to find cell " << cell_global_index;

  throw std::invalid_argument(ostr.str());
}

} // namespace opensn
