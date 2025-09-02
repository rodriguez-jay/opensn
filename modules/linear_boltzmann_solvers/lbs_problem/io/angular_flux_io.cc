// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/io/lbs_problem_io.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "framework/runtime.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/hdf_utils.h"

namespace opensn
{

void
LBSSolverIO::WriteAngularFluxes(
  DiscreteOrdinatesProblem& do_problem,
  const std::string& file_base,
  std::optional<const std::reference_wrapper<std::vector<std::vector<double>>>> opt_src)
{
  // Open the HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "WriteAngularFluxes: Failed to open " + file_name + ".");

  // Select source vector
  std::vector<std::vector<double>>& src =
    opt_src.has_value() ? opt_src.value().get() : do_problem.GetPsiNewLocal();

  log.Log() << "Writing angular flux to " << file_base;

  // Write macro info
  const auto& grid = do_problem.GetGrid();
  const auto& discretization = do_problem.GetSpatialDiscretization();
  const auto& groupsets = do_problem.GetGroupsets();

  auto num_local_cells = grid->local_cells.size();
  auto num_local_nodes = discretization.GetNumLocalNodes();
  auto num_groupsets = groupsets.size();

  H5CreateAttribute(file_id, "num_groupsets", num_groupsets);

  // Store Mesh Information
  std::vector<uint64_t> cell_ids, num_cell_nodes;
  cell_ids.reserve(num_local_cells);
  num_cell_nodes.reserve(num_local_cells);

  std::vector<double> nodes_x, nodes_y, nodes_z;
  nodes_x.reserve(num_local_nodes);
  nodes_y.reserve(num_local_nodes);
  nodes_z.reserve(num_local_nodes);

  for (const auto& cell : grid->local_cells)
  {
    cell_ids.push_back(cell.global_id);
    num_cell_nodes.push_back(discretization.GetCellNumNodes(cell));

    const auto& nodes = discretization.GetCellNodeLocations(cell);
    for (const auto& node : nodes)
    { 
      nodes_x.push_back(node.x);
      nodes_y.push_back(node.y);
      nodes_z.push_back(node.z);
    }
  }

  // Write mesh data to h5 inside the mesh group
  H5CreateGroup(file_id, "mesh");
  H5CreateAttribute(file_id, "mesh/num_local_cells", num_local_cells);
  H5CreateAttribute(file_id, "mesh/num_local_nodes", num_local_nodes);
  H5WriteDataset1D(file_id, "mesh/cell_ids", cell_ids);
  H5WriteDataset1D(file_id, "mesh/num_cell_nodes", num_cell_nodes);
  H5WriteDataset1D(file_id, "mesh/nodes_x", nodes_x);
  H5WriteDataset1D(file_id, "mesh/nodes_y", nodes_y);
  H5WriteDataset1D(file_id, "mesh/nodes_z", nodes_z);

  // Go through each groupset
  for (const auto& groupset : groupsets)
  {
    // Write groupset info
    const auto& uk_man = groupset.psi_uk_man_;
    const auto& quadrature = groupset.quadrature;

    auto groupset_id = groupset.id;
    auto num_gs_dirs = quadrature->omegas.size();
    auto num_gs_groups = groupset.groups.size();

    const auto group_name = "groupset_" + std::to_string(groupset_id);
    H5CreateGroup(file_id, group_name);
    H5CreateAttribute(file_id, group_name + "/num_directions", num_gs_dirs);
    H5CreateAttribute(file_id, group_name + "/num_groups", num_gs_groups);

    // Write the groupset angular flux data
    std::vector<double> values;
    for (const auto& cell : grid->local_cells)
      for (uint64_t i = 0; i < discretization.GetCellNumNodes(cell); ++i)
        for (uint64_t n = 0; n < num_gs_dirs; ++n)
          for (uint64_t g = 0; g < num_gs_groups; ++g)
          {
            const auto dof_map = discretization.MapDOFLocal(cell, i, uk_man, n, g);
            values.push_back(src[groupset_id][dof_map]);
          }
    H5WriteDataset1D(file_id, group_name + "/values", values);
  }
  H5Fclose(file_id);
}

void
LBSSolverIO::ReadAngularFluxes(
  DiscreteOrdinatesProblem& do_problem,
  const std::string& file_base,
  std::optional<std::reference_wrapper<std::vector<std::vector<double>>>> opt_dest)
{
  // Open HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "Failed to open " + file_name + ".");

  // Select destination vector
  std::vector<std::vector<double>>& dest =
    opt_dest.has_value() ? opt_dest.value().get() : do_problem.GetPsiNewLocal();

  log.Log() << "Reading angular flux file from " << file_base;

  // Read macro data and check for compatibility
  uint64_t file_num_groupsets;
  uint64_t file_num_local_cells;
  uint64_t file_num_local_nodes;

  H5ReadAttribute(file_id, "num_groupsets", file_num_groupsets);
  H5ReadAttribute(file_id, "mesh/num_local_cells", file_num_local_cells);
  H5ReadAttribute(file_id, "mesh/num_local_nodes", file_num_local_nodes);

  const auto& grid = do_problem.GetGrid();
  const auto& discretization = do_problem.GetSpatialDiscretization();
  const auto& groupsets = do_problem.GetGroupsets();

  const auto num_local_nodes = discretization.GetNumLocalNodes();
  const auto num_groupsets = groupsets.size();

  OpenSnLogicalErrorIf(file_num_local_nodes != num_local_nodes,
                       "Incompatible number of local nodes found in file " + file_name + ".");
  OpenSnLogicalErrorIf(file_num_groupsets != num_groupsets,
                       "Incompatible number of groupsets found in file " + file_name + ".");

  // Read in mesh information
  std::vector<uint64_t> file_cell_ids, file_num_cell_nodes;
  H5ReadDataset1D<uint64_t>(file_id, "mesh/cell_ids", file_cell_ids);
  H5ReadDataset1D<uint64_t>(file_id, "mesh/num_cell_nodes", file_num_cell_nodes);

  std::vector<double> nodes_x, nodes_y, nodes_z;
  H5ReadDataset1D<double>(file_id, "mesh/nodes_x", nodes_x);
  H5ReadDataset1D<double>(file_id, "mesh/nodes_y", nodes_y);
  H5ReadDataset1D<double>(file_id, "mesh/nodes_z", nodes_z);

  // Validate mesh compatibility
  uint64_t curr_node = 0;
  std::map<uint64_t, std::map<uint64_t, uint64_t>> file_cell_nodal_mapping;
  for (uint64_t c = 0; c < file_num_local_cells; ++c)
  {
    const uint64_t cell_global_id = file_cell_ids[c];
    const auto& cell = grid->cells[cell_global_id];

    if (not grid->IsCellLocal(cell_global_id))
      continue;

    // Check for cell compatibility
    const auto& nodes = discretization.GetCellNodeLocations(cell);
    OpenSnLogicalErrorIf(nodes.size() != file_num_cell_nodes[c],
                         "Incompatible number of cell nodes encountered on cell " +
                           std::to_string(cell_global_id) + ".");

    std::vector<Vector3> file_nodes;
    file_nodes.reserve(file_num_cell_nodes[c]);
    for (uint64_t n = 0; n < file_num_cell_nodes[c]; ++n)
    {
      file_nodes.emplace_back(nodes_x[curr_node], nodes_y[curr_node], nodes_z[curr_node]);
      ++curr_node;
    }

    // Map the system nodes to file nodes
    auto& mapping = file_cell_nodal_mapping[cell_global_id];
    for (uint64_t n = 0; n < file_num_cell_nodes[c]; ++n)
    {
      bool mapping_found = false;
      for (uint64_t m = 0; m < nodes.size(); ++m)
        if ((nodes[m] - file_nodes[n]).NormSquare() < 1.0e-12)
        {
          mapping[n] = m;
          mapping_found = true;
        }
      OpenSnLogicalErrorIf(not mapping_found,
                           "Incompatible node locations for cell " +
                             std::to_string(cell_global_id) + ".");
    }
  }

  // Read groupset data
  dest.clear();
  for (uint64_t gs = 0; gs < num_groupsets; ++gs)
  {
    uint64_t file_num_gs_dirs;
    uint64_t file_num_gs_groups;

    auto group_name = "groupset_" + std::to_string(gs);
    H5ReadAttribute(file_id, group_name + "/num_directions", file_num_gs_dirs);
    H5ReadAttribute(file_id, group_name + "/num_groups", file_num_gs_groups);

    const auto& groupset = groupsets.at(gs);
    const auto& uk_man = groupset.psi_uk_man_;
    const auto& quadrature = groupset.quadrature;

    const auto num_gs_dirs = quadrature->omegas.size();
    const auto num_gs_groups = groupset.groups.size();
    OpenSnLogicalErrorIf(file_num_gs_dirs != num_gs_dirs,
                         "Incompatible number of groupset angles found in file " + file_name +
                           " for groupset " + std::to_string(gs) + ".");
    OpenSnLogicalErrorIf(file_num_gs_groups != num_gs_groups,
                         "Incompatible number of groupset groups found in file " + file_name +
                           " for groupset " + std::to_string(gs) + ".");

    // Size the groupset angular flux vector
    const auto num_local_gs_dofs = discretization.GetNumLocalDOFs(uk_man);
    dest.emplace_back(num_local_gs_dofs, 0.0);
    auto& psi = dest.back();

    // Read the groupset angular flux vector
    uint64_t v = 0;
    std::vector<double> values;
    H5ReadDataset1D<double>(file_id, group_name + "/values", values);
    for (uint64_t c = 0; c < file_num_local_cells; ++c)
    {
      // bool isBndry = false; 
      const auto cell_global_id = file_cell_ids[c];
      const auto& cell = grid->cells[cell_global_id];

      const auto& unit_cell_matrices = lbs_problem.GetUnitCellMatrices();
	    const auto& fe_values = unit_cell_matrices.at(cell.local_id);
      
      for (uint64_t i = 0; i < discretization.GetCellNumNodes(cell); ++i)
      {
        const auto& cell_mapping = discretization.GetCellMapping(cell);
        const auto& node_locations = cell_mapping.GetNodeLocations();
        const auto& node_vec = node_locations[i];

        // if (node_vec.x == 8.0 || node_vec.y == 8.0 || node_vec.z == 8.0)
        // {
        //   isBndry = true;
        //   std::cout << "Cell : " << cell_global_id << std::endl;
        //   std::cout << "Vec Pos : " << node_vec[0] << " " << node_vec[1] << " " << node_vec[2] << std::endl;
        //   std::cout << num_gs_dirs << std::endl;
        //   std::cout << num_gs_groups << std::endl;
        //   exit(0);
        // }

        for (uint64_t n = 0; n < num_gs_dirs; ++n)
          for (uint64_t g = 0; g < num_gs_groups; ++g)
          {
            const auto& imap = file_cell_nodal_mapping.at(cell_global_id).at(i);
            const auto dof_map = discretization.MapDOFLocal(cell, imap, uk_man, n, g);
            psi[dof_map] = values[v];

            // if (isBndry)
            //   std::cout << "Group : " << g << " Psi : " << values[v] << std::endl;

            ++v;
          }
      }
    }
  }
  H5Fclose(file_id);
}

void
LBSSolverIO::WriteSurfaceAngularFluxes(
  LBSProblem& lbs_problem,
  const std::string& file_base,
  std::vector<std::string>& bndrys,
  std::optional<std::pair<std::string, double>> surfaces)
{
  // Open the HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "WriteSurfaceAngularFluxes: Failed to open " + file_name + ".");

  // Perform checks
  // OpenSnLogicalErrorIf(not options_.save_angular_flux,
  //                      "The option `save_angular_flux` must be set to `true` in order "
  //                      "to compute outgoing currents.");

  // if (surfaces.has_value())
  // {
  //   const std::string& surface_id = surfaces->first;
  //   double slice = surfaces->second;
  // }

  log.Log() << "Writing surface angular flux to " << file_base;

  std::vector<std::vector<double>>& psi = lbs_problem.GetPsiNewLocal();
  const auto& supported_bd_ids = lbs_problem.supported_boundary_ids;
  const auto& supported_bd_names = lbs_problem.supported_boundary_names;

  // Write macro info
  const auto& grid = lbs_problem.GetGrid();
  const auto& discretization = lbs_problem.GetSpatialDiscretization();
  const auto& groupsets = lbs_problem.GetGroupsets();

  auto num_local_cells = grid->local_cells.size();
  auto num_local_nodes = discretization.GetNumLocalNodes();
  auto num_groupsets = groupsets.size();

  // Store groupsets
  H5CreateAttribute(file_id, "num_groupsets", num_groupsets);

  // Check the boundary IDs
  std::vector<uint64_t> bndry_ids;
  const auto unique_bids = grid->GetUniqueBoundaryIDs();
  for (const auto& bndry : bndrys)
  {
    // Verify if supplied boundary has a valid boundary ID
    const auto bndry_id = supported_bd_names.at(bndry);
    bndry_ids.push_back(bndry_id);
    const auto id = std::find(unique_bids.begin(), unique_bids.end(), bndry_id);
    OpenSnInvalidArgumentIf(id == unique_bids.end(),
                            "Boundary " + bndry + "not found on grid.");
  }

  std::vector<std::string> bndry_tags;

  // The mesh map is structured as follows;
  // Bndry_id -> Cell_i    -> [nodes_i, nodes_i+1, ..., nodes_n]
  //          -> Cell_i+1  -> [nodes_i, nodes_i+1, ..., nodes_n]
  //          -> ...       -> [...]
  //          -> Cell_n    -> [...]
  // std::map<std::string, std::vector<std::pair<uint64_t, std::vector<double>>>> mesh_map;

  // Map of Boundary IDs to Structured CellData object
  std::map<std::string, std::vector<LBSSolverIO::CellData>> mesh_map;
  
  // Go through each groupset
  unsigned int gset = 0;
  for (const auto& groupset : groupsets)
  {
    // Write groupset info
    std::map<uint64_t, std::vector<uint64_t>> surf_map;
    
    std::map<std::string, std::vector<std::vector<double>>> data_map;
    std::map<std::string, std::vector<double>> mass_map;

    const auto& uk_man = groupset.psi_uk_man_;
    const auto& quadrature = groupset.quadrature;

    auto groupset_id = groupset.id;
    auto num_gs_dirs = quadrature->omegas.size();
    auto num_gs_groups = groupset.groups.size();

    // std::cout << "Num Dirs: " << num_gs_dirs << std::endl;

    // Loop over all cells
    std::vector<LBSSolverIO::CellData> cell_data;
    for (const auto& cell : grid->local_cells)
    {  
      // const auto& cell_mapping = discretization_->GetCellMapping(cell);
      const uint64_t& cell_id = cell.global_id;
      const auto& cell_mapping = discretization.GetCellMapping(cell);
      const auto& node_locations = cell_mapping.GetNodeLocations();

      const auto& unit_cell_matrices = lbs_problem.GetUnitCellMatrices();
	    const auto& fe_values = unit_cell_matrices.at(cell.local_id);

      // std::cout << "Num Faces:" << cell.faces.size() << std::endl;
      // Loop over each face of the cell
      unsigned int f = 0;
      for (const auto& face : cell.faces)
      {
        bool isSurf = false;
        std::string bndry_name;

        // Surface Mapping
        if (surfaces.has_value())
        {
          const std::string& surf_id = surfaces->first;
          double slice = surfaces->second;

          const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);
          for (unsigned int fi = 0; fi < num_face_nodes; ++fi)
          {
            const auto i = cell_mapping.MapFaceNode(f, fi);
            const auto& node_vec = node_locations[i];
            if (node_vec.z == slice)
            {
              std::cout << "Surface ID: " << surf_id << std::endl;
              std::cout << "Cell ID: " << cell_id << std::endl;
              std::cout << "Face ID: " << f << std::endl;
              std::cout << "Node ID: " << i << std::endl;
              std::cout << "Sliced @: " << slice << std::endl;
              std::cout << "Num Nodes: " << num_face_nodes << std::endl;
              std::cout << "Vec Pos : " << node_vec.x << " " 
                                        << node_vec.y << " " 
                                        << node_vec.z << std::endl << std::endl;
              std::stringstream surf_str;
              surf_str << surf_id << cell_id;
              bndry_name = surf_str.str();
              isSurf = true;
              bndry_tags.push_back(bndry_name);
            }
          }
        }
        //

        // Boundary Mapping
        const auto it = std::find(bndry_ids.begin(), bndry_ids.end(), face.neighbor_id);
        if (not face.has_neighbor and it != bndry_ids.end())
        {
          bndry_name = supported_bd_ids.at(*it);
          isSurf = true;
          bndry_tags.push_back(bndry_name);
          // std::cout << bndry_name << std::endl;
        }

        // Write Surface Data
        if (isSurf)
        {
          double lkgrate = 0.0;
          const auto& int_f_shape_i = fe_values.intS_shapeI[f];
          const auto& M_ij = fe_values.intS_shapeI_shapeJ[f];

          const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);

          // std::vector<LBSSolverIO::NodeData> node_data; 
          for (unsigned int fi = 0; fi < num_face_nodes; ++fi)
          {
            const auto i = cell_mapping.MapFaceNode(f, fi);
            const auto& node_vec = node_locations[i];
            // node_data.push_back({i, node_Vec});
            cell_data[cell_id].nodes.push_back({i, {node_vec[0], node_vec[1], node_vec[2]}});

            // if (gset == 0)
              // mesh_map[bndry_name].push_back({cell_id, {node_vec[0], node_vec[1], node_vec[2]}});

            for (unsigned int d = 0; d < num_gs_dirs; ++d)
            {
              const auto& omega_d = quadrature->omegas[d];
              const auto weight_d = quadrature->weights[d];
              const auto mu_d = omega_d.Dot(face.normal);
               
              std::vector<double> data_vec;
              data_vec.insert(data_vec.end(), {omega_d.x, omega_d.y, omega_d.z});
              data_vec.push_back(mu_d);
              data_vec.push_back(weight_d);
              data_vec.push_back(int_f_shape_i(i));
              for (uint64_t g = 0; g < num_gs_groups; ++g)
              {
                const auto dof_map = discretization.MapDOFLocal(cell, i, uk_man, d, g);
                data_vec.push_back(psi[groupset_id][dof_map]);
              }
              // Move the vector to avoid unecessary copy
              data_map[bndry_name].push_back(std::move(data_vec));
            }

            for (unsigned int fj = 0; fj < num_face_nodes; ++fj)
            {
              const auto j = cell_mapping.MapFaceNode(f, fj);
              mass_map[bndry_name].push_back(M_ij(i, j));
            }
          }
        }
        ++f;
      }
      mesh_map[bndry_tags.back()].push_back(cell_data[cell_id]);
    }

    std::string group_name = "groupset_" + std::to_string(groupset_id);

    H5CreateGroup(file_id, group_name);
    H5CreateAttribute(file_id, group_name + "/num_directions", num_gs_dirs);
    H5CreateAttribute(file_id, group_name + "/num_groups", num_gs_groups);

    /////////////////////////////////////////////////////////////////////////
    // Fix mesh map to save number of faces and a vector with the number 
    // of nodes !!
    /////////////////////////////////////////////////////////////////////////
    // Write mesh information
    if (gset == 0)
    {
      H5CreateGroup(file_id, "mesh");
      // Per boundary id
      for (const auto& [key, node_vectors] : mesh_map) 
      {
        // Create Group
        const auto& bndry_id = key.c_str();
        std::vector<uint64_t> cell_ids;
        std::vector<double> nodes_x, nodes_y, nodes_z;
        for (const auto& [cell_id, vec] : node_vectors)
        {
          cell_ids.push_back(cell_id);
          nodes_x.push_back(vec[0]);
          nodes_y.push_back(vec[1]);
          nodes_z.push_back(vec[2]);
        }

        std::string bndry_mesh = std::string("mesh/") + bndry_id;
        H5CreateGroup(file_id, bndry_mesh);
        H5WriteDataset1D(file_id, bndry_mesh + "/cell_ids", cell_ids);
        H5WriteDataset1D(file_id, bndry_mesh + "/nodes_x", nodes_x);
        H5WriteDataset1D(file_id, bndry_mesh + "/nodes_y", nodes_y);
        H5WriteDataset1D(file_id, bndry_mesh + "/nodes_z", nodes_z);
      }
    }

    // Write data information
    for (const auto& [key, data_vectors] : data_map) 
    {
      const auto& bndry_name = key.c_str();
      std::string bndry_grp = group_name + std::string("/") + bndry_name;
      // H5CreateGroup(file_id, group_name + "/" + bndry_id);
      H5CreateGroup(file_id, bndry_grp);

      // Write the groupset surface angular flux data
      std::vector<double> omega;
      std::vector<double> mu;
      std::vector<double> wt_d;
      std::vector<double> fe_shape;
      std::vector<double> surf_flux;
      for (const auto&vec : data_vectors)
      {
        omega.insert(omega.end(), {vec[0], vec[1], vec[2]});
        mu.push_back(vec[3]);
        wt_d.push_back(vec[4]);
        fe_shape.push_back(vec[5]);
        // surf_flux.push_back(vec[6]);
        surf_flux.insert(surf_flux.end(), vec.begin()+6, vec.end());
      }
      H5WriteDataset1D(file_id, bndry_grp + "/omega", omega);
      H5WriteDataset1D(file_id, bndry_grp + "/mu", mu);
      H5WriteDataset1D(file_id, bndry_grp + "/wt_d", wt_d);
      H5WriteDataset1D(file_id, bndry_grp + "/fe_shape", fe_shape);
      H5WriteDataset1D(file_id, bndry_grp + "/surf_flux", surf_flux);
    }

    // Write mass matrix information
    for (const auto& [key, mass_vector] : mass_map) 
    {
      const auto& bndry_id = key.c_str();
      std::string bndry_grp = group_name + std::string("/") + bndry_id;
      H5WriteDataset1D(file_id, bndry_grp + "/M_ij", mass_vector);
    }
    ++gset;
  }

  ssize_t num_open_objs = H5Fget_obj_count(file_id, H5F_OBJ_ALL);
  // std::cout << num_open_objs << std::endl;
  H5Fclose(file_id);
}

std::vector<LBSSolverIO::SurfaceAngularFluxes>
LBSSolverIO::ReadSurfaceAngularFluxes(
  LBSProblem& lbs_problem,
  const std::string& file_base,
  std::vector<std::string>& bndrys)
{
  std::vector<SurfaceAngularFluxes> surf_fluxes;

  // Open HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "Failed to open " + file_name + ".");

  log.Log() << "Reading surface angular flux file from " << file_base;

  const auto& supported_bd_ids = lbs_problem.supported_boundary_ids;
  const auto& supported_bd_names = lbs_problem.supported_boundary_names;

  // Read macro data and check for compatibility
  uint64_t file_num_groupsets;
  H5ReadAttribute(file_id, "num_groupsets", file_num_groupsets);

  const auto& grid = lbs_problem.GetGrid();
  const auto& discretization = lbs_problem.GetSpatialDiscretization();
  const auto& groupsets = lbs_problem.GetGroupsets();
  auto num_groupsets = groupsets.size();

  OpenSnLogicalErrorIf(file_num_groupsets != num_groupsets,
                       "Incompatible number of groupsets found in file " + file_name + ".");
  std::cout << "Num Groupsets " << file_num_groupsets << std::endl;

  // Check the boundary IDs
  std::vector<uint64_t> bndry_ids;
  // std::vector<std::string> bndry_names;
  const auto unique_bids = grid->GetUniqueBoundaryIDs();
  for (const auto& bndry : bndrys)
  {
    // const auto bndry_id = supported_bd_ids.at(bndry);
    const auto bndry_id = supported_bd_names.at(bndry);

    // bndry_names.push_back(bndry.first);
    bndry_ids.push_back(bndry_id);
    const auto id = std::find(unique_bids.begin(), unique_bids.end(), bndry_id);
    OpenSnInvalidArgumentIf(id == unique_bids.end(),
                            "Boundary " + bndry + "not found on grid.");
  }

  ///////////////////////////////////////////////////////////////////////////
  // * Double check multiple groupsets!! There might be something there wrong
  // * Add a way to get mesh data for each surface node -> M_ij
  ///////////////////////////////////////////////////////////////////////////

  // Go through each groupset
  for (const auto& groupset : groupsets)
  {
    const auto& uk_man = groupset.psi_uk_man_;
    const auto& quadrature = groupset.quadrature;

    auto groupset_id = groupset.id;
    auto num_gs_dirs = quadrature->omegas.size();
    auto num_gs_groups = groupset.groups.size();

    uint64_t file_num_gs_dirs;
    uint64_t file_num_gs_groups;
    std::string group_name = "groupset_" + std::to_string(groupset_id);
    H5ReadAttribute(file_id, group_name + "/num_directions", file_num_gs_dirs);
    H5ReadAttribute(file_id, group_name + "/num_groups", file_num_gs_groups);
    // OpenSnLogicalErrorIf(file_num_gs_dirs != num_gs_dirs,
    //                      "Incompatible number of groupset angles found in file " + file_name +
    //                        " for groupset " + groupset + ".");
    // OpenSnLogicalErrorIf(file_num_gs_groups != num_gs_groups,
    //                      "Incompatible number of groupset groups found in file " + file_name +
    //                        " for groupset " + groupset + ".");

    for (const auto& bndry : bndrys)
    {
      SurfaceAngularFluxes fluxes;

      std::cout << "Bndry Name: " << bndry << std::endl;
      std::string bndry_grp = group_name + "/" + bndry;

      H5ReadDataset1D<double>(file_id, bndry_grp + "/omega", fluxes.omega);
      H5ReadDataset1D<double>(file_id, bndry_grp + "/mu", fluxes.mu);
      H5ReadDataset1D<double>(file_id, bndry_grp + "/wt_d", fluxes.wt_d);
      H5ReadDataset1D<double>(file_id, bndry_grp + "/M_ij", fluxes.M_ij);
      H5ReadDataset1D<double>(file_id, bndry_grp + "/surf_flux", fluxes.psi);

      // double totlkg_rate = 0.0;

      // std::string bndry_mesh = "mesh/" + bndry;
      // std::vector<double> cell_ids; 
      // H5ReadDataset1D<double>(file_id, bndry_mesh + "/cell_ids", cell_ids);
      // std::cout << "NCells : " << cell_ids.size() << std::endl;
      // exit(0);


      // auto num_dirs = fluxes.mu.size();
      // for (unsigned int d = 0; d < num_dirs; ++d)
      // {
      //   std::cout << "Omega : " << fluxes.omega[3*d + 0] << " " 
      //                           << fluxes.omega[3*d + 1] << " " 
      //                           << fluxes.omega[3*d + 2] << std::endl;
      //   const auto& mu_d = fluxes.mu[d];
      //   std::cout << "Mu : " << mu_d << std::endl;
      //   if (mu_d > 0.0)
      //   {
      //    for (uint64_t g = 0; g < num_gs_groups; ++g)
      //    {
      //      std::cout << "Index : " << std::to_string(d * num_gs_groups + g) << std::endl;
      //      std::cout << "Value : " << std::to_string(fluxes.psi[d * num_gs_groups + g]) << std::endl;
      //      totlkg_rate = totlkg_rate + fluxes.psi[d * num_gs_groups + g];
      //      std::cout << "Flux :: " << std::to_string(fluxes.psi[d * num_gs_groups + g]) << std::endl;
      //    }
      //   }
      // }
      // std::cout << "Total Lkg Rate : " << totlkg_rate << std::endl;

      surf_fluxes.push_back(std::move(fluxes));
    }
  }
  
  H5Fclose(file_id);

  return surf_fluxes;
}

} // namespace opensn
