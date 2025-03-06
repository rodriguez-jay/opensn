// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_solver/io/lbs_solver_io.h"
#include "modules/linear_boltzmann_solvers/lbs_solver/lbs_solver.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/hdf_utils.h"

namespace opensn
{

void
LBSSolverIO::WriteAngularFluxes(
  LBSSolver& lbs_solver,
  const std::string& file_base,
  std::optional<const std::vector<uint64_t>> bndry_ids,
  std::optional<const std::reference_wrapper<std::vector<std::vector<double>>>> opt_src)
{
  // Open the HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "WriteAngularFluxes: Failed to open " + file_name + ".");

  // Select source vector
  std::vector<std::vector<double>>& src =
    opt_src.has_value() ? opt_src.value().get() : lbs_solver.PsiNewLocal();

  log.Log() << "Writing angular flux to " << file_base;

  // Write macro info
  const auto& grid = lbs_solver.Grid();
  const auto& discretization = lbs_solver.SpatialDiscretization();
  const auto& groupsets = lbs_solver.Groupsets();

  auto num_local_cells = grid.local_cells.size();
  auto num_local_nodes = discretization.GetNumLocalNodes();
  auto num_groupsets = groupsets.size();

  H5CreateAttribute(file_id, "num_groupsets", num_groupsets);

  // Check the boundary IDs
  const auto& bndry = *bndry_ids;
  if (bndry_ids && !bndry_ids->empty()) 
  {  
    const auto unique_bids = grid.GetDomainUniqueBoundaryIDs();
    for (const auto& bid : bndry)
    {
      const auto it = std::find(unique_bids.begin(), unique_bids.end(), bid);
      OpenSnInvalidArgumentIf(it == unique_bids.end(),
                              "Boundary ID " + std::to_string(bid) + "not found on grid.");
    }
  }

  // Store Mesh Information
  std::vector<uint64_t> cell_ids, num_cell_nodes;
  cell_ids.reserve(num_local_cells);
  num_cell_nodes.reserve(num_local_cells);

  std::vector<double> nodes_x, nodes_y, nodes_z;
  nodes_x.reserve(num_local_nodes);
  nodes_y.reserve(num_local_nodes);
  nodes_z.reserve(num_local_nodes);

  for (const auto& cell : grid.local_cells)
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

  // Note: Does  closing groups make writing faster??
  // Write mesh data to h5 inside the mesh group
  // hid_t mesh = H5CreateGroup(file_id, "mesh");
  // // hid_t global_mesh = H5CreateGroup(mesh, "global");
  // H5CreateAttribute(mesh, "num_local_cells", num_local_cells);
  // H5CreateAttribute(mesh, "num_local_nodes", num_local_nodes);
  // H5WriteDataset1D(mesh, "cell_ids", cell_ids);
  // H5WriteDataset1D(mesh, "num_cell_nodes", num_cell_nodes);
  // H5WriteDataset1D(mesh, "nodes_x", nodes_x);
  // H5WriteDataset1D(mesh, "nodes_y", nodes_y);
  // H5WriteDataset1D(mesh, "nodes_z", nodes_z);

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

    std::vector<double> values;
    for (const auto& cell : grid.local_cells)
    {  
      for (uint64_t i = 0; i < discretization.GetCellNumNodes(cell); ++i)
        for (uint64_t n = 0; n < num_gs_dirs; ++n)
          for (uint64_t g = 0; g < num_gs_groups; ++g)
          {
            const auto dof_map = discretization.MapDOFLocal(cell, i, uk_man, n, g);
            values.push_back(src[groupset_id][dof_map]);
          }
      H5WriteDataset1D(file_id, group_name + "/values", values);

      // // Write the groupset surface angular flux data
      // if (bndry_ids) 
      // {
      //   std::vector<double> surf_flux;
      //   std::vector<double> mu;
      //   std::vector<double> coeff;

      //   // const auto& cell_mapping = discretization_->GetCellMapping(cell);
      //   const auto& cell_mapping = discretization.GetCellMapping(cell);
        
	    //   const auto& unit_cell_matrices = lbs_solver.GetUnitCellMatrices();
	    //   const auto& fe_values = unit_cell_matrices.at(cell.local_id);

      //   unsigned int f = 0;
      //   for (const auto& face : cell.faces)
      //   {
      //     // If face is on the specified boundary
      //     const auto it = std::find(bndry.begin(), bndry.end(), face.neighbor_id);
      //     if (not face.has_neighbor and it != bndry.end())
      //     {
      //       // To do: Get the id name!
      //       const auto& int_f_shape_i = fe_values.intS_shapeI[f];
      //       const auto num_face_nodes = cell_mapping.NumFaceNodes(f);
      //       for (unsigned int fi = 0; fi < num_face_nodes; ++fi)
      //       {
      //         const auto i = cell_mapping.MapFaceNode(f, fi);
      //         // Maybe change to (dir : num_gs_dirs)?
      //         for (unsigned int n = 0; n < num_gs_dirs; ++n)
      //         {
      //           const auto& omega_n = quadrature->omegas[n];
      //           const auto& weight_n = quadrature->weights[n];
      //           const auto mu_n = omega_n.Dot(face.normal);

      //           if (mu_n <= 0.0)
      //             continue;

      //           mu.push_back(mu_n);
      //           coeff.push_back(weight_n * mu_n * int_f_shape_i(i));
      //           for (uint64_t g = 0; g < num_gs_groups; ++g)
      //           {
      //             const auto dof_map = discretization.MapDOFLocal(cell, i, uk_man, n, g);
      //             surf_flux.push_back(src[groupset_id][dof_map]);
      //           }
      //         }
      //       }
      //     }
      //     ++f;
      //   }
      //   H5WriteDataset1D(file_id, group_name + "/surf_flux", surf_flux);
      //   H5WriteDataset1D(file_id, group_name + "/mu", mu);
      //   H5WriteDataset1D(file_id, group_name + "/coeff", coeff);
      // }
    }
  }
  H5Fclose(file_id);
}

void
LBSSolverIO::WriteSurfaceAngularFluxes(
  LBSSolver& lbs_solver,
  const std::string& file_base,
  const std::map<std::string, uint64_t>& bndry_map)
{
  // Open the HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "WriteSurfaceAngularFluxes: Failed to open " + file_name + ".");

  log.Log() << "Writing surface angular flux to " << file_base;

  std::vector<std::vector<double>>& src = lbs_solver.PsiNewLocal();
  const auto& supported_boundary_ids = lbs_solver.supported_boundary_ids;

  // Write macro info
  const auto& grid = lbs_solver.Grid();
  const auto& discretization = lbs_solver.SpatialDiscretization();
  const auto& groupsets = lbs_solver.Groupsets();

  auto num_local_cells = grid.local_cells.size();
  auto num_local_nodes = discretization.GetNumLocalNodes();
  auto num_groupsets = groupsets.size();

  // Store groupsets
  H5CreateAttribute(file_id, "num_groupsets", num_groupsets);

  // // Check the boundary IDs
  std::vector<uint64_t> bndry_ids;
  std::vector<std::string> bndry_names;
  const auto unique_bids = grid.GetDomainUniqueBoundaryIDs();
  for (const auto& bndry : bndry_map)
  {
    bndry_names.push_back(bndry.first);
    bndry_ids.push_back(bndry.second);
    const auto id = std::find(unique_bids.begin(), unique_bids.end(), bndry.second);
    OpenSnInvalidArgumentIf(id == unique_bids.end(),
                            "Boundary ID " + std::to_string(bndry.second) + "not found on grid.");
  }

  std::map<std::string, std::vector<std::pair<uint64_t, std::vector<double>>>> mesh_map;
  // Go through each groupset
  unsigned int gset = 0;
  for (const auto& groupset : groupsets)
  {
    // Write groupset info
    std::map<std::string, std::vector<std::vector<double>>> data_map;

    const auto& uk_man = groupset.psi_uk_man_;
    const auto& quadrature = groupset.quadrature;

    auto groupset_id = groupset.id;
    auto num_gs_dirs = quadrature->omegas.size();
    auto num_gs_groups = groupset.groups.size();

    const auto group_name = "groupset_" + std::to_string(groupset_id);
    H5CreateGroup(file_id, group_name);
    H5CreateAttribute(file_id, group_name + "/num_directions", num_gs_dirs);
    H5CreateAttribute(file_id, group_name + "/num_groups", num_gs_groups);

    for (const auto& cell : grid.local_cells)
    {  
      // const auto& cell_mapping = discretization_->GetCellMapping(cell);
      const uint64_t& cell_id = cell.global_id;
      const auto& cell_mapping = discretization.GetCellMapping(cell);
      const auto& node_locations = cell_mapping.GetNodeLocations();
	    const auto& unit_cell_matrices = lbs_solver.GetUnitCellMatrices();
	    const auto& fe_values = unit_cell_matrices.at(cell.local_id);

      unsigned int f = 0;
      for (const auto& face : cell.faces)
      {
        // If face is on the specified boundary
        const auto it = std::find(bndry_ids.begin(), bndry_ids.end(), face.neighbor_id);
        if (not face.has_neighbor and it != bndry_ids.end())
        {
          auto& bndry_name = supported_boundary_ids.at(*it);
          // std::cout << "Neighbor ID: " << *it << std::endl;

          // To do: Get the id name!
          const auto& int_f_shape_i = fe_values.intS_shapeI[f];
          const auto num_face_nodes = cell_mapping.NumFaceNodes(f);

          for (unsigned int fi = 0; fi < num_face_nodes; ++fi)
          {
            const auto i = cell_mapping.MapFaceNode(f, fi);
            const auto& node_vec = node_locations[i];

            if (gset == 0)
              mesh_map[bndry_name].push_back({cell_id, {node_vec[0], node_vec[1], node_vec[2]}});

            // std::cout << mesh_map[bndry_name] << std::endl;
            // for (const auto& value : mesh_map[bndry_name]) 
            // {
            //   std::cout << value << std::endl;
            // }
            // std::cout << "Bndry Name:" << bndry_name << std::endl;
            // std::cout << "Pos: " << node_vec.x << " " << node_vec.y << " " << node_vec.z << std::endl;

            // std::cout << "Boundary: " << bndry_name << std::endl;
            // std::cout << "Cell: " << cell.global_id << std::endl; 
            // std::cout << "Num Directions: " << num_gs_dirs << std::endl << std::endl;
            // std::cout << "Node Location: " << node_vec[0] << " " << node_vec[1] << " " << node_vec[2] << std::endl << std::endl;
            for (unsigned int n = 0; n < num_gs_dirs; ++n)
            {
              const auto& omega_n = quadrature->omegas[n];
              const auto weight_n = quadrature->weights[n];
              const auto mu_n = omega_n.Dot(face.normal);

              // std::cout << "MU:Weight  " << mu_n << " : " << weight_n << std::endl;
              // std::cout << "Omega: " << omega_n.x << " " << omega_n.y << " " << omega_n.z << std::endl << std::endl;

              std::vector<double> data_vec;
              data_vec.push_back(mu_n);
              data_vec.push_back(weight_n * mu_n * int_f_shape_i(i));
              for (uint64_t g = 0; g < num_gs_groups; ++g)
              {
                const auto dof_map = discretization.MapDOFLocal(cell, i, uk_man, n, g);
                data_vec.push_back(src[groupset_id][dof_map]);
                // std::cout << "Mu:Flux: " << mu_n << " : " << src[groupset_id][dof_map] << std::endl;
              }
              // Move the vector to avoid unecessary copy
              data_map[bndry_name].push_back(std::move(data_vec));
            }
          }
        }
        ++f;
      }
    }

    // Write mesh information
    if (gset == 0)
    {
      // hid_t mesh_h5 = H5CreateGroup(file_id, "mesh");
      H5CreateGroup(file_id, "mesh");
      // Per boundary id
      for (const auto& [key, node_vectors] : mesh_map) 
      {
        std::cout << "Boundary: " << key << " has " << node_vectors.size() << " node vectors." << std::endl;
        // hid_t boundary_h5 = H5CreateGroup(mesh_h5, key.c_str());
        const std::string& bndry_id = key.c_str();
        H5CreateGroup(file_id, "mesh/" +  bndry_id);

        std::vector<uint64_t> cell_ids;
        std::vector<double> nodes_x, nodes_y, nodes_z;
        for (const auto& [cell_id, vec] : node_vectors)
        {
          // std::cout << "Cell: " << cell_id << " | Pos: ";
          cell_ids.push_back(cell_id);

          for (const double val : vec)
          {
            // std::cout << val << " ";
          }
          // std::cout << std::endl;
          nodes_x.push_back(vec[0]);
          nodes_y.push_back(vec[1]);
          nodes_z.push_back(vec[2]);
        }
        std::cout << "Num X Vectors: " << nodes_x.size() << std::endl;
        H5WriteDataset1D(file_id, "mesh/" + bndry_id + "/cell_ids", cell_ids);
        H5WriteDataset1D(file_id, "mesh/" + bndry_id + "/nodes_x", nodes_x);
        H5WriteDataset1D(file_id, "mesh/" + bndry_id + "/nodes_y", nodes_y);
        H5WriteDataset1D(file_id, "mesh/" + bndry_id + "/nodes_z", nodes_z);
      }
    }

    // Write data information
    for (const auto& [key, data_vectors] : data_map) 
    {
      std::cout << "Boundary: " << key << " has " << data_vectors.size() << " data vectors." << std::endl;
      // hid_t boundary_h5 = H5CreateGroup(groupset_h5, key.c_str());
      const std::string& bndry_id = key.c_str();
      H5CreateGroup(file_id, group_name + "/" + bndry_id);

      // Write the groupset surface angular flux data
      std::vector<double> mu;
      std::vector<double> coeff;
      std::vector<double> surf_flux;
      for (const auto&vec : data_vectors)
      {
        // std::cout << "Vals: ";
        for (const double val : vec)
        {
          // std::cout << val << " ";
        }
        // std::cout << std::endl;
        mu.push_back(vec[0]);
        coeff.push_back(vec[1]);
        surf_flux.push_back(vec[2]);
      }
      H5WriteDataset1D(file_id, group_name + "/" + bndry_id + "/surf_flux", surf_flux);
      H5WriteDataset1D(file_id, group_name + "/" + bndry_id + "/mu", mu);
      H5WriteDataset1D(file_id, group_name + "/" + bndry_id + "/coeff", coeff);

      ++gset;
    }
  }
  H5Fclose(file_id);
}

void
LBSSolverIO::ReadAngularFluxes(
  LBSSolver& lbs_solver,
  const std::string& file_base,
  std::optional<std::reference_wrapper<std::vector<std::vector<double>>>> opt_dest)
{
  // Open HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "Failed to open " + file_name + ".");

  // Select destination vector
  std::vector<std::vector<double>>& dest =
    opt_dest.has_value() ? opt_dest.value().get() : lbs_solver.PsiNewLocal();

  log.Log() << "Reading angular flux file from " << file_base;

  // Read macro data and check for compatibility
  uint64_t file_num_groupsets;
  uint64_t file_num_local_cells;
  uint64_t file_num_local_nodes;

  H5ReadAttribute(file_id, "num_groupsets", file_num_groupsets);
  H5ReadAttribute(file_id, "mesh/num_local_cells", file_num_local_cells);
  H5ReadAttribute(file_id, "mesh/num_local_nodes", file_num_local_nodes);

  const auto& grid = lbs_solver.Grid();
  const auto& discretization = lbs_solver.SpatialDiscretization();
  const auto& groupsets = lbs_solver.Groupsets();

  const auto num_local_nodes = discretization.GetNumLocalNodes();
  const auto num_groupsets = groupsets.size();

  OpenSnLogicalErrorIf(file_num_local_nodes != num_local_nodes,
                       "Incompatible number of local nodes found in file " + file_name + ".");
  OpenSnLogicalErrorIf(file_num_groupsets != num_groupsets,
                       "Incompatible number of groupsets found in file " + file_name + ".");

  // Read in mesh information
  const auto file_cell_ids = H5ReadDataset1D<uint64_t>(file_id, "mesh/cell_ids");
  const auto file_num_cell_nodes = H5ReadDataset1D<uint64_t>(file_id, "mesh/num_cell_nodes");

  const auto nodes_x = H5ReadDataset1D<double>(file_id, "mesh/nodes_x");
  const auto nodes_y = H5ReadDataset1D<double>(file_id, "mesh/nodes_y");
  const auto nodes_z = H5ReadDataset1D<double>(file_id, "mesh/nodes_z");

  // Validate mesh compatibility
  uint64_t curr_node = 0;
  std::map<uint64_t, std::map<uint64_t, uint64_t>> file_cell_nodal_mapping;
  for (uint64_t c = 0; c < file_num_local_cells; ++c)
  {
    const uint64_t cell_global_id = file_cell_ids[c];
    const auto& cell = grid.cells[cell_global_id];

    if (not grid.IsCellLocal(cell_global_id))
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
    const auto values = H5ReadDataset1D<double>(file_id, group_name + "/values");
    for (uint64_t c = 0; c < file_num_local_cells; ++c)
    {
      const auto cell_global_id = file_cell_ids[c];
      const auto& cell = grid.cells[cell_global_id];
      for (uint64_t i = 0; i < discretization.GetCellNumNodes(cell); ++i)
        for (uint64_t n = 0; n < num_gs_dirs; ++n)
          for (uint64_t g = 0; g < num_gs_groups; ++g)
          {
            const auto& imap = file_cell_nodal_mapping.at(cell_global_id).at(i);
            const auto dof_map = discretization.MapDOFLocal(cell, imap, uk_man, n, g);
            psi[dof_map] = values[v];
            ++v;
          }
    }
  }
  H5Fclose(file_id);
}

void
LBSSolverIO::ReadSurfaceAngularFluxes(
  LBSSolver& lbs_solver,
  const std::string& file_base)
{
  // Open HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "Failed to open " + file_name + ".");

  // Select destination vector
  // std::vector<std::vector<double>>& dest =
  //   opt_dest.has_value() ? opt_dest.value().get() : lbs_solver.PsiNewLocal();
  std::vector<std::vector<double>>& dest = lbs_solver.PsiNewLocal();

  log.Log() << "Reading surface angular flux file from " << file_base;

  // Read macro data and check for compatibility
  uint64_t file_num_groupsets;
  uint64_t file_num_local_cells;
  uint64_t file_num_local_nodes;

  H5ReadAttribute(file_id, "num_groupsets", file_num_groupsets);

  // Read mesh information
  hid_t mesh_h5 = H5Gopen(file_id, "mesh", H5P_DEFAULT);
  hsize_t num_bndrys;
  H5Gget_num_objs(mesh_h5, &num_bndrys);
  std::vector<std::string> bndry_names;
  std::cout << "Groups inside 'mesh': " << num_bndrys << std::endl;
  for (hsize_t i = 0; i < num_bndrys; i++)
  {
      char bndry_name[256]; // Buffer to store name
      H5Gget_objname_by_idx(mesh_h5, i, bndry_name, sizeof(bndry_name));

      bndry_names.push_back(bndry_name);
      std::cout << " - " << bndry_name << std::endl;
  }

  for (const auto& bndry : bndry_names)
  {
    std::cout << "Boundary: " << bndry << std::endl;
    const auto nodes_x = H5ReadDataset1D<double>(file_id, "mesh/"+bndry+"/nodes_x");
    const auto nodes_y = H5ReadDataset1D<double>(file_id, "mesh/"+bndry+"/nodes_y");
    const auto nodes_z = H5ReadDataset1D<double>(file_id, "mesh/"+bndry+"/nodes_z");
    std::cout << "Num Nodes X: " << nodes_x.size() << std::endl;
  }

  // Read data information
  std::cout << "Reading Data" << std::endl;
  for (uint64_t gs = 0; gs < file_num_groupsets; ++gs)
  {
    const std::string groupset_name = "groupset_" + std::to_string(gs);
    hid_t groupset_h5 = H5Gopen(file_id, groupset_name.c_str(), H5P_DEFAULT);
    hsize_t num_bndrys;
    H5Gget_num_objs(groupset_h5, &num_bndrys);
    std::vector<std::string> bndry_names;
    std::cout << "Groups inside "+groupset_name+": " << num_bndrys << std::endl;
    for (hsize_t i = 0; i < num_bndrys; i++)
    {
        char bndry_name[256]; // Buffer to store name
        H5Gget_objname_by_idx(mesh_h5, i, bndry_name, sizeof(bndry_name));

        bndry_names.push_back(bndry_name);
        std::cout << " - " << bndry_name << std::endl;
    }
    for (const auto& bndry : bndry_names)
    {
      std::cout << "Boundary: " << bndry << std::endl;
      const auto mu = H5ReadDataset1D<double>(file_id, groupset_name+"/"+bndry+"/mu");
      const auto coeff = H5ReadDataset1D<double>(file_id, groupset_name+"/"+bndry+"/coeff");
      const auto psi = H5ReadDataset1D<double>(file_id, groupset_name+"/"+bndry+"/surf_flux");
      std::cout << "Num Vals: " << psi.size() << std::endl;
    }
  }
  exit(0);


  H5ReadAttribute(file_id, "mesh/num_local_cells", file_num_local_cells);
  H5ReadAttribute(file_id, "mesh/num_local_nodes", file_num_local_nodes);

  const auto& grid = lbs_solver.Grid();
  const auto& discretization = lbs_solver.SpatialDiscretization();
  const auto& groupsets = lbs_solver.Groupsets();

  const auto num_local_nodes = discretization.GetNumLocalNodes();
  const auto num_groupsets = groupsets.size();

  OpenSnLogicalErrorIf(file_num_local_nodes != num_local_nodes,
                       "Incompatible number of local nodes found in file " + file_name + ".");
  OpenSnLogicalErrorIf(file_num_groupsets != num_groupsets,
                       "Incompatible number of groupsets found in file " + file_name + ".");

  // Read in mesh information
  const auto file_cell_ids = H5ReadDataset1D<uint64_t>(file_id, "mesh/cell_ids");
  const auto file_num_cell_nodes = H5ReadDataset1D<uint64_t>(file_id, "mesh/num_cell_nodes");

  // const auto nodes_x = H5ReadDataset1D<double>(file_id, "mesh/nodes_x");
  // const auto nodes_y = H5ReadDataset1D<double>(file_id, "mesh/nodes_y");
  // const auto nodes_z = H5ReadDataset1D<double>(file_id, "mesh/nodes_z");

  // Validate mesh compatibility
  uint64_t curr_node = 0;
  std::map<uint64_t, std::map<uint64_t, uint64_t>> file_cell_nodal_mapping;
  for (uint64_t c = 0; c < file_num_local_cells; ++c)
  {
    const uint64_t cell_global_id = file_cell_ids[c];
    const auto& cell = grid.cells[cell_global_id];

    if (not grid.IsCellLocal(cell_global_id))
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
      // file_nodes.emplace_back(nodes_x[curr_node], nodes_y[curr_node], nodes_z[curr_node]);
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
    const auto values = H5ReadDataset1D<double>(file_id, group_name + "/values");
    for (uint64_t c = 0; c < file_num_local_cells; ++c)
    {
      const auto cell_global_id = file_cell_ids[c];
      const auto& cell = grid.cells[cell_global_id];
      for (uint64_t i = 0; i < discretization.GetCellNumNodes(cell); ++i)
        for (uint64_t n = 0; n < num_gs_dirs; ++n)
          for (uint64_t g = 0; g < num_gs_groups; ++g)
          {
            const auto& imap = file_cell_nodal_mapping.at(cell_global_id).at(i);
            const auto dof_map = discretization.MapDOFLocal(cell, imap, uk_man, n, g);
            psi[dof_map] = values[v];
            ++v;
          }
    }
  }
  H5Fclose(file_id);
}

} // namespace opensn
