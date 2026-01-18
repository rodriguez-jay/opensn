// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/io/lbs_problem_io.h"
#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/discrete_ordinates_problem.h"
#include "framework/runtime.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/utils/hdf_utils.h"
#include "framework/math/spatial_discretization/finite_element/finite_element_data.h"


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
  uint64_t file_num_groupsets = 0;
  uint64_t file_num_local_cells = 0;
  uint64_t file_num_local_nodes = 0;

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
    uint64_t file_num_gs_dirs = 0;
    uint64_t file_num_gs_groups = 0;

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

      const auto& unit_cell_matrices = do_problem.GetUnitCellMatrices();
	    const auto& fe_values = unit_cell_matrices.at(cell.local_id);
      
      for (uint64_t i = 0; i < discretization.GetCellNumNodes(cell); ++i)
      {
        const auto& cell_mapping = discretization.GetCellMapping(cell);
        const auto& node_locations = cell_mapping.GetNodeLocations();
        const auto& node_vec = node_locations[i];
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
  }
  H5Fclose(file_id);
}

void
LBSSolverIO::WriteSurfaceAngularFluxes(
  DiscreteOrdinatesProblem& do_problem,
  const std::string& file_base,
  std::vector<std::string>& bndry_surfs,
  std::vector<std::pair<std::string, std::pair<std::string, double>>>& int_surfs)
{
  // Open the HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "WriteSurfaceAngularFluxes: Failed to open " + file_name + ".");

  // Check the user provided surfaces
  if (bndry_surfs.empty() && int_surfs.empty())
    OpenSnLogicalError("No surface provided. Provide either boundary names or internal surface definitions.");

  // Get angular fluxes
  std::vector<std::vector<double>>& psi = do_problem.GetPsiNewLocal();

  // Write macro info
  const auto& grid = do_problem.GetGrid();

  std::map<std::string, std::uint64_t> allowed_bd_names = grid->GetBoundaryNameMap();
  std::map<std::uint64_t, std::string> allowed_bd_ids = grid->GetBoundaryIDMap();

  const auto& discretization = do_problem.GetSpatialDiscretization();
  const auto& groupsets = do_problem.GetGroupsets();

  auto num_local_cells = grid->local_cells.size();
  auto num_local_nodes = discretization.GetNumLocalNodes();
  auto num_groupsets = groupsets.size();

  // Check the boundary IDs
  std::vector<uint64_t> bndry_ids;
  if (!bndry_surfs.empty())
  {
    const auto unique_bids = grid->GetUniqueBoundaryIDs();
    for (const auto& bndry : bndry_surfs)
    {
      // Verify if supplied boundary has a valid boundary ID
      const auto bndry_id = allowed_bd_names.at(bndry);
      const auto id = std::find(unique_bids.begin(), unique_bids.end(), bndry_id);
      OpenSnInvalidArgumentIf(id == unique_bids.end(),
                              "Boundary " + bndry + "not found on grid.");
      bndry_ids.push_back(bndry_id);
    }
  }

  log.Log() << "Writing surface angular flux data to " << file_base;

  // Get mapping for the surface data
  std::unordered_set<std::string> surf_tags;
  std::map<std::string, std::vector<uint64_t>> cell_map, node_map;
  std::map<std::string, std::vector<double>> x_map, y_map, z_map;

  // Go through each groupset
  unsigned int gset = 0;
  for (const auto& groupset : groupsets)
  {
    // Write groupset info
    std::map<std::string, std::vector<std::vector<double>>> data_map;
    std::map<std::string, std::vector<double>> mass_map;

    auto groupset_id = groupset.id;
    const auto& uk_man = groupset.psi_uk_man_;
    const auto& quadrature = groupset.quadrature;
    const auto& num_gs_dirs = quadrature->omegas.size();
    auto num_gs_groups = groupset.groups.size();

    // Loop over all cells
    for (const auto& cell : grid->local_cells)
    {  
      const uint64_t& cell_id = cell.global_id;
      const auto& cell_mapping = discretization.GetCellMapping(cell);
      const auto& node_locations = cell_mapping.GetNodeLocations();
      uint64_t num_cell_nodes = 0;

      const auto fe_vol_data = cell_mapping.MakeVolumetricFiniteElementData();
      const auto& unit_cell_matrices = do_problem.GetUnitCellMatrices();
	    const auto& fe_values = unit_cell_matrices.at(cell.local_id);

      // Go through each face of the cell
      unsigned int f = 0;
      for (const auto& face : cell.faces)
      {
        bool isSurf = false;
        std::string surf_name;

        // Internal surface mapping
        if (!int_surfs.empty())
        {
          for (const auto& surface : int_surfs)
          {
            const std::string& surf_id = surface.first;
            const std::string& axis = surface.second.first;
            const double& slice = surface.second.second;

            const auto num_face_nodes = cell_mapping.GetNumFaceNodes(f);
            unsigned int nodes_on_face = 0;
            for (unsigned int fi = 0; fi < num_face_nodes; ++fi)
            {
              const auto i = cell_mapping.MapFaceNode(f, fi);
              const auto& node_vec = node_locations[i];

              // Error message for not being on slice!!!
              bool on_axis = false;
              if (axis == "x") on_axis = (node_vec.x == slice);
              else if (axis == "y") on_axis = (node_vec.y == slice);
              else if (axis == "z") on_axis = (node_vec.z == slice);
              else OpenSnLogicalError("Invalid axis provided.");

              if (on_axis)
                ++nodes_on_face; 
            }

            // Ensure each node is on the prescribed face.
            if (nodes_on_face == num_face_nodes)
            {
              // Assign an upwinding or downwinding tag to the surface
              const auto& omega_0 = quadrature->omegas[0]; 
              const auto mu_0 = omega_0.Dot(face.normal);
              surf_name = surf_id + (mu_0 > 0 ? "_u" : "_d");
              isSurf = true;
            }
          }
        }

        // Boundary surface mapping
        const auto it = std::find(bndry_ids.begin(), bndry_ids.end(), face.neighbor_id);
        if (!bndry_surfs.empty() && not face.has_neighbor and it != bndry_ids.end())
        {
          surf_name = allowed_bd_ids.at(*it);
          isSurf = true;
        }

        // Write suface data 
        if (isSurf)
        {
          surf_tags.insert(surf_name);

          const auto& int_f_shape_i = fe_values.intS_shapeI[f];
          const auto& M_ij = fe_values.intS_shapeI_shapeJ[f];
          const uint64_t& num_face_nodes = cell_mapping.GetNumFaceNodes(f);


          const auto fe_srf_data = cell_mapping.MakeSurfaceFiniteElementData(f);
          // for (const auto& qp : fe_srf_data.GetQuadraturePointIndices())
          // {
          //   std::cout << "F " << f <<" QP : " << qp << " QPoints : "
          //             << fe_vol_data.QPointXYZ(qp)[0] << " " 
          //             << fe_vol_data.QPointXYZ(qp)[1] << " "
          //             << fe_vol_data.QPointXYZ(qp)[2] << " " 
          //             << std::endl;
          // }
          // // exit(0);

          cell_map[surf_name].push_back(cell_id);
          node_map[surf_name].push_back(num_face_nodes);

          num_cell_nodes += num_face_nodes;
          for (unsigned int fi = 0; fi < num_face_nodes; ++fi)
          {
            uint64_t i = cell_mapping.MapFaceNode(f, fi);
            const auto& node_vec = node_locations[i];
            if (gset == 0)
            {
              x_map[surf_name].push_back(node_vec[0]);
              y_map[surf_name].push_back(node_vec[1]);
              z_map[surf_name].push_back(node_vec[2]);

              // std::cout << "Points : "
              //         << node_vec[0] << " " 
              //         << node_vec[1] << " "
              //         << node_vec[2] << " " 
              //         << std::endl;

              // std::cout << "F " << fi << " QPoints : "
              //         << fe_vol_data.QPointXYZ(fi)[0] << " " 
              //         << fe_vol_data.QPointXYZ(fi)[1] << " "
              //         << fe_vol_data.QPointXYZ(fi)[2] << " " 
              //         << std::endl;

              // x_map[surf_name].push_back(fe_vol_data.QPointXYZ(fi)[0]);
              // y_map[surf_name].push_back(fe_vol_data.QPointXYZ(fi)[1]);
              // z_map[surf_name].push_back(fe_vol_data.QPointXYZ(fi)[2]);
            }

            for (unsigned int d = 0; d < num_gs_dirs; ++d)
            {
              const auto& omega_d = quadrature->omegas[d];
              const auto& weight_d = quadrature->weights[d];
              const auto mu_d = omega_d.Dot(face.normal);
              std::vector<double> data_vec;
              data_vec.insert(data_vec.end(), {omega_d.x, omega_d.y, omega_d.z});
              data_vec.push_back(mu_d);
              data_vec.push_back(weight_d);
              data_vec.push_back(int_f_shape_i(i));
              for (uint64_t g = 0; g < num_gs_groups; ++g)
              {
                const auto dof_map = discretization.MapDOFLocal(cell, i, uk_man, d, g);
                // std::cout << surf_name << std::endl;
                // if (surf_name == "inter_y_u")
                // {
                //   std::cout << "X : " << node_vec[0] << std::endl;
                //   std::cout << "Y : " << node_vec[1] << std::endl;
                //   std::cout << "Psi: " << psi[groupset_id][dof_map] << std::endl;
                // }
                data_vec.push_back(psi[groupset_id][dof_map]);
              }
              // Move the vector to avoid unecessary copy
              data_map[surf_name].push_back(std::move(data_vec));
            }

            for (unsigned int fj = 0; fj < num_face_nodes; ++fj)
            {
              const auto j = cell_mapping.MapFaceNode(f, fj);
              mass_map[surf_name].push_back(M_ij(i, j));
            }
          } // for face node
        }
        ++f;
      } // for face
    } // for cell

    // Export data to HDF5
    H5CreateAttribute(file_id, "num_groupsets", num_groupsets);
    std::string group_name = "groupset_" + std::to_string(groupset_id);
    H5CreateGroup(file_id, group_name);
    H5CreateAttribute(file_id, group_name + "/num_directions", num_gs_dirs);
    H5CreateAttribute(file_id, group_name + "/num_groups", num_gs_groups);

    H5CreateGroup(file_id, "mesh");
    for (const auto& surf_id : surf_tags)
    {
      if (gset == 0)
      {
        const auto& cell_ids = cell_map.at(surf_id);
        const auto& num_face_nodes = node_map.at(surf_id);
        const auto& x_surf = x_map.at(surf_id);
        const auto& y_surf = y_map.at(surf_id);
        const auto& z_surf = z_map.at(surf_id);

        std::string surf_mesh = std::string("mesh/") + surf_id;
        H5CreateGroup(file_id, surf_mesh);
        H5WriteDataset1D(file_id, surf_mesh + "/cell_ids", cell_ids);
        H5WriteDataset1D(file_id, surf_mesh + "/num_face_nodes", num_face_nodes);
        H5WriteDataset1D(file_id, surf_mesh + "/nodes_x", x_surf);
        H5WriteDataset1D(file_id, surf_mesh + "/nodes_y", y_surf);
        H5WriteDataset1D(file_id, surf_mesh + "/nodes_z", z_surf);
      }

      std::string surf_grp = group_name + std::string("/") + surf_id;
      H5CreateGroup(file_id, surf_grp);

      // Write the groupset surface angular flux data
      std::vector<double> omega;
      std::vector<double> mu;
      std::vector<double> wt_d;
      std::vector<double> fe_shape;
      std::vector<double> surf_flux;

      const auto& data_vectors = data_map.at(surf_id);
      for (const auto&vec : data_vectors)
      {
        omega.insert(omega.end(), {vec[0], vec[1], vec[2]});
        mu.push_back(vec[3]);
        wt_d.push_back(vec[4]);
        fe_shape.push_back(vec[5]);
        surf_flux.insert(surf_flux.end(), vec.begin()+6, vec.end());
      }

      // std::vector<double> omegas;
      // std::vector<double> weights;
      // for (unsigned int d = 0; d < num_gs_dirs; ++d)
      // {
      //   const auto& omega_d = quadrature->omegas[d];
      //   omegas.insert(omegas.end(), {omega_d.x, omega_d.y, omega_d.z});
      //   weights.push_back(quadrature->weights[d]);
      // }
      // H5WriteDataset1D(file_id, surf_grp + "/omega", omegas);
      // H5WriteDataset1D(file_id, surf_grp + "/wt_d", weights);
      
      H5WriteDataset1D(file_id, surf_grp + "/omega", omega);
      H5WriteDataset1D(file_id, surf_grp + "/wt_d", wt_d);
      H5WriteDataset1D(file_id, surf_grp + "/mu", mu);
      H5WriteDataset1D(file_id, surf_grp + "/fe_shape", fe_shape);
      H5WriteDataset1D(file_id, surf_grp + "/surf_flux", surf_flux);

      // Write mass matrix information
      const auto& mass_vector = mass_map.at(surf_id);
      H5WriteDataset1D(file_id, surf_grp + "/M_ij", mass_vector);
    }
    ++gset;
  } // for groupset

  ssize_t num_open_objs = H5Fget_obj_count(file_id, H5F_OBJ_ALL);
  H5Fclose(file_id);
}

std::vector<LBSSolverIO::SurfaceAngularFlux>
LBSSolverIO::ReadSurfaceAngularFluxes(
  DiscreteOrdinatesProblem& do_problem,
  const std::string& file_base,
  std::vector<std::string>& surfaces)
{
  std::vector<SurfaceAngularFlux> surf_fluxes;

  // Open HDF5 file
  std::string file_name = file_base + std::to_string(opensn::mpi_comm.rank()) + ".h5";
  hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  OpenSnLogicalErrorIf(file_id < 0, "Failed to open " + file_name + ".");

  log.Log() << "Reading surface angular flux file from " << file_base;

  // Read macro data and check for compatibility
  uint64_t file_num_groupsets;
  H5ReadAttribute(file_id, "num_groupsets", file_num_groupsets);

  const auto& grid = do_problem.GetGrid();
  const auto& discretization = do_problem.GetSpatialDiscretization();
  const auto& groupsets = do_problem.GetGroupsets();
  auto num_groupsets = groupsets.size();
  OpenSnLogicalErrorIf(file_num_groupsets != num_groupsets,
                       "Incompatible number of groupsets found in file " + file_name + ".");

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

    // Read each surface for a given groupset
    for (const auto& surface : surfaces)
    {
      SurfaceMap surf_map;
      SurfaceData surf_data;
      SurfaceAngularFlux surf_flux;

      std::string mesh_tag = "mesh/" + surface;
      H5ReadDataset1D<double>(file_id, mesh_tag + "/cell_ids", surf_map.cell_ids);
      H5ReadDataset1D<double>(file_id, mesh_tag + "/num_face_nodes", surf_map.num_face_nodes);
      H5ReadDataset1D<double>(file_id, mesh_tag + "/nodes_x", surf_map.nodes_x);
      H5ReadDataset1D<double>(file_id, mesh_tag + "/nodes_y", surf_map.nodes_y);
      H5ReadDataset1D<double>(file_id, mesh_tag + "/nodes_z", surf_map.nodes_z);

      std::string surf_grp = group_name + "/" + surface;
      H5ReadDataset1D<double>(file_id, surf_grp + "/omega", surf_data.omega);
      H5ReadDataset1D<double>(file_id, surf_grp + "/mu", surf_data.mu);
      H5ReadDataset1D<double>(file_id, surf_grp + "/wt_d", surf_data.wt_d);
      H5ReadDataset1D<double>(file_id, surf_grp + "/M_ij", surf_data.M_ij);
      H5ReadDataset1D<double>(file_id, surf_grp + "/surf_flux", surf_data.psi);

      // Get strides
      std::vector<double> cell_index = {0};
      std::vector<double> node_index = {0};
      std::vector<double> dir_index = {0};
      size_t node_indx = 0;
      double stride = 0;
      auto num_cells = surf_map.cell_ids.size();
      for (size_t ci = 0; ci < num_cells; ++ci)
      {
        auto& num_face_nodes = surf_map.num_face_nodes[ci];
        for (size_t ni = 0; ni < num_face_nodes; ++ni)
        {
          for (size_t d = 0; d < file_num_gs_dirs; ++d)
          {
            for (size_t g = 0; g < file_num_gs_groups; ++g)
              ++stride;
            if ((d + 1 < file_num_gs_dirs) || (ni + 1 < num_face_nodes) || (ci + 1 < num_cells))
              dir_index.push_back(stride);
          }

          if ((ni + 1 < num_face_nodes) || (ci + 1 < num_cells))
            node_index.push_back(stride);
        }
        if (ci + 1 < num_cells)
          cell_index.push_back(stride);
        node_indx += num_face_nodes;
      }
      surf_data.node_index = node_index;
      surf_data.dir_index = dir_index;

      surf_flux.mapping = std::move(surf_map);
      surf_flux.data = std::move(surf_data);
      surf_fluxes.push_back(std::move(surf_flux));
    }
  }
  H5Fclose(file_id);

  return surf_fluxes;
}

} // namespace opensn
