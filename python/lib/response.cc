// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "python/lib/py_wrappers.h"
#include "modules/linear_boltzmann_solvers/response_evaluator/response_evaluator.h"
#include <pybind11/stl.h>
#include <memory>

namespace opensn
{

// Wrap response evaluator
void
WrapResEval(py::module& response)
{
  // clang-format off
  // response evaluator
  auto res_eval = py::class_<ResponseEvaluator, std::shared_ptr<ResponseEvaluator>>(
    response,
    "ResponseEvaluator",
    R"(
    Response evaluator by folding sources against adjoint solutions.

    Wrapper of :cpp:class:`opensn::ResponseEvaluator`.

    The workflow for this utility is constructed to minimize the file reading necessary for
    evaluations. To begin, one should add all adjoint solutions that are desired for response
    computations into the buffer. Then, one should define the different forward source
    configurations of interest. With this, the user can now iterate over the source
    configurations and convolve them against all available adjoint solutions in the buffer.
    )"
  );
  res_eval.def(
    py::init(
      [](py::kwargs& params)
      {
        return ResponseEvaluator::Create(kwargs_to_param_block(params));
      }
    ),
    R"(
    Construct a response evaluator object.

    Parameters
    ----------
    problem: pyopensn.solver.DiscreteOrdinatesProblem
        A handle to an existing LBS problem.
    options: Dict
        The specification of adjoint buffers and forward sources to use. Supported keys:
          - buffers: List[Dict], default=[]
              Each dictionary supports:
                - name: str (required)
                    Buffer name used for response queries.
                - file_prefixes: Dict (required)
                    Prefix map with optional keys:
                      - flux_moments: str
                          Prefix for flux moments files.
                      - angular_fluxes: str
                          Prefix for angular flux files.
          - clear_sources: bool, default=False
              If true, clears any existing forward sources.
          - sources: Dict, default={}
              Dictionary describing forward sources. Supported keys:
                - material: List[Dict], default=[]
                    Each dictionary supports:
                      - block_id: int (required)
                          Block id for the material source.
                      - strength: List[float] (required)
                          Group-wise source strength (length must match number of groups).
                - point: List[pyopensn.source.PointSource], default=[]
                    Point source handles.
                - volumetric: List[pyopensn.source.VolumetricSource], default=[]
                    Volumetric source handles.
                - boundary: List[Dict], default=[]
                    Boundary source dictionaries. Each dictionary supports:
                      - name: str (required)
                          Boundary name that identifies the specific boundary.
                      - type: {'vacuum', 'isotropic', 'reflecting'} (required)
                          Boundary type specification. Only ``isotropic`` is used for responses.
                      - group_strength: List[float], optional
                          Required when ``type='isotropic'``. Isotropic strength per group.
                      - function_name: str, optional, default=''
                          Name of a registered function to evaluate on the boundary.
    )"
  );
  res_eval.def(
    "EvaluateResponse",
    &ResponseEvaluator::EvaluateResponse,
    R"(
    Evaluate a response using the specified adjoint buffer with the currently defined sources in
    the solver.

    Parameters
    ----------
    buffer_name: str
        Adjoint buffer name
    )",
    py::arg("buffer_name")
  );
  res_eval.def(
    "EvaluateSurfaceResponse",
    &ResponseEvaluator::EvaluateSurfaceResponse,
    R"(
    Evaluate the surface inner product of forward and adjoint surface angular fluxes.

    Both data sets must have been added with ``SetOptions(data=...)`` using the
    ``surface_angular_fluxes`` prefix. Faces are matched by location, so the two data sets may have
    been written by different numbers of ranks and partitions, and neither needs to match the
    partitioning of the current run.

    For a boundary with outward normal :math:`\hat{n}`, the response is

    .. math::

       R = \sum_{d:\, \mu_d > 0} \mu_d w_d \sum_{ij} M_{ij} \sum_g
           \psi_{i,d,g}\, \psi^\dagger_{j,d,g},

    with :math:`\mu_d = \Omega_d \cdot \hat{n}` and :math:`M_{ij}` the face mass matrix. An
    interior surface stored under tags ``<name>_u`` and ``<name>_d`` takes, for each direction, the
    forward flux from the side it leaves and the adjoint flux from the other side, with
    :math:`\mu_d` measured against the ``_u`` normal (the positive axis):

    .. math::

       R = \sum_{d:\, \mu_d > 0} \mu_d w_d \sum_{ij} M_{ij} \sum_g
           \psi^{u}_{i,d,g}\, \psi^{\dagger, d}_{j,d,g}
         + \sum_{d:\, \mu_d < 0} \mu_d w_d \sum_{ij} M_{ij} \sum_g
           \psi^{d}_{i,d,g}\, \psi^{\dagger, u}_{j,d,g}.

    Forward and adjoint directions are paired by equal direction vectors. Per-surface terms are
    logged at verbosity 1. This call is collective.

    Parameters
    ----------
    fwd_data_name: str
        Name of the forward data set. It must have been written from a forward problem.
    adj_data_name: str
        Name of the adjoint data set. It must have been written from an adjoint problem.
    surfaces: List[str], default=[]
        Boundary names and interior surface names (without ``_u``/``_d``) to evaluate. If empty,
        every surface in the forward data is evaluated: tag pairs ``<name>_u``/``<name>_d`` form an
        interior surface and any other tag is a boundary.

    Returns
    -------
    float
        The response summed over the selected surfaces.

    Raises
    ------
    ValueError
        If a data set or surface is unknown, a required tag is missing from the adjoint data, or
        the data sets have incompatible quadratures, groupsets, adjoint flags, or face keys.
    RuntimeError
        If the forward and adjoint data do not cover the same faces.
    )",
    py::arg("fwd_data_name"),
    py::arg("adj_data_name"),
    py::arg("surfaces") = std::vector<std::string>{}
  );
  res_eval.def(
    "SetOptions",
    [](ResponseEvaluator& self, py::kwargs& params)
    {
      InputParameters input = ResponseEvaluator::GetOptionsBlock();
      input.AssignParameters(kwargs_to_param_block(params));
      self.SetOptions(input);
    },
    R"(
    Set options for the response evaluator for adding adjoint buffers and defining forward sources.

    Parameters
    ----------
    data: List[Dict], default=[]
        A list of dictionaries containing data specifications. Each dictionary supports:
          - name: str (required)
              Data name used for response queries.
          - file_prefixes: Dict (required)
              Prefix map with optional keys:
                - flux_moments: str
                    Prefix for flux moments files.
                - angular_fluxes: str
                    Prefix for angular flux files.
                - surface_angular_fluxes: str
                    Prefix for the files written by ``WriteSurfaceAngularFluxes``. All files of the
                    prefix are read, whatever number of ranks wrote them, and their faces are
                    distributed over the current ranks. Reading them is collective.
          - surfaces: List[str], default=[]
              Stored surface tags to read. Required with ``surface_angular_fluxes``. Interior
              surfaces are stored as ``<name>_u`` and ``<name>_d``.
    buffers: List[Dict], default=[]
        A list of dictionaries containing adjoint buffer specifications. Each dictionary supports:
          - name: str (required)
              Buffer name used for response queries.
          - file_prefixes: Dict (required)
              Prefix map with optional keys:
                - flux_moments: str
                    Prefix for flux moments files.
                - angular_fluxes: str
                    Prefix for angular flux files.
    clear_sources: bool, default=False
        A flag to clear existing sources.
    sources: Dict, default={}
        Dictionary describing forward sources. Supported keys:
          - material: List[Dict], default=[]
              Each dictionary supports:
                - block_id: int (required)
                    Block id for the material source.
                - strength: List[float] (required)
                    Group-wise source strength (length must match number of groups).
          - point: List[pyopensn.source.PointSource], default=[]
              Point source handles.
          - volumetric: List[pyopensn.source.VolumetricSource], default=[]
              Volumetric source handles.
          - boundary: List[Dict], default=[]
              Boundary source dictionaries. Each dictionary supports:
                - name: str (required)
                    Boundary name that identifies the specific boundary.
                - type: {'vacuum', 'isotropic', 'reflecting'} (required)
                    Boundary type specification. Only ``isotropic`` is used for responses.
                - group_strength: List[float], optional
                    Required when ``type='isotropic'``. Isotropic strength per group.
                - function_name: str, optional, default=''
                    Name of a registered function to evaluate on the boundary.
    )"
    );
  res_eval.def(
    "ClearForwardSources",
    &ResponseEvaluator::ClearForwardSources,
    "Clear the existing forward sources from the response evaluator."
  );
  res_eval.def(
    "SetBufferOptions",
    [](ResponseEvaluator& self, py::kwargs& params)
    {
      InputParameters input = ResponseEvaluator::GetBufferOptionsBlock();
      input.AssignParameters(kwargs_to_param_block(params));
      self.SetBufferOptions(input);
    },
    R"(
    Add adjoint buffers from the options block to the response evaluator.

    Parameters
    ----------
    name: str (required)
        Buffer name used for response queries.
    file_prefixes: Dict (required)
        Prefix map with optional keys:
          - flux_moments: str
              Prefix for flux moments files.
          - angular_fluxes: str
              Prefix for angular flux files.
    )"
  );
  res_eval.def(
    "SetSourceOptions",
    [](ResponseEvaluator& self, py::kwargs& params)
    {
      InputParameters input = ResponseEvaluator::GetSourceOptionsBlock();
      input.AssignParameters(kwargs_to_param_block(params));
      self.SetSourceOptions(input);
    },
    R"(
    Add forward sources from the options block to the response evaluator.

    Parameters
    ----------
    material: List[Dict], default=[]
        Each dictionary supports:
          - block_id: int (required)
              Block id for the material source.
          - strength: List[float] (required)
              Group-wise source strength (length must match number of groups).
    point: List[pyopensn.source.PointSource], default=[]
        Point source handles.
    volumetric: List[pyopensn.source.VolumetricSource], default=[]
        Volumetric source handles.
    boundary: List[Dict], default=[]
        Boundary source dictionaries. Each dictionary supports:
          - name: str (required)
              Boundary name that identifies the specific boundary.
          - type: {'vacuum', 'isotropic', 'reflecting'} (required)
              Boundary type specification. Only ``isotropic`` is used for responses.
          - group_strength: List[float], optional
              Required when ``type='isotropic'``. Isotropic strength per group.
          - function_name: str, optional, default=''
              Name of a registered function to evaluate on the boundary.
    )"
  );
  // clang-format on
}

// Wrap the response components of OpenSn
void
py_response(py::module& pyopensn)
{
  py::module response = pyopensn.def_submodule("response", "Response module.");
  WrapResEval(response);
}

} // namespace opensn
