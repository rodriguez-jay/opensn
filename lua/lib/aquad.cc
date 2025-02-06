// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "lua/lib/aquad.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include "framework/math/quadratures/gausslegendre_quadrature.h"
#include "framework/math/quadratures/gausschebyshev_quadrature.h"

// New
#include "framework/math/quadratures/angular/sldfe_sq_quadrature.h"
//

#include "framework/math/quadratures/angular/cylindrical_quadrature.h"
#include <cstddef>
#include <memory>

using namespace opensn;

namespace opensnlua
{

std::shared_ptr<ProductQuadrature>
AQuadCreateProductQuadrature(ProductQuadratureType type, int n, int m)
{
  if (type == ProductQuadratureType::GAUSS_LEGENDRE)
  {
    bool verbose = false;
    auto new_quad = std::make_shared<AngularQuadratureProdGL>(n, verbose);
    return new_quad;
  }
  else if (type == ProductQuadratureType::GAUSS_LEGENDRE_CHEBYSHEV)
  {
    bool verbose = false;
    std::cout << "here" << std::endl;
    auto new_quad = std::make_shared<AngularQuadratureProdGLC>(n, m, verbose);
    return new_quad;
  }

  opensn::log.LogAllError()
    << "In call to CreateProductQuadrature. Unsupported quadrature type supplied. Given: "
    << (int)type;
  opensn::Exit(EXIT_FAILURE);
  return nullptr;
}

std::shared_ptr<opensn::ProductQuadrature>
AQuadCreateCylindricalProductQuadrature(ProductQuadratureType type, int Np, int Na)
{
  bool verbose = false;
  std::vector<int> vNa;
  vNa.resize(Np, Na);

  switch (type)
  {
    case ProductQuadratureType::GAUSS_LEGENDRE_CHEBYSHEV:
    {
      opensn::log.Log() << "CreateCylindricalProductQuadrature : "
                        << "Creating Gauss-Legendre-Legendre Quadrature\n";

      const auto quad_pol = GaussLegendreQuadrature(Np, verbose);
      std::vector<GaussQuadrature> quad_azi;
      for (const auto& Na : vNa)
        quad_azi.emplace_back(GaussChebyshevQuadrature(Na, verbose));
      const auto new_quad = std::make_shared<CylindricalQuadrature>(quad_pol, quad_azi, verbose);

      return new_quad;
    }

    default:
    {
      opensn::log.LogAllError() << "CreateCylindricalProductQuadrature : "
                                << "Unsupported quadrature type supplied, type="
                                << static_cast<int>(type);
      opensn::Exit(EXIT_FAILURE);
    }
  }
  return nullptr;
}

void
AQuadOptimizeForPolarSymmetry(std::shared_ptr<AngularQuadrature> aquad, double normalization)
{
  if (normalization > 0.0)
    opensn::log.Log() << "Optimizing angular quadrature for polar symmetry. using "
                      << "normalization factor " << normalization << ".";

  aquad->OptimizeForPolarSymmetry(normalization);
}

///////
// NEW
///////
std::shared_ptr<opensn::ProductQuadrature>
AQuadCreateSLDFESQAngularQuadrature(int level)
{
  bool verbose = false;
  auto sldfesq = std::make_shared<SimplifiedLDFESQ::Quadrature>();
  sldfesq->GenerateInitialRefinement(level);

  return nullptr;
}

// void
// AQuadLocallyRefineSLDFESQ(std::shared_ptr<ProductQuadrature> aquad, 
//                         const Vector3& ref_dir,
//                         const double cone_size,
//                         const bool dir_as_plane_normal)
// {

//   auto sldfesq = std::make_shared<SimplifiedLDFESQ::Quadrature>();
//   sldfesq->LocallyRefine(ref_dir, cone_size, ref_dir_as_plane_normal);

//   // Maybe add a type? 
//   // try
//   // {
//   //   auto ref_quadrature = opensn::angular_quadrature_stack.at(handle);
//   //   if (ref_quadrature->type == AngularQuadratureType::SLDFESQ)
//   //   {
//   //     // auto sldfesq = std::dynamic_pointer_cast<SimplifiedLDFESQ::Quadrature>(ref_quadrature);
//   //     auto sldfesq = std::make_shared<SimplifiedLDFESQ::Quadrature>();
//   //     sldfesq->LocallyRefine(ref_dir, cone_size, ref_dir_as_plane_normal);
//   //   }
//   //   else
//   //   {
//   //     opensn::log.LogAllError() << "LocallyRefineSLDFESQAngularQuadrature: "
//   //                                  "Invalid angular quadrature type.";
//   //     opensn::Exit(EXIT_FAILURE);
//   //   }
//   // }
//   // catch (const std::out_of_range& o)
//   // {
//   //   opensn::log.LogAllError() << "LocallyRefineSLDFESQAngularQuadrature: "
//   //                                "Invalid handle to angular quadrature.";
//   //   opensn::Exit(EXIT_FAILURE);
//   // }
//   // catch (...)
//   // {
//   //   opensn::log.LogAllError() << "LocallyRefineSLDFESQAngularQuadrature: "
//   //                                "Call failed with unknown error.";
//   //   opensn::Exit(EXIT_FAILURE);
//   // }
// }

//////




} // namespace opensnlua
