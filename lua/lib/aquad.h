// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/quadratures/angular/product_quadrature.h"
#include "framework/math/quadratures/angular/angular_quadrature.h"

namespace opensnlua
{

std::shared_ptr<opensn::ProductQuadrature>
AQuadCreateProductQuadrature(opensn::ProductQuadratureType type, int n, int m);


std::shared_ptr<opensn::ProductQuadrature>
AQuadCreateCylindricalProductQuadrature(opensn::ProductQuadratureType type, int n, int m);

void AQuadOptimizeForPolarSymmetry(std::shared_ptr<opensn::AngularQuadrature> aquad,
                                   double normalization);

std::shared_ptr<opensn::ProductQuadrature>
AQuadCreateSLDFESQAngularQuadrature(int level);

// void AQuadLocallyRefineSLDFE(std::shared_ptr<opensn::ProductQuadrature> aquad, 
//                              const Vector3& ref_dir,
//                              const double cone_size,
//                              const bool dir_as_plane_normal);

} // namespace opensnlua
