// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

/**
 * This module is the encapsulation of the algorithm depicted in:
 *
 * [1] Golub G.H. "How to generate unknown orthogonal polynomials out of known orthogonal
 *     polynomials", Numerical Analysis Project, Stanford University, November 1991.
 *
 * Comptuting roots of the polynomial is an adaption of Newton's method described in:
 *
 * [2] Barrera-Figueroa V., et al. "Multiple root finder algorithm for Legendre and Chebyshev
 *     polynomials via Newtons method", Annales Mathematicae et Informaticae, volume 33, pages
 *     3-13, 2006.
 *
 * Finally the weights of the resulting Gauss quadrature is obtained as described in:
 *
 * [3] Sloan D.P., "A New Multigroup Monte Carlo Scattering Algorithm Suitable for Neutral and
 *     Charged-Particle Boltzmann and Fokker-Planck Calculations", SAND83-7094, PhD Dissertation,
 *     May 1983.
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

namespace opensn
{

/**
 * Implementation of the GolubFischer Modified ChebyShev Algorithm (MCA) to find
 * moment-preserving angles from a set of moments computed for the expansion of
 * an angular function in Legendre polynomials.
 *
 * Supply an even amount of moments to the method GetDiscreteScatAngles and it will populate its
 * member xn_wn which is a vector of pairs, abscissae and weights. It also return a reference to
 * xn_wn.
 *
 * alpha and beta are the recusrion coefficients of the orthogonal polynomials described in [1].
 */
class GolubFischer
{
public:
  std::vector<std::pair<double, double>> xn_wn;
  std::vector<double> alpha;
  std::vector<double> beta;

public:
  /**
   * Master callable function that will return a reference to the abscissae and weights of the
   * discrete angles.
   */
  std::vector<std::pair<double, double>>& GetDiscreteScatAngles(std::vector<double>& mell);

private:
  /**
   * Applies the Modified Chebyshev Algorithm contained in [1] to find the recursion coefficients
   * for the orthogonal polynomials.
   */
  void MCA(std::vector<double>& mell,
           std::vector<double>& a,
           std::vector<double>& b,
           std::vector<double>& c);
  /// Finds the roots of the orthogonal polynomial.
  void RootsOrtho(int& N, std::vector<double>& alpha, std::vector<double>& beta);
  /// Computes the derivative of the orthogonal polynomials.
  double dOrtho(int ell, double x, std::vector<double>& alpha, std::vector<double>& beta);
  /// Computes the function evaluation of the orthogonal polynomials.
  double Ortho(int ell, double x, std::vector<double>& alpha, std::vector<double>& beta);
};

} // namespace opensn
