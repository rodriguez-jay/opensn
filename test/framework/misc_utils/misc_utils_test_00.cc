#include "framework/utils/utils.h"

#include "framework/runtime.h"
#include "framework/logging/log.h"

#include "lua/framework/console/console.h"

using namespace opensn;

namespace unit_tests
{

ParameterBlock misc_utils_Test00(const InputParameters& params);

RegisterWrapperFunctionInNamespace(unit_tests, misc_utils_Test00, nullptr, misc_utils_Test00);

ParameterBlock
misc_utils_Test00(const InputParameters&)
{
  opensn::log.Log() << "GOLD_BEGIN";
  opensn::log.Log() << "Testing misc_utils::PrintIterationProgress\n";

  const unsigned int I = 4;
  const size_t N = 39;

  std::stringstream progress;
  for (size_t i = 0; i < N; ++i)
  {
    progress << PrintIterationProgress(i, N, I);
  }

  opensn::log.Log() << progress.str();

  opensn::log.Log() << "GOLD_END";
  return ParameterBlock();
}

} //  namespace unit_tests
