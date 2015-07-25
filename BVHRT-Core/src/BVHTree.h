#pragma once

#include "Defines.h"

#ifdef SOA
#include "SoABVHTree.h"
namespace BVHRT
{
typedef class SoABVHTree BVHTree;
}
#endif
