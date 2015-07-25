#include "SoABVHTree.h"

#include "CudaErrorCheck.h"
#include <fstream>

namespace BVHRT
{

SoABVHTree::SoABVHTree(unsigned int numberOfTriangles,
                       bool allocateMemory) : numberOfTriangles(numberOfTriangles)
{
    // Allocate host tree
    if (allocateMemory)
    {
        unsigned int numberOfElements = 2 * numberOfTriangles - 1;
        data.parentIndices = new int[numberOfElements];
        data.leftIndices = new int[numberOfElements];
        data.rightIndices = new int[numberOfElements];
        data.dataIndices = new int[numberOfElements];
        data.boundingBoxMin = new float4[numberOfElements];
        data.boundingBoxMax = new float4[numberOfElements];
        data.area = new float[numberOfElements];
    }
    else
    {
        data.parentIndices = nullptr;
        data.leftIndices = nullptr;
        data.rightIndices = nullptr;
        data.dataIndices = nullptr;
        data.boundingBoxMin = nullptr;
        data.boundingBoxMax = nullptr;
        data.area = nullptr;
    }

    data.rootIndex = 0;
}

SoABVHTree::~SoABVHTree()
{
    // Free host tree
    delete[] data.parentIndices;
    delete[] data.leftIndices;
    delete[] data.rightIndices;
    delete[] data.dataIndices;
    delete[] data.boundingBoxMin;
    delete[] data.boundingBoxMax;
    delete[] data.area;
}

float SoABVHTree::SAH(float nodeTraversalCost, float triangleIntersectionCost) const
{
    int root = data.rootIndex;

    float totalSA = data.area[root];
    float sumInternalSA = totalSA;
    float sumLeavesSA = 0.0;

    // Internal nodes
    for (unsigned int i = 0; i < numberOfTriangles - 1; ++i)
    {
        if (i != root)
        {
            sumInternalSA += data.area[i];
        }
    }

    // Leaves
    for (unsigned int i = numberOfTriangles - 1; i < 2 * numberOfTriangles - 1; ++i)
    {
        sumLeavesSA += data.area[i];
    }

    return (nodeTraversalCost * sumInternalSA + triangleIntersectionCost * sumLeavesSA) / totalSA;
}

void SoABVHTree::DumpTree(const char* fileLocation) const
{
    std::ofstream file(fileLocation);

    file << numberOfTriangles << std::endl;
    for (unsigned int i = 0; i < numberOfTriangles * 2 - 1; i++)
    {
        file << "i: " << i << " Data: " << data.dataIndices[i] << " Left: " << data.leftIndices[i] <<
             " Right: " << data.rightIndices[i] << " Parent: " << data.parentIndices[i] << " ";
        file << "BBoxMin: " << data.boundingBoxMin[i].x << " " << data.boundingBoxMin[i].y << " " <<
             data.boundingBoxMin[i].z;
        file << " BBoxMax: " << data.boundingBoxMax[i].x << " " << data.boundingBoxMax[i].y << " " <<
             data.boundingBoxMax[i].z << std::endl;
    }

    file.close();
}

}
