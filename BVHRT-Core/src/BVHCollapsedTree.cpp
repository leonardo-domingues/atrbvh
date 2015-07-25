#include "BVHCollapsedTree.h"

namespace BVHRT
{

BVHCollapsedTree::BVHCollapsedTree(unsigned int numberOfTriangles, bool allocateMemory) : 
        numberOfTriangles(numberOfTriangles)
{
    if (allocateMemory)
    {
        unsigned int numberOfElements = 2 * numberOfTriangles - 1;
        nodes.parentIndices = new int[numberOfElements];
        nodes.leftIndices = new int[numberOfElements];
        nodes.rightIndices = new int[numberOfElements];
        nodes.dataIndices = new int[numberOfElements];
        nodes.triangleCount = new int[numberOfElements];
        nodes.boundingBoxMin = new float4[numberOfElements];
        nodes.boundingBoxMax = new float4[numberOfElements];
        triangleIndices = new int[numberOfTriangles];
    }
    else
    {
        nodes.parentIndices = nullptr;
        nodes.leftIndices = nullptr;
        nodes.rightIndices = nullptr;
        nodes.dataIndices = nullptr;
        nodes.triangleCount = nullptr;
        nodes.boundingBoxMin = nullptr;
        nodes.boundingBoxMax = nullptr;
        triangleIndices = nullptr;
    }

    rootIndex = 0;
}

BVHCollapsedTree::~BVHCollapsedTree()
{
    delete[] nodes.parentIndices;
    delete[] nodes.leftIndices;
    delete[] nodes.rightIndices;
    delete[] nodes.dataIndices;
    delete[] nodes.triangleCount;
    delete[] nodes.boundingBoxMin;
    delete[] nodes.boundingBoxMax;
    delete[] triangleIndices;
}

}
