#include "CubWrapper.h"

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include "Defines.h"

namespace BVHRT
{

// In and out buffers may be swaped
// Original data is not kept
template <typename T> float DeviceSort(unsigned int numberOfElements, T** keysIn, T** keysOut,
                 unsigned int** valuesIn, unsigned int** valuesOut)
{
    cub::DoubleBuffer<T> keysBuffer(*keysIn, *keysOut);
    cub::DoubleBuffer<unsigned int> valuesBuffer(*valuesIn, *valuesOut);

    // Check how much temporary memory will be required
    void* tempStorage = nullptr;
    size_t storageSize = 0;
    // cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
    // numberOfElements);
    cub::DeviceRadixSort::SortKeys(tempStorage, storageSize, keysBuffer, numberOfElements);

    // Allocate temporary memory
    cudaMalloc(&tempStorage, storageSize);

    float elapsedTime = 0.0f;
#ifdef MEASURE_EXECUTION_TIMES
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif

    // Sort
    cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer,
                                    numberOfElements);

#ifdef MEASURE_EXECUTION_TIMES
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
#endif

    // Free temporary memory
    cudaFree(tempStorage);

    // Update out buffers
    T* current = keysBuffer.Current();
    keysOut = &current;
    unsigned int* current2 = valuesBuffer.Current();
    valuesOut = &current2;

    // Update in buffers
    current = keysBuffer.d_buffers[1 - keysBuffer.selector];
    keysIn = &current;
    current2 = valuesBuffer.d_buffers[1 - valuesBuffer.selector];
    valuesIn = &current2;

    return elapsedTime;
}

float DeviceSort(unsigned int numberOfElements, unsigned int** keysIn, unsigned int** keysOut,
    unsigned int** valuesIn, unsigned int** valuesOut)
{
    return DeviceSort<unsigned int>(numberOfElements, keysIn, keysOut, valuesIn, valuesOut);
}

float DeviceSort(unsigned int numberOfElements, unsigned long long int** keysIn, unsigned long long int** keysOut,
    unsigned int** valuesIn, unsigned int** valuesOut)
{
    return DeviceSort<unsigned long long int>(numberOfElements, keysIn, keysOut, valuesIn, valuesOut);
}

void DeviceSum(unsigned int numberOfTriangles, int* in, int* out, size_t* tempMemorySize, void* tempMemory)
{
    cub::DeviceReduce::Sum(tempMemory, *tempMemorySize, in, out, numberOfTriangles);
}

template <class T> T DeviceSum(unsigned int numberOfElements, T* elements)
{
    T* deviceElementsSum;
    cudaMalloc(&deviceElementsSum, sizeof(T));

    // Calculate the required temporary memory size
    void* tempStorage = nullptr;
    size_t tempStorageSize = 0;
    cub::DeviceReduce::Sum(tempStorage, tempStorageSize, elements, deviceElementsSum,
                           numberOfElements);

    // Allocate temporary memory
    cudaMalloc(&tempStorage, tempStorageSize);

    // Sum priorities
    cub::DeviceReduce::Sum(tempStorage, tempStorageSize, elements, deviceElementsSum,
                           numberOfElements);

    // Read priorities sum from device memory
    T elementsSum;
    cudaMemcpy(&elementsSum, deviceElementsSum, sizeof(T), cudaMemcpyDeviceToHost);

    // Free temporary memory
    cudaFree(tempStorage);
    cudaFree(deviceElementsSum);

    return elementsSum;
}

int DeviceSum(unsigned int numberOfElements, int* elements)
{
    return DeviceSum<int>(numberOfElements, elements);
}

float DeviceSum(unsigned int numberOfElements, float* elements)
{
    return DeviceSum<float>(numberOfElements, elements);
}
}
