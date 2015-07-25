#pragma once

namespace BVHRT
{

/// <summary> Sort the lists of keys and values using CUB's radix sort implementation
///           (http://nvlabs.github.io/cub/). Does not preserve input arrays. </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numberOfElements"> Number of elements. </param>
/// <param name="keysIn">           [in,out] Input keys. </param>
/// <param name="keysOut">          [in,out] Output keys. </param>
/// <param name="valuesIn">         [in,out] Input values. </param>
/// <param name="valuesOut">        [in,out] Output values. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceSort(unsigned int numberOfElements, unsigned int** keysIn, unsigned int** keysOut,
                 unsigned int** valuesIn, unsigned int** valuesOut);

/// <summary> Sort the lists of keys and values using CUB's radix sort implementation
///           (http://nvlabs.github.io/cub/). Does not preserve input arrays. </summary>
///
/// <remarks> Leonardo, 02/11/2015. </remarks>
///
/// <param name="numberOfElements"> Number of elements. </param>
/// <param name="keysIn">           [in,out] Input keys. </param>
/// <param name="keysOut">          [in,out] Output keys. </param>
/// <param name="valuesIn">         [in,out] Input values. </param>
/// <param name="valuesOut">        [in,out] Output values. </param>
///
/// <returns> Execution time, in milliseconds. </returns>
float DeviceSort(unsigned int numberOfElements, unsigned long long int** keysIn, 
        unsigned long long int** keysOut, unsigned int** valuesIn, unsigned int** valuesOut);

/// <summary> Wrapper for cub::DeviceReduce::Sum(). If a temporary memory array is not specified, 
///           calculates the required temporary memory size and returns.</summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numberOfTriangles"> Number of triangles. </param>
/// <param name="in">                Input values. </param>
/// <param name="out">               [out] Output value. </param>
/// <param name="tempMemorySize">    [in,out] If non-null, the size. </param>
/// <param name="tempMemory">        [in,out] (Optional) If non-null, the temporary memory array. 
///                                  </param>
void DeviceSum(unsigned int numberOfTriangles, int* in, int* out, size_t* size,
               void* tempMemory = nullptr);

/// <summary> Sum the elements from the specified array. Wrapper for cub::DeviceReduce::Sum().
///           </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numberOfElements"> Number of elements in the array. </param>
/// <param name="elements">         Elements to be summed. </param>
///
/// <returns> The sum of the input values. </returns>
int DeviceSum(unsigned int numberOfElements, int* elements);

/// <summary> Sum the elements from the specified array. Wrapper for cub::DeviceReduce::Sum().
///           </summary>
///
/// <remarks> Leonardo, 12/29/2014. </remarks>
///
/// <param name="numberOfElements"> Number of elements in the array. </param>
/// <param name="elements">         Elements to be summed. </param>
///
/// <returns> The sum of the input values. </returns>
float DeviceSum(unsigned int numberOfElements, float* elements);

}