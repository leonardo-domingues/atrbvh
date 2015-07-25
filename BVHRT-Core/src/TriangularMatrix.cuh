#pragma once

#include <math.h>
#include <vector_types.h>

// Get the number of elements which can be stored in a diagonal square matrix of size 'dim,
// excluding the main diagonal.
#define TRM_SIZE(dim) (((dim - 1) * (dim)) / 2)

// Get the index of an element in an array that represents an upper triangular matrix.
// The main diagonal elements are not included in the array.
#define UPPER_TRM_INDEX(row, column, dim) \
        (TRM_SIZE(dim) - TRM_SIZE((dim) - (row)) + (column) - (row) - 1)

// Get the row number an element in an upper triangular matrix represented as an array. 
// The main diagonal elements are not included in the array.
#define UPPER_TRM_ROW(index, dim) rowIndex((index), (dim))
__forceinline__ __host__ __device__ int rowIndex(int index, int dim)
{
    return dim - 2 - static_cast<int>(sqrtf(-8.0f * index + 4 * dim * (dim - 1) - 7) / 2.0 - 0.5);
}

// Get the column number an element in an upper triangular matrix represented as an array. 
// The main diagonal elements are not included in the array.
#define UPPER_TRM_COL(index, dim) columnIndex((index), (dim))
__forceinline__ __host__ __device__ int columnIndex(int index, int dim)
{
    int i = rowIndex(index, dim);
    return index + i + 1 - dim * (dim - 1) / 2 + (dim - i) * ((dim - i) - 1) / 2;
}

// Get the index of an element in an array that represents a lower triangular matrix.
// The main diagonal elements are not included in the array.
#define LOWER_TRM_INDEX(row, column) \
        (TRM_SIZE((row)) + (column))

// Get the row number an element in a lower triangular matrix represented as an array. 
// The main diagonal elements are not included in the array.
#define LOWER_TRM_ROW(index) rowIndex((index))
__forceinline__ __host__ __device__ int rowIndex(int index)
{
    return static_cast<int>((-1 + sqrtf(8.0f * index + 1)) / 2) + 1;
}

// Get the column number an element in a lower triangular matrix represented as an array. 
// The main diagonal elements are not included in the array.
#define LOWER_TRM_COL(index) columnIndex((index))
__forceinline__ __host__ __device__ int columnIndex(int index)
{
    int y = rowIndex(index);
    return (index) - (y * (y - 1)) / 2;
}
