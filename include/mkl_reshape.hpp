#pragma once
#include "mkl_basic_operator.h"


template<class T> inline void copy_batch_strided(const MKL_INT N,
                               const T *X, const MKL_INT incX, const MKL_INT stridex,
                               T *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size)
{
    CBLAS_REPEAT_CODE(T, copy_batch_strided, N, X, incX, stridex, Y, incY, stridey, batch_size);
}

template<class T> inline void crop_image(T* output, vec2<size_t> output_shape, vec2<size_t> output_offset, 
                 const T* input,  vec2<size_t> input_shape,  vec2<size_t> input_offset, int step_in = 1, int step_out = 1)
{
    const auto [inputSizeX, inputSizeY] = input_shape;
    const auto [outputSizeX, outputSizeY] = output_shape;
    const auto [offset_x_in, offset_y_in] = input_offset;
    const auto [offset_x_out, offset_y_out] = output_offset;
    const T* pIn = input + offset_y_in * inputSizeX + offset_x_in;
    T* pOut = output +  offset_y_out * outputSizeX + offset_x_out;
    copy_batch_strided<T>(std::min(inputSizeX - offset_x_in, outputSizeX - offset_x_out), pIn, step_in, inputSizeX, pOut, step_out, outputSizeX, std::min(inputSizeY - offset_y_in, outputSizeY - offset_y_out));
}
template<class TFrom, class TTo> inline void crop_to(TTo* output, vec2<size_t> output_shape, vec2<size_t> output_offset, 
                 const TFrom* input,  vec2<size_t> input_shape,  vec2<size_t> input_offset)
{
    static_assert(std::is_standard_layout_v<TFrom> && std::is_standard_layout_v<TTo>);
    constexpr int min_pixel_size = std::min(sizeof(TFrom), sizeof(TTo));

    constexpr int element_count_from = sizeof(TFrom) / min_pixel_size;
    constexpr int element_count_to = sizeof(TTo) / min_pixel_size;
    static_assert(element_count_from * min_pixel_size == sizeof(TFrom));
    static_assert(element_count_to   * min_pixel_size == sizeof(TTo));
    static_assert(element_count_from == 1 || element_count_to == 1);

    output_shape[0] *= element_count_to;
    output_offset[0] *= element_count_to;
    input_shape[0] *= element_count_from;
    input_offset[0] *= element_count_from;
    auto crop = [&](auto t){
        using T = decltype(t);
        crop_image<T>(reinterpret_cast<T*>(output), output_shape, output_offset,
                     reinterpret_cast<const T*>(input), input_shape, input_offset, element_count_from, element_count_to
        );
    };

    if constexpr(sizeof(float) == min_pixel_size){
        crop(float());
    }
    else if constexpr(sizeof(double) == min_pixel_size){
        crop(double());
    }
    else if constexpr(sizeof(std::complex<double>) == min_pixel_size){
        crop(std::complex<double>());
    }
    else{
        unreachable_constexpr_if();
    }
}



template <class T> inline void fftshift_even_only(T *image, size_t width, size_t height)
{
    const size_t sizeX = width;
    const size_t sizeY = height;
    
    T *pA = image;
    T *pB = image + sizeX/2;
    T *pC = image + sizeY/2 * sizeX;
    T *pD = image + sizeY/2 * sizeX + sizeX/2;

    const size_t halfSizeX = (sizeX + 1) / 2;
    const size_t halfSizeY = (sizeY + 1) / 2;
    for (size_t i = 0; i < halfSizeY; i++)
    {
        CBLAS_REPEAT_CODE(T, swap, halfSizeX, pA, 1, pD, 1);
        CBLAS_REPEAT_CODE(T, swap, halfSizeX, pB, 1, pC, 1);
        pA += sizeX;
        pD += sizeX;
        pB += sizeX;
        pC += sizeX;
    }
}
template <class T> inline void fftshift(T *image, size_t width, size_t height)
{
    if((0 == width %2) && ( 0 == height %2)){
        fftshift_even_only(image, width, height);
        return;
    }
    const size_t sizeX = width;
    const size_t sizeY = height;
    
    const size_t halfSizeX = (sizeX + 1) / 2;
    const size_t halfSizeY = (sizeY + 1) / 2;
    T *pA = image;
    T *pB = image + halfSizeX;
    T *pC = image + (sizeY - halfSizeY) * sizeX;
    T *pD = image + halfSizeY * sizeX + halfSizeX;

    std::vector<T> temp(halfSizeX * sizeY);
    copy_batch_strided(halfSizeX, pA, 1, sizeX, temp.data(), 1, halfSizeX, sizeY);
    
    copy_batch_strided(sizeX - halfSizeX, pB, 1, sizeX, pC, 1, sizeX, halfSizeY);
    copy_batch_strided(sizeX - halfSizeX, pD, 1, sizeX, pA, 1, sizeX, sizeY - halfSizeY);

    pB = image + (sizeX - halfSizeX);
    pD = image + (sizeY - halfSizeY) * sizeX + (sizeX - halfSizeX);
    copy_batch_strided(halfSizeX, temp.data(), 1, halfSizeX, pD, 1, sizeX, halfSizeY);
    copy_batch_strided(halfSizeX, temp.data() + halfSizeX * halfSizeY, 1, halfSizeX, pB, 1, sizeX, sizeY - halfSizeY);
}
template<class T, bool is_c_stly_memory_layout = false>
inline void transpose(const T* input, T* output, const std::array<int,2>& shape)
{
    const int rows = shape[0];
    const int cols = shape[1];
    MKL_REPEAT_CODE(T, omatcopy,
        is_c_stly_memory_layout ? 'R' : 'C', // memory layout
        'T',                                 // transpose
        rows, cols,
        mkl_t<T>{1.0},                       // alpha
        (const mkl_t<T>*)input, is_c_stly_memory_layout ? cols : rows,
        (mkl_t<T>*)output, is_c_stly_memory_layout ? rows : cols
    );
}
template<class TFrom, class TTo, class Callback, bool is_c_stly_memory_layout = false>
inline void permuteND(const TFrom* input, TTo* output,
               const std::vector<int>& shape,
               const std::vector<int>& perm,
               Callback convert_callback)
{
    int ndim = static_cast<int>(shape.size());

    // 计算输入 stride
    std::vector<int> in_stride(ndim);
    if constexpr (is_c_stly_memory_layout) {
        // C-style: last dim stride=1
        in_stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            in_stride[i] = in_stride[i + 1] * shape[i + 1];
        }
    } else {
        // Fortran-style: first dim stride=1
        in_stride[0] = 1;
        for (int i = 1; i < ndim; ++i) {
            in_stride[i] = in_stride[i - 1] * shape[i - 1];
        }
    }

    // 输出 shape
    std::vector<int> out_shape(ndim);
    for (int i = 0; i < ndim; ++i)
        out_shape[i] = shape[perm[i]];

    // 输出 stride (和输入相同方式)
    std::vector<int> out_stride(ndim);
    if constexpr (is_c_stly_memory_layout) {
        out_stride[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; --i) {
            out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
        }
    } else {
        out_stride[0] = 1;
        for (int i = 1; i < ndim; ++i) {
            out_stride[i] = out_stride[i - 1] * out_shape[i - 1];
        }
    }

    // 总元素数
    int total = 1;
    for (int d : shape) total *= d;

    // 遍历输出 index
    for (int out_index = 0; out_index < total; ++out_index) {
        // 解码 output 坐标
        std::vector<int> out_coord(ndim);
        int idx = out_index;
        if constexpr (is_c_stly_memory_layout) {
            for (int i = 0; i < ndim; ++i) {
                out_coord[i] = idx / out_stride[i];
                idx %= out_stride[i];
            }
        } else {
            for (int i = ndim - 1; i >= 0; --i) {
                out_coord[i] = idx / out_stride[i];
                idx %= out_stride[i];
            }
        }

        // 反 perm 得到输入坐标
        std::vector<int> in_coord(ndim);
        for (int i = 0; i < ndim; ++i)
            in_coord[perm[i]] = out_coord[i];

        // 输入 index
        int in_index = 0;
        for (int i = 0; i < ndim; ++i)
            in_index += in_coord[i] * in_stride[i];

        // 写入输出
        output[out_index] = convert_callback(input[in_index]);
    }
}

