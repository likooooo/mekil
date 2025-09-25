#pragma once
#include <mkl.h>
#include <complex>
#include <type_traist_notebook/type_traist.hpp>

enum class matrix_major
{
    row_major,
    col_major
};
#define CBLAS_REPEAT_CODE(T, func, ...) \
    if      constexpr(is_s<T>){cblas_s##func(__VA_ARGS__);} \
    else if constexpr(is_d<T>){cblas_d##func(__VA_ARGS__);} \
    else if constexpr(is_c<T>){cblas_c##func(__VA_ARGS__);} \
    else if constexpr(is_z<T>){cblas_z##func(__VA_ARGS__);} \
    else{unreachable_constexpr_if();                   \
}
    
template<class T> inline void copy_batch_strided(const MKL_INT N,
                               const T *X, const MKL_INT incX, const MKL_INT stridex,
                               T *Y, const MKL_INT incY, const MKL_INT stridey,
                               const MKL_INT batch_size)
{
    CBLAS_REPEAT_CODE(T, copy_batch_strided, N, X, incX, stridex, Y, incY, stridey, batch_size);
    
}
template <class T>
inline void CenterCornerFlip(T *image, int widht, int height)
{
    const int sizeX = widht;
    const int sizeY = height;
    const int halfSizeX = sizeX / 2;
    const int halfSizeY = sizeY / 2;
    
    T *pA = image;
    T *pB = image + halfSizeX + sizeX % 2;
    T *pC = image + (halfSizeY  + sizeY % 2) * sizeX;
    T *pD = image + (halfSizeY  + sizeY % 2) * sizeX + halfSizeX + sizeX % 2;

    for (int i = 0; i < halfSizeY; i++)
    {
        if constexpr (is_s<T>)
        {
            cblas_sswap(halfSizeX, pA, 1, pD, 1);
            cblas_sswap(halfSizeX, pB, 1, pC, 1);
        }
        else if constexpr (is_c<T>)
        {
            cblas_cswap(halfSizeX, pA, 1, pD, 1);
            cblas_cswap(halfSizeX, pB, 1, pC, 1);
        }
        else if constexpr (is_d<T>)
        {
            cblas_dswap(halfSizeX, pA, 1, pD, 1);
            cblas_dswap(halfSizeX, pB, 1, pC, 1);
        }
        else if constexpr (is_z<T>)
        {
            cblas_zswap(halfSizeX, pA, 1, pD, 1);
            cblas_zswap(halfSizeX, pB, 1, pC, 1);
        }
        pA += sizeX;
        pD += sizeX;
        pB += sizeX;
        pC += sizeX;
    }
}

template<class T> [[deprecated("use crop_image instead")]] inline void CropImage(T* pOut, const int ostride, const T* pIn, const int istride, const int sizex, const int sizey)
{
    //== 如果 crop image 包含最后一个元素，存在pOut越界的情况。(sizey 包含 pOut的最后一行)
    // 可以从小的image 拷贝到大的， 反过来的话是不安全的
    copy_batch_strided(sizex, pIn, 1, istride, pOut, 1, ostride, sizey);
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

template<class T> inline void VecDiv(int n, T*a, T*b, T* y)
{
    if constexpr(is_s<T>)
    {
        vsDiv(n, a, b, y);
    }
    else if constexpr(is_d<T>)
    {
        vdDiv(n, a, b, y);
    }
    else if constexpr(is_c<T>)
    {
        vcDiv(n, (MKL_Complex8*)a, (MKL_Complex8*)b, (MKL_Complex8*)y);
    }
    else if constexpr(is_z<T>)
    {
        vzDiv(n, (MKL_Complex16*)a, (MKL_Complex16*)b, (MKL_Complex16*)y);
    }
}
template<class T> inline void VecMul(int n, T*a, T*b, T* y)
{
    if constexpr(is_s<T>)
    {
        vsMul(n, a, b, y);
    }
    else if constexpr(is_d<T>)
    {
        vdMul(n, a, b, y);
    }
    else if constexpr(is_c<T>)
    {
        vcMul(n, (MKL_Complex8*)a, (MKL_Complex8*)b, (MKL_Complex8*)y);
    }
    else if constexpr(is_z<T>)
    {
        vzMul(n, (MKL_Complex16*)a, (MKL_Complex16*)b, (MKL_Complex16*)y);
    }
}
template<class T> inline void VecScala(int n, const T a, T* x, int inc = 1)
{
    if constexpr(is_s<T>)
    {
        cblas_sscal(n, a, x, inc);
    }
    else if constexpr(is_d<T>)
    {
        cblas_dscal(n, a, x, inc);
    }
    else if constexpr(is_c<T>)
    {
        cblas_cscal(n, (MKL_Complex8*)&a, (MKL_Complex8*)x, inc);
    }
    else if constexpr(is_z<T>)
    {
        cblas_zscal(n, (MKL_Complex16*)&a, (MKL_Complex16*)x, inc);
    }
}

#ifdef UNFINISHED_CODE
char major = 'r';
template <class T>
void transpose(T *p, int sizey, int sizex)
{
    if constexpr (std::is_same_v<float, std::remove_cv_t<T>>)
    {
        // mkl_simatcopy(major, 'T', sizex, sizey, T(1), p, sizey, sizex);
        mkl_simatcopy(major, 'T', sizey, sizex, T(1), p, sizex, sizey);
    }
    else if constexpr (std::is_same_v<double, std::remove_cv_t<T>>)
    {
        mkl_dimatcopy(major, 'T', sizey, sizex, T(1), p, sizex, sizey);
    }
    else if constexpr (std::is_same_v<std::complex<float>, std::remove_cv_t<T>>)
    {
        mkl_cimatcopy(major, 'T', sizey, sizex, MKL_Complex8(1), (MKL_Complex8 *)p, sizex, sizey);
    }
    else if constexpr (std::is_same_v<std::complex<double>, std::remove_cv_t<T>>)
    {
        mkl_zimatcopy(major, 'T', sizey, sizex, MKL_Complex16(1), (MKL_Complex16 *)p, sizex, sizey);
    }
}
#endif