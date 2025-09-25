#pragma once
#include <mkl.h>
#include <type_traist_notebook/type_traist.hpp>

#ifndef MKL_CALL
#   define MKL_CALL( call )                                                 \
    {                                                                       \
        MKL_LONG status =( call );                                          \
        if ( status != DFTI_NO_ERROR )                                      \
            fprintf( stderr,                                                \
                     "ERROR: MKL call \"%s\" in line %d of file %s failed " \
                     "with "                                                \
                     "code (%d).\n",                                        \
                     #call,                                                 \
                     __LINE__,                                              \
                     __FILE__,                                              \
                     status );                                              \
    }
#endif  // CUFFT_CALL

#define REPEAT_CODE(T, s,d, c, z, ...) \
    if constexpr(is_s<T>){s(__VA_ARGS__);} \
    else if constexpr(is_d<T>){d(__VA_ARGS__);} \
    else if constexpr(is_c<T>){c(__VA_ARGS__);} \
    else if constexpr(is_z<T>){z(__VA_ARGS__);} \
    else{unreachable_constexpr_if();}
namespace mekil
{
    template<class T> struct mkl_mapping{using type = T;};
    template<> struct mkl_mapping<complex_t<float>>{using type = MKL_Complex8;};
    template<> struct mkl_mapping<complex_t<double>>{using type = MKL_Complex16;};
    template<class T> using mkl_t = typename mkl_mapping<T>::type;

    template <typename T> void VtAdd(const int n, const T *x, T *y)
    {
        using Tmkl = mkl_t<T>;
        REPEAT_CODE(T, vsAdd, vdAdd, vcAdd, vzAdd,
            n, (Tmkl*)x, (Tmkl*)y, (Tmkl*)y
        )
    }
    template <typename T> void integral_y(vec2<size_t> shape, T* image)
    {
        using Tmkl = mkl_t<T>;
        const auto [ysize, xsize] = shape;
        T* a = image;
        T* b = image + xsize;
        for(size_t y = 0; y < ysize-1; y++, a+=xsize, b +=xsize)
            VtAdd<T>(xsize, a, b);
    }
    template <typename T> void integral_x(vec2<size_t> shape, T* image)
    {
        using Tmkl = mkl_t<T>;
        const auto [ysize, xsize] = shape;
        auto line_op = [xsize](T* p){
            for(size_t x = 1; x < xsize; x++) p[x] += p[x - 1];
        };
        #pragma omp for
        for(size_t y = 0; y < ysize; y++) line_op(image + y * xsize);
    }
    template <typename T> void VtSub(const int n, const T *x, T *y);
    template <typename T> void VtMul(const int n, const T *x, T *y);
    template <typename T> void VtDiv(const int n, const T *x, T *y);
}