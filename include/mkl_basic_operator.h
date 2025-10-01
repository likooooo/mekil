#pragma once
#include <mkl.h>
#include <type_traist_notebook/type_traist.hpp>
#include <assert.h>

#ifndef MKL_CALL
#   define MKL_CALL( call )                                                 \
    {                                                                       \
        MKL_LONG status =( call );                                          \
        if (status!=DFTI_NO_ERROR&&!DftiErrorClass(status, DFTI_NO_ERROR))  \
            fprintf( stderr,                                                \
                     "ERROR: MKL call \"%s\" in file %s:%d failed "         \
                     "\n* %s\n\n",                                          \
                     #call,                                                 \
                     __FILE__,                                              \
                     __LINE__,                                              \
                     DftiErrorMessage(status) );                            \
        assert(status == DFTI_NO_ERROR);                                    \
    }
#endif  // CUFFT_CALL

#define REPEAT_CODE(T, s,d, c, z, ...) \
    if constexpr(is_s<T>){s(__VA_ARGS__);} \
    else if constexpr(is_d<T>){d(__VA_ARGS__);} \
    else if constexpr(is_c<T>){c(__VA_ARGS__);} \
    else if constexpr(is_z<T>){z(__VA_ARGS__);} \
    else{unreachable_constexpr_if();}

#define __REPEAT_CODE(routing, T, func, ...) \
    if      constexpr(is_s<T>){routing##s##func(__VA_ARGS__);} \
    else if constexpr(is_d<T>){routing##d##func(__VA_ARGS__);} \
    else if constexpr(is_c<T>){routing##c##func(__VA_ARGS__);} \
    else if constexpr(is_z<T>){routing##z##func(__VA_ARGS__);} \
    else     unreachable_constexpr_if();

#define CBLAS_REPEAT_CODE(T, func, ...) __REPEAT_CODE(cblas_, T, func, __VA_ARGS__)
#define MKL_REPEAT_CODE(T, func, ...)   __REPEAT_CODE(  mkl_, T, func, __VA_ARGS__)
#define VEC_REPEAT_CODE(T, func, ...)   __REPEAT_CODE(  v, T, func, __VA_ARGS__)

template<class T> struct mkl_mapping; 
template<>struct mkl_mapping<float> {using type = float;};
template<>struct mkl_mapping<double> {using type = double;};
template<>struct mkl_mapping<std::complex<float>> {using type = MKL_Complex8;};
template<>struct mkl_mapping<std::complex<double>> {using type = MKL_Complex16;};
template<class T> using mkl_t = typename mkl_mapping<T>::type;

