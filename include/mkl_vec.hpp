#pragma once
#include "mkl_basic_operator.h"

namespace mkl::vec
{
    template<class T> inline void add(int n, const T*a, const T*b, T* y)
    {
        VEC_REPEAT_CODE(T, Add, n, reinterpret_cast<const mkl_t<T>*>(a), reinterpret_cast<const mkl_t<T>*>(b), reinterpret_cast<mkl_t<T>*>(y));
    }
    template <typename T> void self_add(const int n, const T *x, T *y)
    {
        // y = y + x
        add(n, x, y, y);
    }
    template<class T> inline void sub(int n, const T*a, const T*b, T* y)
    {
        VEC_REPEAT_CODE(T, Sub, n, reinterpret_cast<const mkl_t<T>*>(a), reinterpret_cast<const mkl_t<T>*>(b), reinterpret_cast<mkl_t<T>*>(y));
    }
    template <typename T> void self_sub(const int n, const T *x, T *y)
    {
        sub(n, x, y, y);
    }
    template<class T> inline void mul(int n, const T*a, const T*b, T* y)
    {
        VEC_REPEAT_CODE(T, Mul, n, reinterpret_cast<const mkl_t<T>*>(a), reinterpret_cast<const mkl_t<T>*>(b), reinterpret_cast<mkl_t<T>*>(y));
    }
    template <typename T> void self_mul(const int n, const T *x, T *y)
    {
        mul(n, x, y, y);
    }
    template<class T> inline void div(int n, const T*a, const T*b, T* y)
    {
        VEC_REPEAT_CODE(T, Div, n, reinterpret_cast<const mkl_t<T>*>(a), reinterpret_cast<const mkl_t<T>*>(b), reinterpret_cast<mkl_t<T>*>(y));
    }
    template <typename T> void self_div(const int n, const T *x, T *y)
    {
        div(n, x, y, y);
    }
    template<class T> inline void add(int n, const T a, T* x)
    {
        //== TODO :SIMD
        std::array<mkl_t<T>, (32 * sizeof(std::complex<double>) / sizeof(T))> buf; 
        buf.fill(a);
        int ibegin = 0;
        while((ibegin + buf.size()) < n){
            mkl::vec::add(buf.size(), buf.data(), x);
            x += buf.size();
            ibegin += buf.size();
        }
        mkl::vec::add(n - ibegin, buf.data(), x);
    }
    template<class T> inline void sub(int n, const T a, T* x)
    {
        add(n, -a, x);
    }
    template<class T> inline void mul(int n, const T a, T* x, std::enable_if_t<is_real_v<T>, int> inc = 1)
    {
        CBLAS_REPEAT_CODE(T, scal, n, a, x, inc);
    }
    template<class T> inline void mul(int n, const T a, T* x, std::enable_if_t<is_complex_v<T>, int> inc = 1)
    {
        CBLAS_REPEAT_CODE(T, scal, n, &a, x, inc);
    }
    template<class T> inline void div(int n, const T a, T* x, int inc = 1)
    {
        CBLAS_REPEAT_CODE(T, scal, n, (T(1)/a), x, inc);
    }
};
