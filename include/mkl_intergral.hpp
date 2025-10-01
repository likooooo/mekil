#pragma once
#include "mkl_basic_operator.h"
#include "mkl_vec.hpp"
namespace mkl
{
    template <typename T> void integral_y(vec2<size_t> shape, T* image)
    {
        using Tmkl = mkl_t<T>;
        const auto [ysize, xsize] = shape;
        T* a = image;
        T* b = image + xsize;
        for(size_t y = 0; y < ysize-1; y++, a+=xsize, b +=xsize)
            mkl::vec::self_add(xsize, a, b);
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
}