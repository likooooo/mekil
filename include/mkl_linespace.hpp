#pragma once
#include "mkl_basic_operator.h"
#include "mkl_reshape.hpp"

namespace mkl
{
    template<class T> void linespace(T*p, size_t num, real_t<T> start, real_t<T> step) 
    {
        if constexpr(is_real_v<T>){
            std::vector<T> idx(num);
            std::iota(idx.begin(), idx.end(), T(0));
            VEC_REPEAT_CODE(T, LinearFrac,static_cast<MKL_INT>(num),
                    idx.data(), idx.data(), 
                    T(step), T(start), T(0), T(1), p);
        }
        else{
            std::vector<real_t<T>> buf(num);
            linespace<real_t<T>>(buf.data(), num, start, step);
            for(auto n : buf) {
                *p = n; p++;
            }
            
        }
    }

    template<class T, size_t N=2> void linespace_nd(T* p/* p should be equal or larger than sum(num) */, 
        vec<size_t, N> num, vec<real_t<T>, N> start, vec<real_t<T>, N> step) 
    {
        for(size_t i = 0; i < N; i++){
            linespace(p, num[i], start[i], step[i]);
            p += num.at(i);
        }
    }
    template <size_t N> vec<size_t, N> prefix_product(const vec<size_t, N>& shape) {
        vec<size_t, N> pref;
        size_t cur = 1;
        for (size_t i = 0; i < N; ++i) {
            pref[i] = cur;
            cur *= shape[i];
        }
        return pref;
    }
    template<class TVec, size_t N=2> void meshgrid_nd(TVec* pVec/* p should be equal or larger than product(num) */, 
        vec<size_t, N> num, vec<real_t<TVec>, N> start, vec<real_t<TVec>, N> step) 
    {
        using T = real_t<TVec>;
        T* p = reinterpret_cast<T*>(pVec);
        if constexpr(N == 1){
            linespace(p, num.front(), start.front(), step.front());
            return;
        }
        else{
            std::vector<T> lines(sum(num));
            linespace_nd(lines.data(), num, start, step);

            std::array<const T*, N> axis_ptrs;
            {
                size_t axis_offset = 0;
                for (size_t d = 0; d < N; ++d) {
                    axis_ptrs[d] = lines.data() + axis_offset;
                    axis_offset += num[d];
                }
            }

            vec<size_t, N> idx{};
            size_t total = product(num);
            for (size_t linear = 0; linear < total; ++linear) {
                size_t t = linear;
                for (int d = 0; d < N; d++) {
                    idx[d] = t % num[d];
                    t /= num[d];
                }

                for (size_t d = 0; d < N; ++d) {
                    *p++ = axis_ptrs[d][idx[d]];
                }
            }
        }
    }
//     template<class T, size_t N = 2> void kspace(T* p, vec<size_t, N> num, vec<real_t<T>, N> start, vec<real_t<T>, N> step, real_t<T> n = 1)
//     {
//         vec<real_t<T>, N> pitch = step * num;
        
//         if constexpr(is_real_v<T>){
//             meshgrid_nd<T>(p, num, start, step);
//         }
//         else{
//             static_assert(sizeof(T) == sizeof(real_t<T>) * N);
//             using vT = typename T::value_type;
//             meshgrid_nd<vT>(reinterpret_cast<vT*>(p), num, start, step);
//         }
//     }
}