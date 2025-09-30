#pragma once
#include <fftw3.h>

namespace mkl
{
    template<class T>
    struct fftw
    {
        using rT = real_t<T>;
        using cT = complex_t<T>;
        using plan_ptr_type = std::conditional_t<is_s<rT>, fftwf_plan, fftw_plan>;
        struct fftw_plan_deleter {
            void operator()(plan_ptr_type p) const {
                if constexpr(is_s<rT>)
                    fftwf_destroy_plan(p);
                else if constexpr(is_d<rT> )
                    fftw_destroy_plan(p);
                else
                    unreachable_constexpr_if();
            }
        };
        using plan_type = std::remove_pointer_t<plan_ptr_type>;
        using plan_holder = std::unique_ptr<plan_type, fftw_plan_deleter>;
        constexpr static int flag = FFTW_ESTIMATE;

        static plan_ptr_type plan_c2c(const std::vector<int>& dim, 
            int direction = FFTW_FORWARD, T* pFrom = nullptr, void* pTo = nullptr)
        {
            const int rank = dim.size();
            if constexpr(is_s<rT>){
                return fftwf_plan_dft(
                    rank, dim.data(),
                    reinterpret_cast<fftwf_complex*>(pFrom),
                    reinterpret_cast<fftwf_complex*>(pTo), 
                    direction, flag
                );
            }
            else if constexpr(is_d<rT>){
                return fftw_plan_dft(
                    rank, dim.data(),
                    reinterpret_cast<fftw_complex*>(pFrom),
                    reinterpret_cast<fftw_complex*>(pTo), 
                    direction, flag
                );
            }
            else{
                unreachable_constexpr_if();
            }
        }
        static plan_ptr_type plan_c2r(const std::vector<int>& dim, 
            T* pFrom = nullptr, void* pTo = nullptr)
        {
            const int rank = dim.size();
            if constexpr(is_s<rT>){
                return fftwf_plan_dft_c2r(
                    rank, dim.data(),
                    reinterpret_cast<fftwf_complex*>(pFrom),
                    pTo, flag
                );
            }
            else if constexpr(is_d<rT>){
                return fftw_plan_dft_c2r(
                    rank, dim.data(),
                    reinterpret_cast<fftw_complex*>(pFrom),
                    pTo, flag
                );
            }
            else{
                unreachable_constexpr_if();
            }
        }
        static plan_ptr_type plan_r2c(const std::vector<int>& dim, 
            T* pFrom = nullptr, void* pTo = nullptr)
        {
            const int rank = dim.size();
            if constexpr(is_s<rT>){
                return fftwf_plan_dft_r2c(
                    rank, dim.data(), pFrom, 
                    reinterpret_cast<fftwf_complex*>(pTo), flag
                );
            }
            else if constexpr(is_d<rT>){
                return fftw_plan_dft_r2c(
                    rank, dim.data(), pFrom, 
                    reinterpret_cast<fftw_complex*>(pTo), flag
                );
            }
            else{
                unreachable_constexpr_if();
            }
        }
        template<class TTo> static plan_holder make_plan(const std::vector<int>& dim, 
            int direction = FFTW_FORWARD, T* pFrom = nullptr, void* pTo = nullptr)
        {
            static_assert(!(is_real_v<T> && is_real_v<TTo>), "fft real to real is invalid.");
            plan_holder p;
            if constexpr(is_real_v<T>){
                assert(direction == FFTW_FORWARD);
                p = plan_holder(plan_r2c(dim, direction, pFrom, pTo));
            }
            else if constexpr(is_real_v<TTo>){
                assert(direction == FFTW_BACKWARD);
                p = plan_holder(plan_c2r(dim, pFrom, pTo));
            }
            else if constexpr(is_complex_v<TTo>){
                p = plan_holder(plan_c2c(dim, direction, pFrom, pTo));
            }
            else 
                unreachable_constexpr_if();
            return p;
        }
        static void transform(plan_ptr_type pPlan, T* pFrom = nullptr, void* pTo = nullptr)
        {
            if constexpr(is_s<rT>)
                fftwf_execute(pPlan.get(), pFrom, pTo);
            else if constexpr(is_d<rT>)
                fftw_execute(pPlan.get(), pFrom, pTo);
            else
                unreachable_constexpr_if();
        }
        static auto sprint_plan(const plan_ptr_type p)
        {
            struct deleter{void operator()(char *s){free(s);}};
            char* s;
            if constexpr(is_s<rT>)
                s = fftwf_sprint_plan(p);
            else if constexpr(is_d<rT>)
                s = fftw_sprint_plan(p);
            else
                unreachable_constexpr_if();
            return std::unique_ptr<char, deleter>(s, deleter());
        }
    };
};