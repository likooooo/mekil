#pragma once
#include <fftw3.h>
#include <assert.h>

#define FFTW_REPEAT_CODE(TYPE, func, ...)                   \
    if constexpr(is_s<TYPE>)      fftwf_##func(__VA_ARGS__);\
    else if constexpr(is_d<TYPE>) fftw_##func(__VA_ARGS__); \
    else                        unreachable_constexpr_if();

namespace mekil
{
    template<class T> struct fftw_mapping{using type = T;};
    template<> struct fftw_mapping<complex_t<float>>{using type = fftwf_complex;};
    template<> struct fftw_mapping<complex_t<double>>{using type = fftw_complex;};
    template<class T> using fftw_t = typename fftw_mapping<T>::type;

    template<class T, class TTo>
    struct fftw
    {
        using rT = real_t<T>;
        using cT = complex_t<T>;
        using plan_ptr_type = std::conditional_t<is_s<rT>, fftwf_plan, fftw_plan>;
        struct fftw_plan_deleter {
            void operator()(plan_ptr_type p) const {
                FFTW_REPEAT_CODE(rT, destroy_plan, p);
            }
        };
        using plan_type = std::remove_pointer_t<plan_ptr_type>;
        using plan_holder = std::unique_ptr<plan_type, fftw_plan_deleter>;
        constexpr static int flag = FFTW_ESTIMATE;

        static plan_ptr_type plan_c2c(const std::vector<int>& dim, 
            int direction = FFTW_FORWARD, void* pFrom = nullptr, void* pTo = nullptr)
        {
            const int rank = dim.size();
            if constexpr(is_s<rT>){
                return fftwf_plan_dft(
                    rank, dim.data(),
                    reinterpret_cast<fftw_t<cT>*>(pFrom),
                    reinterpret_cast<fftw_t<cT>*>(pTo), 
                    direction, flag
                );
            }
            else if constexpr(is_d<rT>){
                return fftw_plan_dft(
                    rank, dim.data(),
                    reinterpret_cast<fftw_t<cT>*>(pFrom),
                    reinterpret_cast<fftw_t<cT>*>(pTo), 
                    direction, flag
                );
            }
            else{
                unreachable_constexpr_if();
            }
        }
        static plan_ptr_type plan_c2r(const std::vector<int>& dim, 
            void* pFrom = nullptr, void* pTo = nullptr)
        {
            const int rank = dim.size();
            if constexpr(is_s<rT>){
                return fftwf_plan_dft_c2r(
                    rank, dim.data(),
                    reinterpret_cast<fftw_t<cT>*>(pFrom),
                    reinterpret_cast<fftw_t<rT>*>(pTo), 
                    flag
                );
            }
            else if constexpr(is_d<rT>){
                return fftw_plan_dft_c2r(
                    rank, dim.data(),
                    reinterpret_cast<fftw_t<cT>*>(pFrom),
                    reinterpret_cast<fftw_t<rT>*>(pTo), 
                    flag
                );
            }
            else{
                unreachable_constexpr_if();
            }
        }
        static plan_ptr_type plan_r2c(const std::vector<int>& dim, 
            void* pFrom = nullptr, void* pTo = nullptr)
        {
            const int rank = dim.size();
            if constexpr(is_s<rT>){
                return fftwf_plan_dft_r2c(
                    rank, dim.data(), 
                    reinterpret_cast<fftw_t<rT>*>(pFrom),
                    reinterpret_cast<fftw_t<cT>*>(pTo), 
                    flag
                );
            }
            else if constexpr(is_d<rT>){
                return fftw_plan_dft_r2c(
                    rank, dim.data(),
                    reinterpret_cast<fftw_t<rT>*>(pFrom),
                    reinterpret_cast<fftw_t<cT>*>(pTo), 
                    flag
                );
            }
            else{
                unreachable_constexpr_if();
            }
        }
        static plan_holder make_plan(const std::vector<int>& dim, 
            int direction = FFTW_FORWARD, void* pFrom = nullptr, void* pTo = nullptr)
        {
            static_assert(!(is_real_v<T> && is_real_v<TTo>), "fft real to real is invalid.");
            plan_holder p;
            if constexpr(is_real_v<T>){
                assert(direction == FFTW_FORWARD);
                p = plan_holder(plan_r2c(dim, pFrom, pTo));
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
        static void transform(plan_ptr_type pPlan, void* pFrom = nullptr, void* pTo = nullptr)
        {
            if(nullptr == pFrom && nullptr == pTo){
                trace_print<void>_("    auto ", "", 0);
                FFTW_REPEAT_CODE(rT, execute, pPlan);
            }
            else{
                assert(nullptr != pFrom);
                if(pTo == nullptr) pTo = (void*)pFrom;
                if constexpr(is_real_v<T> && is_complex_v<TTo>){
                    trace_print<void>_("    r2c ", "", 0);
                    FFTW_REPEAT_CODE(rT, execute_dft_r2c, pPlan, (fftw_t<T>*)pFrom, (fftw_t<TTo>*)pTo);
                }
                else if constexpr(is_real_v<TTo> && is_complex_v<T>){
                    trace_print<void>_("    c2r ", "", 0);
                    FFTW_REPEAT_CODE(rT, execute_dft_c2r, pPlan, (fftw_t<T>*)pFrom, (fftw_t<TTo>*)pTo);
                }
                else if constexpr(is_complex_v<T> && is_complex_v<TTo>){
                    trace_print<void>_("    c2c ", "", 0);
                    FFTW_REPEAT_CODE(rT, execute_dft, pPlan, (fftw_t<T>*)pFrom, (fftw_t<TTo>*)pTo);
                }
                else 
                    unreachable_constexpr_if();
            }
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