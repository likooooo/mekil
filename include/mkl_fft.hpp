#pragma once
#include <mkl.h>
#include "mkl_basic_operator.h"
#include <type_traist_notebook/type_traist.hpp>
#include <mkl_dfti.h>

namespace mekil
{
    template<class TSpatial>
    struct fft_io_type{
        using spatial_type = mkl_t<TSpatial>;
        using fourier_type =  mkl_t<complex_t<TSpatial>>;
    };
    struct mkl_fft_plan_deleter {
        void operator()(DFTI_DESCRIPTOR_HANDLE* handle) const {
            if(handle && *handle){
                DftiFreeDescriptor(handle);
            }
            delete handle;
        }
    };

    template<class T> struct mklFFT
    {
        constexpr static DFTI_CONFIG_VALUE dft_precision = std::array<DFTI_CONFIG_VALUE, 2>{DFTI_SINGLE, DFTI_DOUBLE}.at(8 ==sizeof(real_t<T>));
        constexpr static DFTI_CONFIG_VALUE domain    = std::array<DFTI_CONFIG_VALUE, 2>{DFTI_REAL, DFTI_COMPLEX}.at(is_complex_v<T>);
        using spatial_type = typename fft_io_type<T>::spatial_type;
        using fourier_type = typename fft_io_type<T>::fourier_type;

        static void exec_forward(DFTI_DESCRIPTOR_HANDLE handle, void* in, void* out=nullptr){
            if(out == nullptr) out = in;
            MKL_CALL(DftiComputeForward(handle, (spatial_type*)in, (fourier_type*)out));
        }
        static void exec_backward(DFTI_DESCRIPTOR_HANDLE handle, void* in, void* out=nullptr){
            if(out == nullptr) out = in;
            MKL_CALL(DftiComputeBackward(handle, (fourier_type*)in, (spatial_type*)out));
        }
        using pPlan_t = std::unique_ptr<DFTI_DESCRIPTOR_HANDLE, mkl_fft_plan_deleter>;
        static pPlan_t make_row_major_plan(const std::vector<MKL_LONG>& row_major_dims, int batch_size=1)
        {
            pPlan_t pPlan(new DFTI_DESCRIPTOR_HANDLE, mkl_fft_plan_deleter());
            if(row_major_dims.size() == 1){
                MKL_CALL(DftiCreateDescriptor(pPlan.get(), dft_precision, domain, 1, row_major_dims.front()));
            }
            else{
                MKL_CALL(DftiCreateDescriptor(pPlan.get(), dft_precision, domain, row_major_dims.size(), row_major_dims.data()));
            }
            if(batch_size > 1){
                MKL_CALL(DftiSetValue(*pPlan, DFTI_NUMBER_OF_TRANSFORMS, batch_size));
            }
            MKL_CALL(DftiCommitDescriptor(*pPlan));
            return pPlan;
        }

        static pPlan_t make_plan(std::vector<MKL_LONG> col_major_dims, int batch_size=1)
        {
            if(col_major_dims.back() <= 1) col_major_dims.pop_back();
            std::reverse(col_major_dims.begin(), col_major_dims.end());
            return make_row_major_plan(col_major_dims, batch_size);
        }
    };
   
}