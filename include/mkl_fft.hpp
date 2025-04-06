#pragma once
#include <mkl.h>
#include "mkl_basic_operator.h"
#include <type_traist_notebook/type_traist.hpp>

namespace mekil
{
    template<class TSpatial>
    struct fft_io_type{
        using spatial_type = mkl_t<TSpatial>;
        using fourier_type =  mkl_t<complex_t<TSpatial>>;
    };

    
    // struct mkl_plan_deleter {
    //     void operator()(DFTI_DESCRIPTOR_HANDLE* plan) const {
    //         DftiFreeDescriptor(plan);
    //         delete plan;
    //     }
    // };
    // template<class T> inline  std::unique_ptr<DFTI_DESCRIPTOR_HANDLE, mkl_plan_deleter> make_row_major_plan(const std::vector<int>& row_major_dims, cufftType toward /* cuFFT<T>::forward */, int batch_size = 1){
    //     std::unique_ptr<DFTI_DESCRIPTOR_HANDLE, mkl_plan_deleter> pPlan(new DFTI_DESCRIPTOR_HANDLE, mkl_plan_deleter());
    //     // MKL_CALL(cufftPlanMany(
    //     //     pPlan.get(), row_major_dims.size(), 
    //     //     const_cast<int*>(row_major_dims.data()), 
    //     //     nullptr, 1, 0, 
    //     //     nullptr, 1, 0, 
    //     //     toward, batch_size)
    //     // );
        
    //     if(1 == row_major_dims.size()){
    //         MKL_CALL(DftiCreateDescriptor(pPlan.get(), mekil::mklFFT<T>::dft_precision,
    //             DFTI_COMPLEX, 1, *dim));
    //     }
    //     else{
    //         MKL_CALL(DftiCreateDescriptor(pPlan.get(), mekil::mklFFT<T>::dft_precision,
    //             DFTI_COMPLEX, dim_size, dim));
    //     }
    //     return pPlan;
    // }
    // inline std::unique_ptr<DFTI_DESCRIPTOR_HANDLE, mkl_plan_deleter> make_plan(std::vector<int> col_maojr_dims, cufftType toward, int batch_size = 1){
    //     std::reverse(col_maojr_dims.begin(), col_maojr_dims.end());
    //     return make_row_major_plan(col_maojr_dims, toward, batch_size);
    // }

    template<class T> struct mklFFT
    {
        constexpr static DFTI_CONFIG_VALUE dft_precision = std::array<DFTI_CONFIG_VALUE, 2>{DFTI_SINGLE, DFTI_DOUBLE}.at(8 ==sizeof(real_t<T>));

        static void exec_forward(void* in, MKL_LONG* dim, size_t dim_size, void* out = nullptr)
        {
            DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
            MKL_CALL(DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
            MKL_CALL(DftiCommitDescriptor(my_desc_handle));
            MKL_CALL(DftiFreeDescriptor(&my_desc_handle));
        }
        static void exec_backward(void* in, void* out = nullptr)
        {
 
        }
    };
   
}