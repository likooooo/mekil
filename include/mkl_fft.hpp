#pragma once
#include "mkl_basic_operator.h"
#include "mkl_wrapper.hpp"
#include <assert.h>
#if defined(HAVE_FFTW) || defined(HAVE_FFTWF)
#   include "fftw_fft.hpp"
#endif
//== 
// TODO : 对于 mkl::fft 不支持超过 1d inplace real->complex.
// 实际上我也并没有找到相关的资料, 我目前的猜测是 mkl 不支持, 目前的解决方案是通过引入 fftw 实现.
// mkl 相比于 fftw 后者更灵活, 在没有解决 TODO 的情况下, 我建议直接用 fftw
// 

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

    inline int print_dft_descriptor(DFTI_DESCRIPTOR_HANDLE handle)
    {
        if (!handle) {
            std::cerr << "DFTI descriptor is null" << std::endl;
            return 1;
        }

        auto get_str = [](MKL_LONG v, const std::string& type) -> std::string {
            if (type == "precision") {
                if (v == DFTI_SINGLE) return "single";
                if (v == DFTI_DOUBLE) return "double";
            } else if (type == "domain") {
                if (v == DFTI_REAL) return "real";
                if (v == DFTI_COMPLEX) return "complex";
            }
            return std::to_string(v);
        };

        MKL_LONG precision = 0, domain = 0, dim = 0, num_trans = 0;
        double fscale = 0.0, bscale = 0.0;

        DftiGetValue(handle, DFTI_PRECISION, &precision);
        DftiGetValue(handle, DFTI_FORWARD_DOMAIN, &domain);
        DftiGetValue(handle, DFTI_DIMENSION, &dim);
        DftiGetValue(handle, DFTI_NUMBER_OF_TRANSFORMS, &num_trans);
        DftiGetValue(handle, DFTI_FORWARD_SCALE, &fscale);
        DftiGetValue(handle, DFTI_BACKWARD_SCALE, &bscale);

        std::cout << "==== MKL DFTI Descriptor Info ====" << std::endl;
        std::cout << "Precision          : " << get_str(precision, "precision") << std::endl;
        std::cout << "Domain             : " << get_str(domain, "domain") << std::endl;
        std::cout << "Dimension          : " << dim << std::endl;

        if (dim > 0) {
            std::vector<MKL_LONG> sizes(dim);
            DftiGetValue(handle, DFTI_LENGTHS, sizes.data());
            std::cout << "Lengths            : [";
            for (int i = 0; i < dim; i++) {
                std::cout << sizes[i] << (i + 1 < dim ? ", " : "");
            }
            std::cout << "]" << std::endl;
        }

        std::cout << "Batch (Transforms) : " << num_trans << std::endl;
        std::cout << "Forward scale      : " << fscale << std::endl;
        std::cout << "Backward scale     : " << bscale << std::endl;

        // 额外参数
        MKL_LONG storage = 0;
        if (DftiGetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, &storage) == 0) {
            std::cout << "Conjugate-even storage : "
                    << (storage == DFTI_COMPLEX_COMPLEX ? "complex-complex" :
                        storage == DFTI_COMPLEX_REAL    ? "complex-real" :
                                                            std::to_string(storage))
                    << std::endl;
        }

        MKL_LONG placement = 0;
        if (DftiGetValue(handle, DFTI_PLACEMENT, &placement) == 0) {
            std::cout << "Placement              : "
                    << (placement == DFTI_INPLACE ? "inplace" :
                        placement == DFTI_NOT_INPLACE ? "not-inplace" :
                                                        std::to_string(placement))
                    << std::endl;
        }

        std::cout << "==================================" << std::endl;
        return 0;
    }
    template<class T> struct mklFFT
    {
        constexpr static DFTI_CONFIG_VALUE dft_precision = std::array<DFTI_CONFIG_VALUE, 2>{DFTI_SINGLE, DFTI_DOUBLE}.at(8 ==sizeof(real_t<T>));
        constexpr static DFTI_CONFIG_VALUE domain    = std::array<DFTI_CONFIG_VALUE, 2>{DFTI_REAL, DFTI_COMPLEX}.at(is_complex_v<T>);
        using spatial_type = typename fft_io_type<T>::spatial_type;
        using fourier_type = typename fft_io_type<T>::fourier_type;

        static void exec_forward(DFTI_DESCRIPTOR_HANDLE handle, void* in, void* out=nullptr)
        {
            // if(out == nullptr) out = in;
            // //== transpose for col-major
            // enum DFTI_CONFIG_VALUE precision;
            // MKL_CALL(DftiGetValue(handle, DFTI_PLACEMENT, &precision));
            // if(DFTI_INPLACE == precision && is_real_v<T> && out == in)
            // {
            //     MKL_LONG retrieved_lengths[5] = {0}; 
            //     MKL_CALL(DftiGetValue(handle, DFTI_LENGTHS, retrieved_lengths));
            //     //== support 2d only
            //     assert(0 != retrieved_lengths[0] && 0 == retrieved_lengths[2]);
            //     if(retrieved_lengths[1] != 0){
            //         transpose<T>((T*)in, (T*)out, {int(retrieved_lengths[1]), int((retrieved_lengths[0] / 2 + 1) * 2)}); 
            //     }
            // }
            if(nullptr != out){
                MKL_CALL(DftiComputeForward(handle, (spatial_type*)in, (fourier_type*)out));
            }
            else{
                MKL_LONG placement = 0;
                MKL_CALL(DftiGetValue(handle, DFTI_PLACEMENT, &placement));
                if(placement != DFTI_INPLACE) print_dft_descriptor(handle);
                assert(placement == DFTI_INPLACE);
                MKL_CALL(DftiComputeForward(handle, (spatial_type*)in));
            }
        }
        static void exec_backward(DFTI_DESCRIPTOR_HANDLE handle, void* in, void* out=nullptr)
        {
            // if(out == nullptr) out = in;
            if(nullptr != out){
                MKL_CALL(DftiComputeBackward(handle, (fourier_type*)in,  (spatial_type*)out));
            }
            else{
                MKL_LONG placement = 0;
                MKL_CALL(DftiGetValue(handle, DFTI_PLACEMENT, &placement));
                if(placement != DFTI_INPLACE) print_dft_descriptor(handle);
                assert(placement == DFTI_INPLACE);
                MKL_CALL(DftiComputeBackward(handle, (fourier_type*)in));
            }

            // //== transpose for col-major
            // enum DFTI_CONFIG_VALUE precision;
            // MKL_CALL(DftiGetValue(handle, DFTI_PLACEMENT, &precision));
            // if(DFTI_INPLACE == precision && is_real_v<T> && out == in)
            // {
            //     MKL_LONG retrieved_lengths[5] = {0}; 
            //     MKL_CALL(DftiGetValue(handle, DFTI_LENGTHS, retrieved_lengths));
            //     //== support 2d only
            //     assert(0 != retrieved_lengths[0] && 0 == retrieved_lengths[2]);
            //     if(retrieved_lengths[1] != 0){
            //         transpose<T>((T*)in, (T*)out, {int((retrieved_lengths[0] / 2 + 1) * 2), int(retrieved_lengths[1])}); 
            //     }
            // }
        }
        using pPlan_t = std::unique_ptr<DFTI_DESCRIPTOR_HANDLE, mkl_fft_plan_deleter>;
        static pPlan_t make_row_major_plan(const std::vector<MKL_LONG>& row_major_dims, bool inplace, real_t<T> normalize_factor, int batch_size)
        {
            pPlan_t pPlan(new DFTI_DESCRIPTOR_HANDLE, mkl_fft_plan_deleter());
            enum DFTI_CONFIG_VALUE test[2] ={dft_precision, domain};
            if(row_major_dims.size() == 1){
                MKL_CALL(DftiCreateDescriptor(pPlan.get(), dft_precision, domain, 1, row_major_dims.front()));
            }
            else{
                MKL_CALL(DftiCreateDescriptor(pPlan.get(), dft_precision, domain, row_major_dims.size(), row_major_dims.data()));
            }
            if(batch_size > 1){
                MKL_CALL(DftiSetValue(*pPlan, DFTI_NUMBER_OF_TRANSFORMS, batch_size));
            }
            if(inplace && DFTI_COMPLEX == domain){
                MKL_CALL(DftiSetValue(*pPlan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX));
            }
            else if(inplace && DFTI_REAL == domain && row_major_dims.size() > 1){
                inplace = false;
                // MKL_CALL(DftiSetValue(*pPlan, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_REAL_COMPLEX));
            }
            if(!inplace){
                MKL_CALL(DftiSetValue(*pPlan, DFTI_PLACEMENT, DFTI_NOT_INPLACE));
            }
            else{
                MKL_CALL(DftiSetValue(*pPlan, DFTI_PLACEMENT, DFTI_INPLACE));
            }
            //== 0 : normalized like matlab; 
            //   1 : disable
            // else: user defined
            if(0 == normalize_factor){
                normalize_factor = 1.0;
                for(MKL_LONG n : row_major_dims) normalize_factor *= n;
                normalize_factor = 1/normalize_factor;
            }
            MKL_CALL(DftiSetValue(*pPlan, DFTI_BACKWARD_SCALE, normalize_factor));
            MKL_CALL(DftiCommitDescriptor(*pPlan));
            return pPlan;
        }
        static pPlan_t make_plan(std::vector<MKL_LONG> col_major_dims,  bool inplace = false, real_t<T> normalize_factor = 0, int batch_size=1)
        {
            if(col_major_dims.back() <= 1) col_major_dims.pop_back();
            std::reverse(col_major_dims.begin(), col_major_dims.end());
            return make_row_major_plan(col_major_dims, inplace, normalize_factor, batch_size);
        }
    };
}