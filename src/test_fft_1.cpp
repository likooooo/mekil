
#include <mkl_fft.hpp>
#include <assert.h>
#include <py_helper.hpp>

template<class T> inline std::pair<int ,int> cal_layout(const std::vector<int>&  col_major_dim){
    auto prod = std::accumulate(col_major_dim.begin() + 1, col_major_dim.end(), (size_t)1, [](auto a, auto b) {return a * b; });
    auto change_fastest_axis = (std::is_floating_point_v<T> ? (col_major_dim.front() / 2 + 1) * 2 : col_major_dim.front());
    return {change_fastest_axis, prod};
}

template<class T>
void test_mkl_fft_check(const std::string& space, std::vector<int> col_major_dims)
{
    printf("*%s test mklFFT<%s>", space.c_str(), TypeReflection<T>().c_str());
    std::cout << col_major_dims << std::endl;
    using namespace mekil;

    using fft_t = mklFFT<T>;
    using spatial_type = T;
    using fourier_type = complex_t<T>;

    // 计算布局
    auto [x, y] = cal_layout<fourier_type>(col_major_dims);
    std::vector<spatial_type> image(x * y);

    {   // 随机填充
        uniform_random<spatial_type> r(1e-2, 1);
        std::generate(image.begin(), image.end(), r);
    }
    for(auto& n : image)n = 1;
    // === Forward FFT ===
    auto plan_fwd = fft_t::make_plan({col_major_dims.begin(), col_major_dims.end()});
    std::vector<fourier_type> freq(x * y);

    std::cout << "input     =" << image << std::endl;
    fft_t::exec_forward(*plan_fwd, image.data(), freq.data());

    // 用 python golden 验证 forward
    // catch_py_error(assert(py_loader(".")["cuda_test_fft_golden"]["check_rfft"](
    //     create_ndarray_from_vector(image, col_major_dims),
    //     create_ndarray_from_vector(freq, {static_cast<int>(freq.size())/y, y})
    // )));
    std::cout << "fft result=" << freq << std::endl;

    // === Backward FFT ===
    auto plan_bwd = fft_t::make_plan({col_major_dims.begin(), col_major_dims.end()});
    std::vector<spatial_type> recovered(x * y);

    fft_t::exec_backward(*plan_bwd, freq.data(), recovered.data());
    std::cout << "ifft result=" << recovered << std::endl;

    // backward 结果默认没有除 N，手动归一化
    double scale = 1.0;
    for(auto d : col_major_dims) scale *= d;
    for(auto& v : recovered) v /= scale;

    // 验证 fft->ifft ≈ 原图
    for(size_t i=0;i<image.size();++i){
        if(std::abs(recovered[i] - image[i]) > 1e-3){
            throw std::runtime_error("FFT->IFFT mismatch!");
        }
    }

    printf("*%s    test success\n", space.c_str());
}

// =============== 驱动函数 ===============
int main()
{
    std::complex<float> c2c_input[32];
std::complex<float> c2c_output[32];
float r2c_input[32];

float r2c_output[34];
DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
MKL_LONG status;
    for(int i = 0; i < 32;i++){
        r2c_input[i] = i;
        c2c_input[i] = i;
    }

/* ...put values into c2c_input[i] 0<=i<=31 */
/* ...put values into r2c_input[i] 0<=i<=31 */

status = DftiCreateDescriptor(&my_desc1_handle, DFTI_SINGLE,
                              DFTI_COMPLEX, 1, 32);
status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
status = DftiCommitDescriptor(my_desc1_handle);
status = DftiComputeForward(my_desc1_handle, c2c_input, c2c_output);
status = DftiFreeDescriptor(&my_desc1_handle);
/* result is c2c_output[i] 0<=i<=31 */
status = DftiCreateDescriptor(&my_desc2_handle, DFTI_SINGLE,
                              DFTI_REAL, 1, 32);
status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
status = DftiCommitDescriptor(my_desc2_handle);
status = DftiComputeForward(my_desc2_handle, r2c_input, r2c_output);
status = DftiFreeDescriptor(&my_desc2_handle);
/* result is the complex r2c_data[i] 0<=i<=31   and is stored in CCS format*/
    for(int i = 0; i < 16;i++){
        std::cout << vec2<complex_t<float>>{reinterpret_cast<std::complex<float>*>(r2c_output)[i], c2c_output[i]} << std::endl;
    }
return 0;
    std::vector<std::vector<int>> dims_1d = {{7}, {8}};
    std::vector<std::vector<int>> dims_2d = {{7,5}, {8,6}};
    std::vector<std::vector<int>> dims_3d = {{7,5,3}, {8,6,4}};

    // 1D
    for(auto d : dims_1d){
        test_mkl_fft_check<float>("1D", d);
        test_mkl_fft_check<double>("1D", d);
        test_mkl_fft_check<std::complex<float>>("1D", d);
        test_mkl_fft_check<std::complex<double>>("1D", d);
    }

    // 2D
    for(auto d : dims_2d){
        test_mkl_fft_check<float>("2D", d);
        test_mkl_fft_check<double>("2D", d);
        test_mkl_fft_check<std::complex<float>>("2D", d);
        test_mkl_fft_check<std::complex<double>>("2D", d);
    }

    // 3D
    for(auto d : dims_3d){
        test_mkl_fft_check<float>("3D", d);
        test_mkl_fft_check<double>("3D", d);
        test_mkl_fft_check<std::complex<float>>("3D", d);
        test_mkl_fft_check<std::complex<double>>("3D", d);
    }
}
