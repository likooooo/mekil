#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <type_traist_notebook/type_traist.hpp>
#include <mkl_fft.hpp>

template<class T>void inplace_test()
{
    constexpr size_t N = 32;
    vec<real_t<T>, N + 2> r2c_data;
    vec<complex_t<T>, N> c2c_data; 
    for(size_t i = 0; i < N; i++){
        c2c_data.at(i) = r2c_data.at(i) = 2 * cos(2 * M_PI * 2 * i / N) + 10 * cos(2 * M_PI * 3 * i / N);
    }

    DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
    DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
    MKL_LONG status;

    /* ...put values into c2c_data[i] 0<=i<=31 */
    /* ...put values into r2c_data[i] 0<=i<=31 */

    status = DftiCreateDescriptor(&my_desc1_handle, mekil::mklFFT<T>::dft_precision,
                                DFTI_COMPLEX, 1, N);
    status = DftiCommitDescriptor(my_desc1_handle);
    status = DftiComputeForward(my_desc1_handle, c2c_data.data());
    status = DftiFreeDescriptor(&my_desc1_handle);
    /* result is c2c_data[i] 0<=i<=31 */
    status = DftiCreateDescriptor(&my_desc2_handle, mekil::mklFFT<T>::dft_precision,
                                DFTI_REAL, 1, N);
    status = DftiCommitDescriptor(my_desc2_handle);
    status = DftiComputeForward(my_desc2_handle, r2c_data.data());
    status = DftiFreeDescriptor(&my_desc2_handle);
    // std::cout << c2c_data << std::endl;
    // std::cout << reinterpret_cast<vec<complex_t<T>, N / 2 +1>&>(r2c_data) << std::endl;
    std::cout << (reinterpret_cast<vec<complex_t<T>, N / 2 +1>&>(r2c_data) - reinterpret_cast<vec<complex_t<T>, N/2 + 1>&>(c2c_data))<< std::endl;
}

template<class T>
void outplace_test()
{
    constexpr size_t N = 32;
    vec<real_t<T>, N + 2> r2c_input{0};
    vec<complex_t<T>, N> c2c_input; 
    for(size_t i = 0; i < N; i++){
        c2c_input.at(i) = r2c_input.at(i) = 2 * cos(2 * M_PI * 2 * i / N) + 10 * cos(2 * M_PI * 3 * i / N);
    }
    vec<real_t<T>, N + 2> r2c_output;
    vec<complex_t<T>, N> c2c_output; 

    DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
    DFTI_DESCRIPTOR_HANDLE my_desc2_handle = NULL;
    MKL_LONG status;


    status = DftiCreateDescriptor(&my_desc1_handle, mekil::mklFFT<T>::dft_precision,
                                DFTI_COMPLEX, 1, 32);
    status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptor(my_desc1_handle);
    status = DftiCreateDescriptor(&my_desc2_handle, mekil::mklFFT<T>::dft_precision,
                                DFTI_REAL, 1, 32);
    status = DftiSetValue(my_desc2_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptor(my_desc2_handle);

    status = DftiComputeForward(my_desc1_handle, c2c_input.data(), c2c_output.data());
    status = DftiFreeDescriptor(&my_desc1_handle);
    /* result is c2c_output[i] 0<=i<=31 */
    status = DftiComputeForward(my_desc2_handle, r2c_input.data(), r2c_output.data());
    status = DftiFreeDescriptor(&my_desc2_handle);

    auto error = (reinterpret_cast<vec<complex_t<T>, N / 2 +1>&>(r2c_output) - reinterpret_cast<vec<complex_t<T>, N/2 + 1>&>(c2c_output));
    std::cout << error << std::endl;
}
int main()
{
    
    inplace_test<float>();
    outplace_test<float>();
    /* result is the complex value r2c_data[i] 0<=i<=31 */
    /* and is stored in CCS format*/
}

// int main() {
//     // 设置FFT参数
//     const int N = 16;           // 实数输入数据点数
//     const int N_out = N/2 + 1;  // 复数输出数据点数 (对称性)

//     // 分配内存
//     float *in = (float*)mkl_malloc(N * sizeof(float), 64);      // 实数输入
//     MKL_Complex8 *out = (MKL_Complex8*)mkl_malloc(N_out * sizeof(MKL_Complex8), 64); // 复数输出

//     // 初始化输入数据 (示例: 正弦波)
//     for (int i = 0; i < N; i++) {
//         in[i] = cos(2 * M_PI * 2 * i / N) + 10 * cos(2 * M_PI * 3 * i / N);
//     }

//     // 创建FFT描述符
//     DFTI_DESCRIPTOR_HANDLE handle;
//     MKL_LONG status;

//     // 创建实数到复数FFT描述符
//     status = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_REAL, 1, N);
//     if (status != DFTI_NO_ERROR) {
//         printf("Error creating descriptor\n");
//         return 1;
//     }

//     // 设置输出为紧凑格式 (只存储非冗余部分)
//     status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
//     if (status != DFTI_NO_ERROR) {
//         printf("Error setting output format\n");
//         return 1;
//     }

//     // 提交描述符
//     status = DftiCommitDescriptor(handle);
//     if (status != DFTI_NO_ERROR) {
//         printf("Error committing descriptor\n");
//         return 1;
//     }

//     printf("Input (real):\n");
//     for (int i = 0; i < N; i++) {
//         printf("%.3f ", in[i]);
//     }
//     // 执行FFT (实数到复数)
//     status = DftiComputeForward(handle, in, out);
//     if (status != DFTI_NO_ERROR) {
//         printf("Error computing FFT\n");
//         return 1;
//     }

//     // 打印结果
//     printf("Input (real):\n");
//     for (int i = 0; i < N; i++) {
//         printf("%.3f ", in[i]);
//     }
//     printf("\n\nFFT output (complex):\n");
//     for (int i = 0; i < N_out; i++) {
//         printf("Bin %2d: %7.3f + %7.3fj\n", i, out[i].real, out[i].imag);
//     }

//     // 清理
//     DftiFreeDescriptor(&handle);
//     mkl_free(in);
//     mkl_free(out);

//     return 0;
// }