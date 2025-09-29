
#include <mkl_fft.hpp>
#include <assert.h>

template<class T> inline std::pair<int ,int> cal_layout(const std::vector<int>&  col_major_dim){
    auto prod = std::accumulate(col_major_dim.begin() + 1, col_major_dim.end(), (size_t)1, [](auto a, auto b) {return a * b; });
    auto change_fastest_axis = (std::is_floating_point_v<T> ? (col_major_dim.front() / 2 + 1) * 2 : col_major_dim.front());
    return {change_fastest_axis, prod};
}

template<class T>
void test_mkl_fft_ifft_out_of_place(const std::string& space, std::vector<int> col_major_dims)
{
    printf("*%s test mklFFT_out_of_place<%s>", space.c_str(), TypeReflection<T>().c_str());
    std::cout << col_major_dims << std::endl;
    using namespace mekil;
    using fft_t = mklFFT<T>;
    using spatial_type = T;
    using fourier_type = complex_t<T>;

    int prod = 1;for(int n:col_major_dims) prod *= n;
    std::vector<spatial_type> image(prod);
    for(int i = 0; i < prod;i++) image.at(i) = real_t<T>(i)/prod;

    auto plan_fwd = fft_t::make_plan({col_major_dims.begin(), col_major_dims.end()});
    std::vector<fourier_type> freq(prod);
    fft_t::exec_forward(*plan_fwd, image.data(), freq.data());
    std::vector<spatial_type> recovered(prod);
    fft_t::exec_backward(*plan_fwd, freq.data(), recovered.data());

    for(size_t i=0;i<prod;++i){
        if(std::abs((recovered.at(i) - image.at(i))) > (is_s<real_t<T>> ? 1e-6 : 1e-15)){
            throw std::runtime_error("FFT->IFFT mismatch! " + std::to_string(std::abs(recovered.at(i) - image.at(i))));
        }
    }
    printf("*%s    test success\n", space.c_str());
}

template<class T>
void test_mkl_fft_ifft_inplace(const std::string& space, std::vector<int> col_major_dims)
{
    printf("*%s test mklFFT_inplace<%s>", space.c_str(), TypeReflection<T>().c_str());
    std::cout << col_major_dims << std::endl;
    using namespace mekil;
    using fft_t = mklFFT<T>;
    using spatial_type = T;
    using fourier_type = complex_t<T>;

    auto [xstride, y] = cal_layout<spatial_type>(col_major_dims);
    int N = 1; for(int n:col_major_dims) N *= n;
    std::vector<spatial_type> image(xstride * y);
    for(int iy = 0; iy < y; iy++){
        int ix = 0;
        for(; ix < col_major_dims.front(); ix++){
            image.at(iy * xstride + ix) = real_t<T>((ix+1) * (iy+1)) / real_t<T>(N*N);
        }
        //== PADDING
        for(;ix < xstride; ix++){
            image.at(iy * xstride + ix) = NAN;
        }
    }

    auto plan_fwd = fft_t::make_plan({col_major_dims.begin(), col_major_dims.end()}, true);
    std::vector<spatial_type> freq = image;
    fft_t::exec_forward(*plan_fwd, freq.data());
    std::vector<spatial_type> recovered = freq;
    fft_t::exec_backward(*plan_fwd, recovered.data());

    for(int iy = 0; iy < y; iy++){
        int ix = 0;
        for(; ix < col_major_dims.front(); ix++){
            int i = ix + iy * xstride;
            if(std::abs(recovered.at(i) - image.at(i)) > 1e-6){
                throw std::runtime_error("FFT->IFFT mismatch! " + std::to_string(std::abs(recovered.at(i) - image.at(i))));
            }
        }
    }
    if(is_real_v<T>){
        recovered.resize(xstride);
        std::cout <<"*"<< space <<  "    final=" << recovered << std::endl;
    }
    printf("*%s    test success\n\n", space.c_str());
}

template<class T>
void test_mkl_fft_ifft(const std::string& space, std::vector<int> col_major_dims)
{
    test_mkl_fft_ifft_out_of_place<T>(space, col_major_dims);
    if(col_major_dims.size() > 1 && is_real_v<T>){
        printf("*%s test mklFFT_inplace<%s>", space.c_str(), TypeReflection<T>().c_str());
        std::cout << col_major_dims << std::endl;
        printf("*%s    test failed. (NOT SUPPORT)\n\n",space.c_str());
    }
    else{
        test_mkl_fft_ifft_inplace<T>(space, col_major_dims);
    }
}
int main()
{
    std::vector<std::vector<int>> dims_1d = {{7}, {8}};
    std::vector<std::vector<int>> dims_2d = {{7,5}, {8,6}};
    std::vector<std::vector<int>> dims_3d = {{7,5,3}, {8,6,4}};

    // return 0;
    for(auto d : dims_2d){
        test_mkl_fft_ifft<std::complex<float>>("2D", d);
        test_mkl_fft_ifft<std::complex<double>>("2D", d);
    }

    for(auto d : dims_3d){
        test_mkl_fft_ifft<std::complex<float>>("3D", d);
        test_mkl_fft_ifft<std::complex<double>>("3D", d);
    }
    for(auto d : dims_1d){
        test_mkl_fft_ifft<std::complex<float>>("1D", d);
        test_mkl_fft_ifft<std::complex<double>>("1D", d);
    }
    for(auto d : dims_1d){
        test_mkl_fft_ifft<float>("1D", d);
        test_mkl_fft_ifft<double>("1D", d);
    }
    for(auto d : dims_2d){
        test_mkl_fft_ifft<float>("2D", d);
        test_mkl_fft_ifft<double>("2D", d);
    }
    for(auto d : dims_3d){
        test_mkl_fft_ifft<float>("3D", d);
        test_mkl_fft_ifft<double>("3D", d);
    }
}
