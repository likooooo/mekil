#include <mkl_fft.hpp>

std::vector<int> random_shape()
{
    uniform_random<int> random_dim(1, 4);
    uniform_random<int> random_size(1, 128);
    // uniform_random<int> random_size(2, 5);

    std::vector<int> shape(random_dim());
    for(auto& n : shape) n = random_size(); 
    return shape;
}
template<class T> std::vector<T> test_pattern(const std::vector<int>& shape)
{
    size_t N = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    uniform_random<int> random(-5, 5);
    std::vector<T> pattern(N, 0);
    auto [xstride, ysize] = mekil::cal_fft_memory_layout<T>(shape);
    xstride = shape.front();
    size_t a = std::floor(0.25 * xstride);
    size_t b = std::floor(0.75 * xstride);
    size_t size = (1 < shape.size()? shape.at(1): 1); 
    size_t a1 = size_t(std::floor(0.25 * size));
    size_t b1 = size_t(std::ceil(0.75 * size));  
    for(size_t y = 0; y < ysize; y++){
        size_t i = y % size;
        if(a1 < i && i < b1) continue;
        for(size_t x = 0; x < shape.front(); x++){
            if(x <= a || x >=b){
                auto& n = pattern.at(x + y*shape.front());
                if constexpr(is_complex_v<T>) n = T(random(), random());
                else n = random(); 
            }
        }
    }
    return pattern;
}
template<class scalar> int fft_operator_inplace_test()
{
    using scalarTo = complex_t<scalar>;
    auto shape = random_shape();
    printf("\n * [inplace] fft operator test (%s -> %s), shape : %s\n", 
        TypeReflection<scalar>().c_str(), 
        TypeReflection<scalarTo>().c_str(), 
        to_string(shape).c_str()
    );
    std::vector<scalar> origin = test_pattern<scalar>(shape);
    auto [xstride, ysize] = mekil::cal_fft_memory_layout<scalar>(shape);
    std::vector<scalar> input; 
    if(is_real_v<scalar>){
        std::vector<scalar> origin_with_padding(xstride * ysize);
        crop_image<scalar>(origin_with_padding.data(), {size_t(xstride), size_t(ysize)}, {0, 0}, origin.data(), {origin.size()/ysize, size_t(ysize)}, {0, 0});
        input.swap(origin_with_padding);
    }
    else{
        input = origin;
    }
    auto row_major_dim = shape;
    std::reverse(row_major_dim.begin(), row_major_dim.end());
    using fft = mekil::fftw<scalar, scalarTo>;
    using ifft = mekil::fftw<scalarTo, scalar>;

    void* pIn, *pOut;
    pIn = pOut = input.data();
    auto check_fft_result = [&](){return true;};
    {
        trace_print<void>_("    fft ", "", 0);
        fft::transform(fft::make_plan(row_major_dim, FFTW_FORWARD, pIn, pOut).get());
    }
    assert(check_fft_result());
    {
        trace_print<void>_("    ifft", "", 0);
        ifft::transform(ifft::make_plan(row_major_dim, FFTW_BACKWARD).get(), pOut, pIn);
    }
    std::vector<scalar> recorverd;
    if(is_real_v<scalar>){
        std::vector<scalar> origin_without_padding(origin.size());
        crop_image<scalar>(origin_without_padding.data(), {origin.size()/ysize, size_t(ysize)}, {0, 0}, input.data(), {size_t(xstride), size_t(ysize)}, {0, 0});
        recorverd.swap(origin_without_padding);
    }
    else{
        recorverd = input;
    }
    recorverd /= scalar(origin.size());
    recorverd -= origin;
    real_t<scalar> max_error = std::abs(*std::max_element(recorverd.begin(), recorverd.end(), [](scalar a, scalar b){
        return std::abs(a) < std::abs(b);
    }));
    max_error /= origin.size();
    if(max_error > 1e-6){
        print_matrix(origin, ysize, origin.size()/ysize);
    }
    printf("    max error          %e\n"
           "    N                  %e\n", 
           max_error, 
           1.0 * origin.size()
    );
    printf("    accumulative error %e\n"
           "    epsilon            %e\n", 
        max_error * origin.size(), 
        std::numeric_limits<real_t<scalar>>::epsilon()
    );
    assert(max_error < 1e-6);
    return 0;
}
template<class scalar> int fft_operator_out_of_place_test()
{
    using scalarTo = complex_t<scalar>;
    auto shape = random_shape();
    printf("\n * [out of place] fft operator test (%s -> %s), shape : %s\n", 
        TypeReflection<scalar>().c_str(), 
        TypeReflection<scalarTo>().c_str(), 
        to_string(shape).c_str()
    );
    std::vector<scalar> origin = test_pattern<scalar>(shape);
    // shape ={5,5};
    // origin = {
    //     3,5,0,0,-1,
    //     -3,5,0,-2,-4,
    //     0,0,0,0,0,
    //     0,0,0,0,0,
    //     -3,1,0,-5,-5,
    // };
    std::vector<scalar> input = origin; 
    auto [xstride, ysize] = mekil::cal_fft_memory_layout<scalar>(shape);
    // print_matrix(input, ysize, input.size()/ysize);
    // printf("====\n");
    auto row_major_dim = shape;
    std::reverse(row_major_dim.begin(), row_major_dim.end());
    using fft = mekil::fftw<scalar, scalarTo>;
    using ifft = mekil::fftw<scalarTo, scalar>;

    void* pIn, *pOut;
    pIn = input.data();
    std::vector<scalarTo> output(is_real_v<scalar> ? size_t(xstride/2 * ysize) : origin.size());
    pOut = output.data();
    auto check_fft_result = [&](){return true;};
    {
        trace_print<void>_("    fft ", "", 0);
        fft::transform(fft::make_plan(row_major_dim, FFTW_FORWARD, pIn, pOut).get());
    }
    // print_matrix(output, ysize, output.size()/ysize);
    // printf("====\n");
    assert(check_fft_result());
    {
        trace_print<void>_("    ifft", "", 0);
        ifft::transform(ifft::make_plan(row_major_dim, FFTW_BACKWARD, pOut, pIn).get());
    }
     
    input /= scalar(origin.size());
    std::vector<scalar> recorverd = input;
    recorverd -= origin;
    real_t<scalar> max_error = std::abs(*std::max_element(recorverd.begin(), recorverd.end(), [](scalar a, scalar b){
        return std::abs(a) < std::abs(b);
    }));
    max_error /= origin.size();
    if(max_error > 1e-6){
        print_matrix(origin, ysize, origin.size()/ysize);
        printf("===== recorved =====\n");
        print_matrix(input, ysize, input.size()/ysize);
    }
    printf("    max error          %e\n"
           "    N                  %e\n", 
           max_error, 
           1.0 * origin.size()
    );
    printf("    accumulative error %e\n"
           "    epsilon            %e\n", 
        max_error * origin.size(), 
        std::numeric_limits<real_t<scalar>>::epsilon()
    );
    assert(max_error < 1e-6);
    return 0;
}

auto run_test()
{
    return std::array{
        fft_operator_inplace_test<float>(),
        fft_operator_inplace_test<std::complex<float>>(),
        fft_operator_inplace_test<double>(),
        fft_operator_inplace_test<std::complex<double>>(),

        fft_operator_out_of_place_test<float>(),
        fft_operator_out_of_place_test<std::complex<float>>(),
        fft_operator_out_of_place_test<double>(),
        fft_operator_out_of_place_test<std::complex<double>>(),
        0
    };
}

int main()
{
    mekil::print_fftw_version();
    int repeat_count = 2;
    for(int i = 0; i < repeat_count; i++){
       run_test();
    }
    std::cout <<"\n---------------------------\n   test end\n";
}
