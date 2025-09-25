#include <type_traist_notebook/type_traist.hpp>
#include <mkl_wrapper.hpp>
#include <assert.h>

template<class T>
struct complex_decompse
{
    template<bool is_real_part = true>
    static std::vector<real_t<T>> get_component_in_complex(const std::vector<complex_t<T>>& input, const std::vector<int>& shape)
    {
        int xsize = shape.front();
        int ysize = 1;
        for(size_t i = 1; i < shape.size(); i++) ysize *= shape.at(i);
        std::vector<real_t<T>> output(xsize * ysize);
        copy_batch_strided<real_t<T>>(xsize, ((const real_t<T>*)input.data()) + int(!is_real_part), 2, 2*xsize, output.data(), 1, xsize, ysize);
        return output;
    }

    static std::vector<complex_t<T>> random_complex(const std::vector<int>& shape)
    {
        int prod = shape.at(0);
        for(size_t i = 1; i < shape.size(); i++) prod *= shape.at(i);
        std::vector<complex_t<T>> output(prod);
        uniform_random<complex_t<T>> rand(0, 1);
        for(auto& n : output)  n = rand();
        return output;
    }

    static int test_complex_to_real(const std::vector<complex_t<T>>& input, const std::vector<int>& shape)
    {
        std::vector<real_t<T>> real_part = get_component_in_complex<true>(input, shape);
        std::vector<real_t<T>> imag_part = get_component_in_complex<false>(input, shape);

        for(size_t i = 0; i < real_part.size(); i++){
            assert(real_part.at(i) == input.at(i).real());
            assert(imag_part.at(i) == input.at(i).imag());
        }
        return 0;
    }

};
void test_complex_decompose()
{
    std::array<std::vector<int>, 5> shape_case{
        std::vector<int>{1270},
        std::vector<int>{45, 233},
        std::vector<int>{37, 26, 7},
        std::vector<int>{3, 5, 7, 9},
        std::vector<int>{1933, 2758},
    };
    for(const auto& shape : shape_case){
        {
            auto input =  complex_decompse<float>::random_complex(shape);
            complex_decompse<float>::test_complex_to_real(input, shape);
            complex_decompse<complex_t<float>>::test_complex_to_real(input, shape);
        }
        {
            auto input =  complex_decompse<double>::random_complex(shape);
            complex_decompse<double>::test_complex_to_real(input, shape);
            complex_decompse<complex_t<double>>::test_complex_to_real(input, shape);
        }
    }
}
void test_crop_to()
{
    std::array<vec2<size_t>, 4> shape_case{
        vec2<size_t>{3, 233},
        vec2<size_t>{217, 73},
        vec2<size_t>{1, 67},
        vec2<size_t>{367, 1},
    };
    for(const auto& shape : shape_case){
        auto input =  complex_decompse<float>::random_complex({int(shape[0]), int(shape[1])});
        std::vector<float> output(input.size());
        crop_to<std::complex<float>, float>(output.data(), shape, {0, 0}, input.data(), shape, {0, 0});
        for(size_t i = 0; i < output.size(); i++)  assert(output.at(i) == input.at(i).real());
        crop_to<std::complex<float>, float>(output.data(), shape, {0, 0}, 
            reinterpret_cast<const std::complex<float>*>(reinterpret_cast<const float*>(input.data()) + 1), 
            shape, {0, 0}
        );
        for(size_t i = 0; i < output.size(); i++)  assert(output.at(i) == input.at(i).imag());
    }
}
int test_crop_image() 
{
    auto print_image = [](const std::vector<double>& img, int width, int height) {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                std::cout << img[y*width + x] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "-------------------\n";
    };
    // ----------------- 1. 小图拷贝到大图 -----------------
    std::vector<double> small_img(4*4);
    for(int i = 0; i < 16; i++) small_img[i] = i;

    std::vector<double> big_img(6*6, -1); // 初始化为 -1
    crop_image(big_img.data(), {6,6}, {1,1},
               small_img.data(), {4,4}, {0,0});

    std::cout << "Small -> Big:\n";
    print_image(big_img, 6,6);

    // ----------------- 2. 大图拷贝到小图 -----------------
    std::vector<double> big_img2(6*6);
    for(int i = 0; i < 36; i++) big_img2[i] = i;

    std::vector<double> small_img2(4*4, -1);
    crop_image(small_img2.data(), {4,4}, {0,0},
               big_img2.data(), {6,6}, {2,2});

    std::cout << "Big -> Small:\n";
    print_image(small_img2, 4,4);

    // ----------------- 3. 两张大图拷贝重叠区域 -----------------
    std::vector<double> big_img3(6*6);
    std::vector<double> big_img4(6*6, -1);
    for(int i = 0; i < 36; i++) big_img3[i] = i;

    crop_image(big_img4.data(), {6,6}, {2,2},
               big_img3.data(), {6,6}, {1,1});

    std::cout << "Big -> Big overlapping:\n";
    print_image(big_img3, 6,6);
    print_image(big_img4, 6,6);

    return 0;
}



int main()
{
    test_crop_to();
    test_crop_image();
    test_complex_decompose();
    std::cout << "all test done\n";
}