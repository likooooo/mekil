#include <mkl_wrapper.hpp>
#include <mkl_vec.hpp>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <cmath>

template<typename T>
void check_equal(const std::vector<T>& a, const std::vector<T>& b, const std::string& msg) {
    if (a.size() != b.size())
        throw std::runtime_error("Size mismatch in test: " + msg);

    for (size_t i = 0; i < a.size(); i++) {
        if constexpr (std::is_floating_point<T>::value) {
            if (std::fabs(a[i] - b[i]) > 1e-6)
                throw std::runtime_error("Value mismatch at " + std::to_string(i) + " in test: " + msg);
        } else {
            if (a[i] != b[i])
                throw std::runtime_error("Value mismatch at " + std::to_string(i) + " in test: " + msg);
        }
    }
}

// ------------------- 各种测试 -------------------

void test_int_2D_Cstyle() {
    std::vector<int> shape = {2, 3};
    std::vector<int> perm  = {1, 0}; // 转置

    int input[6]  = {1, 2, 3, 4, 5, 6};
    int output[6];
    auto cb = [](int v) { return v; };

    permuteND<int,int,decltype(cb), true>(input, output, shape, perm, cb);

    // 转置结果: [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
    std::vector<int> expect = {1,4,2,5,3,6};
    check_equal(std::vector<int>(output, output+6), expect, "int_2D_Cstyle");
}

void test_float_3D_Fstyle() {
    std::vector<int> shape = {2, 2, 2};
    std::vector<int> perm  = {2, 1, 0}; // reverse axes

    float input[8] = {0,1,2,3,4,5,6,7};
    float output[8];
    auto cb = [](float v) { return v + 0.5f; }; // 顺便测试 callback

    permuteND<float,float,decltype(cb), false>(input, output, shape, perm, cb);

    // 验证：手工构造预期 (反转坐标 + 加0.5)
    std::vector<float> expect = {0.5f, 4.5f, 2.5f, 6.5f, 1.5f, 5.5f, 3.5f, 7.5f};
    check_equal(std::vector<float>(output, output+8), expect, "float_3D_Fstyle");
}

void test_double_1D() {
    std::vector<int> shape = {5};
    std::vector<int> perm  = {0}; // 恒等

    double input[5] = {10,20,30,40,50};
    double output[5];
    auto cb = [](double v) { return v*2; };

    permuteND<double,double,decltype(cb), true>(input, output, shape, perm, cb);

    std::vector<double> expect = {20,40,60,80,100};
    check_equal(std::vector<double>(output, output+5), expect, "double_1D");
}

void test_int_4D_mixed() {
    std::vector<int> shape = {2, 2, 2, 2};
    std::vector<int> perm  = {1, 0, 3, 2};

    int input[16];
    for (int i = 0; i < 16; i++) input[i] = i;
    int output[16];

    auto cb = [](int v){return v;};

    permuteND<int,int,decltype(cb), true>(input, output, shape, perm, cb);

    std::vector<int> out(output, output+16);
    std::vector<int> in(input, input+16);
    std::sort(out.begin(), out.end());
    std::sort(in.begin(), in.end());
    check_equal(out, in, "int_4D_mixed (set equality)");
}
void test_transpose()
{
    std::array<int,2> shape = {2,3};
    float A[6] = {1,2,3,4,5,6};
    float B[6];

    transpose<float,true>(A, B, shape); // C-style

    mkl::vec::add(6, B,B,B);
    for (int i=0; i<6; i++) std::cout << B[i] << " ";
    std::cout << std::endl;
}
int main() {
    try {
        test_int_2D_Cstyle();
        test_float_3D_Fstyle();
        test_double_1D();
        test_int_4D_mixed();
        test_transpose();
    } catch (const std::exception& ex) {
        std::cerr << "Test failed: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}