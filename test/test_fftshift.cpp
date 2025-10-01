#include <mkl_reshape.hpp>

// 检查函数：如果不一致，直接抛异常
template <class T>
void assert_equal(const std::vector<T>& a, const std::vector<T>& b, const std::string& msg, int row, int col)
{
    if (a != b) {
        print_matrix(a, row, col);
        printf("==except==\n");
        print_matrix(b, row, col);
        std::cerr << "Test failed: " << msg << std::endl;
        throw std::runtime_error("Mismatch detected in " + msg);
    }
}

int main()
{
    {
        // Case 1: 4x4 偶数维
        std::vector<float> img = {
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,
            13, 14, 15, 16
        };

        std::vector<float> expected = {
            11, 12,  9, 10,
            15, 16, 13, 14,
             3,  4,  1,  2,
             7,  8,  5,  6
        };

        fftshift(img.data(), 4, 4);
        assert_equal(img, expected, "4x4 even case", 4, 4);
    }

    {
        // Case 2: 5x5 奇数维
        std::vector<double> img = {
             1,  2,  3,  4,  5,
             6,  7,  8,  9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25
        };
        //== expected is same with matlab & numpy
        std::vector<double> expected = {
            19, 20, 16, 17, 18,
            24, 25, 21, 22, 23,
             4,  5,  1,  2,  3,
             9, 10,  6,  7,  8,
            14, 15, 11, 12, 13
        };

        fftshift(img.data(), 5, 5);
        assert_equal(img, expected, "5x5 odd case", 5, 5);
    }

    {
        // Case 3: 1xN 边界情况
        std::vector<std::complex<float>> img = {1, 2, 3, 4};
        std::vector<std::complex<float>> expected = {3, 4, 1, 2}; 
        fftshift(img.data(), 4, 1);
        assert_equal(img, expected, "1xN degenerate case", 1, 4);
    }

    {
        // Case 4: Nx1 边界情况
        std::vector<std::complex<double>> img = {1, 2, 3, 4};
        std::vector<std::complex<double>> expected = {3, 4, 1, 2};
        fftshift(img.data(), 1, 4);
        assert_equal(img, expected, "Nx1 degenerate case", 4, 1);
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}