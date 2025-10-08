#include <mkl_linespace.hpp>

int main()
{
    // ===== Test linespace =====
    {
        std::cout << "Test linespace:\n";
        size_t n = 5;
        std::vector<double> arr(n);
        mkl::linespace(arr.data(), n, 0.0, 1.0); // 0,1,2,3,4
        for (auto v : arr) std::cout << v << " ";
        std::cout << "\n\n";
    }

    // ===== Test meshgrid_nd N=2 =====
    {
        std::cout << "Test meshgrid_nd N=2:\n";
        vec<size_t, 2> num = {3, 4};
        vec<float, 2> start = {0.0, 10.0};
        vec<float, 2> step = {1.0, 2.0};

        std::vector<vec2<float>> out(product(num));
        mkl::meshgrid_nd(out.data(), num, start, step);
        print_matrix(out, num[1], num[0]);
    }

    // ===== Test meshgrid_nd N=3 =====
    {
        std::cout << "Test meshgrid_nd N=3:\n";
        vec<size_t, 3> num = {2, 2, 2};
        vec<double, 3> start = {0.0, 10.0, 100.0};
        vec<double, 3> step = {1.0, 10.0, 100.0};

        std::vector<vec3<double>> out(product(num));
        mkl::meshgrid_nd(out.data(), num, start, step);
        print_matrix(out, product(num)/num.front(), num.front());
    }
    
}