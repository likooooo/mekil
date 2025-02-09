#include <mkl_wrapper.hpp>
#include <type_traist_notebook/pretty_print.hpp>

int main()
{
    int xsize, ysize;
    xsize = ysize = 4;
    std::vector<float> vec(xsize * ysize);    
    vec.front() = 1;
    std::cout << "CenterCornerFlip test. from " << vec;
    CenterCornerFlip(vec.data(), xsize, ysize);
    std::cout <<" to "<< vec << std::endl;
}