#include <cpu_backend.hpp>
#include <mkl_vec.hpp>
#include <mkl_intergral.hpp>

namespace uca
{
    template<class T> void cpu_backend_impl(cpu_backend<T>& cpu)
    {
        cpu.enable =  true;
        cpu.VtAdd = mkl::vec::self_add<T>;
        cpu.integral_x = mkl::integral_x<T>;
        cpu.integral_y = mkl::integral_y<T>;
        // cpu.fft =
        // cpu.fft_outplace =
        // cpu.ifft =
        // cpu.ifft_outplace =
    }
    template<> cpu_backend<float>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<double>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<complex_t<float>>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<complex_t<double>>::cpu_backend(){cpu_backend_impl(*this);}
}