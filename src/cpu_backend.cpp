#include <cpu_backend.hpp>
#include <mkl_basic_operator.h>

namespace uca
{
    template<class T> void cpu_backend_impl(cpu_backend<T>& cpu)
    {
        cpu.enable =  true;
        cpu.VtAdd = mekil::VtAdd<T>;
        cpu.integral_x = mekil::integral_x<T>;
        cpu.integral_y = mekil::integral_y<T>;
    }
    template<> cpu_backend<float>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<double>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<complex_t<float>>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<complex_t<double>>::cpu_backend(){cpu_backend_impl(*this);}
}