#include <cpu_backend.hpp>
#include <mkl_vec.hpp>
#include <mkl_intergral.hpp>
#include <fftw_fft.hpp>

template<class TFrom, class TTo, int direction = FFTW_FORWARD> void self_fft_impl(TFrom* self, size_t* shape, size_t n)
{
    using fft = mekil::fftw<TFrom, TTo>;
    std::vector<int> row_major_dim(n);
    for(int i = 0; i < n; i++){
        row_major_dim.at(i) = shape[n - 1 - i];
    }
    fft::transform(fft::make_plan(row_major_dim, direction, self, self).get());
}
template<class TFrom, class TTo, int direction = FFTW_FORWARD> void fft_impl(const TFrom* from, TTo* to, size_t* shape, size_t n)
{
    using fft = mekil::fftw<TFrom, TTo>;
    std::vector<int> row_major_dim(n);
    for(int i = 0; i < n; i++){
        row_major_dim.at(i) = shape[n - 1 - i];
    }
    fft::transform(fft::make_plan(row_major_dim, direction, const_cast<TFrom*>(from), to).get());
}
namespace uca
{
    template<class T> void cpu_backend_impl(cpu_backend<T>& cpu)
    {
        cpu.enable =  true;
        cpu.VtAdd = mkl::vec::self_add<T>;
        cpu.integral_x = mkl::integral_x<T>;
        cpu.integral_y = mkl::integral_y<T>;
        cpu.self_fft = self_fft_impl<T, complex_t<T>, FFTW_FORWARD>;
        cpu.self_ifft = self_fft_impl<complex_t<T>, T, FFTW_BACKWARD>;
        cpu.fft = fft_impl<T, complex_t<T>, FFTW_FORWARD>;
        cpu.ifft = fft_impl<complex_t<T>, T, FFTW_BACKWARD>;
    }
    template<> cpu_backend<float>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<double>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<complex_t<float>>::cpu_backend(){cpu_backend_impl(*this);}
    template<> cpu_backend<complex_t<double>>::cpu_backend(){cpu_backend_impl(*this);}
}