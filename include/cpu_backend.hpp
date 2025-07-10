#pragma once
#include <type_traist_notebook/type_traist.hpp>
#include <type_traist_notebook/uca/backend.hpp>
namespace uca
{    
    template<class T>struct cpu_backend : backend<T>
    {
        using value_type = T;
        using alloc_type = std::allocator<T>;
        cpu_backend();
        static cpu_backend& ref()
        {
            static cpu_backend cpu;
            return cpu;
        }
    };
    template<class T> using cpu = cpu_backend<T>;
}