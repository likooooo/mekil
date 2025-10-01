#include <mkl_basic_operator.h>

int main()
{
    MKLVersion mkl_version;
    mkl_get_version(&mkl_version);
    printf("\nYou are using oneMKL %d.%d\n", mkl_version.MajorVersion, mkl_version.UpdateVersion);
    return 0;
}