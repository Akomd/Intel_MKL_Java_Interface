#include "mkl.h"
#include "mkl_blas_for_java.hpp"


/* MKL Blas Level 1 */
extern "C" double sum(const int dataSize, const double data[])
{
    const int incx = 1;
    return dasum(&dataSize, data, &incx);
}

extern "C" void aV1_PV0(const int dim, const double a, const double V1[], double V0[])
{
    const int incxy = 1;
    daxpy(&dim, &a, V1, &incxy, V0, &incxy);
}

extern "C" void a_PV0(const int dim, const double a, double V0[])
{
    const double V1[] = {1.0};
    const int incx = 0;
    const int incy = 1;
    daxpy(&dim, &a, V1, &incx, V0, &incy);
}

extern "C" void copyV1toV2(const int dim, const double V1[], double V2[])
{
    const int incxy = 1;
    dcopy(&dim, V1, &incxy, V2, &incxy);
}

extern "C" double innerProductV1V2(const int dim, const double V1[], const double V2[])
{
    const int incxy = 1;
    return ddot(&dim, V1, &incxy, V2, &incxy);
}

extern "C" double normV1(const int dim, const double V1[])
{
    const int incx = 1;
    return cblas_dnrm2(dim, V1, incx);
}

extern "C" void aV0(const int dim, const double a, double V0[])
{
    const int incx = 1;
    dscal(&dim, &a, V0, &incx);
}

extern "C" void swapV1V2(const int dim, double V1[], double V2[])
{
    const int incxy = 1;
    dswap(&dim, V1, &incxy, V2, &incxy);
}

extern "C" int indexOfMaxAbsV1(const int dim, const double V1[])
{
    const int incx = 1;
    return idamax(&dim, V1, &incx);
}

extern "C" int indexOfMinAbsV1(const int dim, const double V1[])
{
    const int incx = 1;
    return idamin(&dim, V1, &incx);
}

/* MKL Blas Level 2 */
extern "C" void aM1_TV1_PbV0(const char isTransposeM1, const int M1_row, const int M1_col, const double a,
                             const double M1[], const double V1[], const double b, double V0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_TRANSPOSE trans = CblasNoTrans;
    if(isTransposeM1 == 'n'){
        trans = CblasNoTrans;
    } else if(isTransposeM1 == 't'){
        trans = CblasTrans;
    }
    const int incxy = 1;
    cblas_dgemv(Layout, trans, M1_row, M1_col, a, M1, M1_col, V1, incxy, b, V0, incxy);
}

extern "C" void aV1_TtransV2_PM0(const int M0_row, const int M0_col, const double a,
                                 const double V1[], const double V2[], double M0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    const int incxy = 1;
    cblas_dger(Layout, M0_row, M0_col, a, V1, incxy, V2, incxy, M0, M0_col);
}

extern "C" void aM1_TV1_PbV0_for_symmetry_M1(const char referenceUpOrLo, const int dim, const double a,
                             const double symmetryM1[], const double V1[], const double b, double V0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    if(referenceUpOrLo == 'u')
    {
        uplo = CblasUpper;
    } else if(referenceUpOrLo == 'l'){
        uplo = CblasLower;
    }
    const int incxy = 1;
    cblas_dsymv(Layout, uplo, dim, a, symmetryM1, dim, V1, incxy, b, V0, incxy);
}

extern "C" void aV1_TtransV1_PM0(const char referenceUpOrLo, const int dim, const double a,
                                 const double V1[], double symmetryM0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    if(referenceUpOrLo == 'u')
    {
        uplo = CblasUpper;
    } else if(referenceUpOrLo == 'l'){
        uplo = CblasLower;
    }
    const int incx = 1;
    cblas_dsyr(Layout, uplo, dim, a, V1, incx, symmetryM0, dim);
}

extern "C" void aV1_TtransV2_PaV2_TtransV1_PM0(const char referenceUpOrLo, const int dim, const double a,
                                               const double V1[], const double V2[], double symmetryM0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    if(referenceUpOrLo == 'u')
    {
        uplo = CblasUpper;
    } else if(referenceUpOrLo == 'l'){
        uplo = CblasLower;
    }
    const int incxy = 1;
    cblas_dsyr2(Layout, uplo, dim, a, V1, incxy, V2, incxy, symmetryM0, dim);
}

/* MKL CBlas Level 3 */
extern "C" void aM1_TM2_PbM0(const int M1M0_row, const int M2M0_col, const int M1_colM2_row,
                             const double a, const double M1[], const double M2[], const double b, double M0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    const CBLAS_TRANSPOSE trans = CblasNoTrans;
    cblas_dgemm(Layout, trans, trans, M1M0_row, M2M0_col, M1_colM2_row,
          a, M1, M1_colM2_row, M2, M2M0_col, b, M0, M2M0_col);
}

extern "C" void aM1_TM2_PbM0_for_symmetry_M1(const char referenceUpOrLo, const int M2M0_row, const int M2M0_col,
                             const double a, const double symmetryM1[], const double M2[],
                             const double b, double M0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    const CBLAS_SIDE side = CblasLeft;
    CBLAS_UPLO uplo = CblasUpper;
    if(referenceUpOrLo == 'u')
    {
        uplo = CblasUpper;
    } else if(referenceUpOrLo == 'l'){
        uplo = CblasLower;
    }
    cblas_dsymm(Layout, side, uplo, M2M0_row, M2M0_col, a,
                symmetryM1, M2M0_row, M2, M2M0_col, b, M0, M2M0_col);
}

extern "C" void aM2_TM1_PbM0(const char referenceUpOrLo, const int M2M0_row, const int M2M0_col,
                             const double a, const double symmetryM1[], const double M2[],
                             const double b, double M0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    const CBLAS_SIDE side = CblasRight;
    CBLAS_UPLO uplo = CblasUpper;
    if(referenceUpOrLo == 'u')
    {
        uplo = CblasUpper;
    } else if(referenceUpOrLo == 'l'){
        uplo = CblasLower;
    }
    cblas_dsymm(Layout, side, uplo, M2M0_row, M2M0_col, a,
                symmetryM1, M2M0_col, M2, M2M0_col, b, M0, M2M0_col);
}

extern "C" void aM1_TtransM1_PbM0(const char referenceUpOrLo, const int M1_rowM0_dim, const int M1_col,
                                  const double a, const double M1[], const double b, double symmetryM0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    if(referenceUpOrLo == 'u')
    {
        uplo = CblasUpper;
    } else if(referenceUpOrLo == 'l'){
        uplo = CblasLower;
    }
    const CBLAS_TRANSPOSE trans = CblasNoTrans;
    cblas_dsyrk(Layout, uplo, trans, M1_rowM0_dim, M1_col, a, M1, M1_col, b, symmetryM0, M1_rowM0_dim);
}

extern "C" void aM1_TtransM2_PaM2_TtransM1_PbM0(const char referenceUpOrLo,
                                                const int M1M2_rowM0_dim, const int M1M2_col,
                                                const double a, const double M1[], const double M2[],
                                                const double b, double symmetryM0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_UPLO uplo = CblasUpper;
    if(referenceUpOrLo == 'u')
    {
        uplo = CblasUpper;
    } else if(referenceUpOrLo == 'l'){
        uplo = CblasLower;
    }
    const CBLAS_TRANSPOSE trans = CblasNoTrans;
    cblas_dsyr2k(Layout, uplo, trans, M1M2_rowM0_dim, M1M2_col,
                 a, M1, M1M2_col, M2, M1M2_col, b, symmetryM0, M1M2_rowM0_dim);
}

/* MKL CBLAS */
extern "C" void aV1_ElementwiseMultiplyV2_PbV0(const int dim, const double a, const double V1[],
                                               const double V2[], const double b, double V0[])
{
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    const CBLAS_UPLO uplo = CblasLower;
    const int diagonalFlag = 0;
    const int lda = 1;
    const int incxy = 1;

    cblas_dsbmv(Layout, uplo, dim, diagonalFlag, a, V1, lda, V2, incxy, b, V0, incxy);
}

/* MKL Blas-like Extension */
extern "C" void copyatransM1ToM2(const int M2_row, const int M2_col,
                                 const double a, const double M1[], double M2[])
{
    const char row_major = 'r';
    const char trans = 't';
    mkl_domatcopy(row_major, trans, M2_col, M2_row, a, M1, M2_row, M2, M2_col);
}

/* MKL LAPACKE */
extern "C" int doLUDecompositionWithLAPACKE_dgetrf(int M0_row, int M0_col, double M0[], int ipiv[])
{
    return LAPACKE_dgetrf(LAPACK_ROW_MAJOR, M0_row, M0_col, M0, M0_col, ipiv);
}

extern "C" int calcInverseWithLAPACKE_dgetri(int dim, double LUMatrix[], const int ipiv[])
{
    return LAPACKE_dgetri(LAPACK_ROW_MAJOR, dim, LUMatrix, dim, ipiv);
}

/* Vector Mathematical Functions */
extern "C" void elementWiseAdd(const int dim, const double V1[], const double V2[], double result[])
{
    vdAdd(dim, V1, V2, result);
}

extern "C" void elementWiseSub(const int dim, const double V1[], const double V2[], double result[])
{
    vdSub(dim, V1, V2, result);
}

extern "C" void elementWiseMul(const int dim, const double V1[], const double V2[], double result[])
{
    vdMul(dim, V1, V2, result);
}

extern "C" void elementWiseInv(const int dim, const double V1[],  double result[])
{
    vdInv(dim, V1, result);
}

extern "C" void elementWiseDiv(const int dim, const double V1[], const double V2[], double result[])
{
    vdDiv(dim, V1, V2, result);
}

extern "C" void elementWiseSqrt(const int dim, const double V1[],  double result[])
{
    vdSqrt(dim, V1, result);
}

extern "C" void elementWiseInvSqrt(const int dim, const double V1[],  double result[])
{
    vdInvSqrt(dim, V1, result);
}

extern "C" void elementWiseExp(const int dim, const double V1[],  double result[])
{
    vdExp(dim, V1, result);
}

extern "C" void elementWiseLn(const int dim, const double V1[],  double result[])
{
    vdLn(dim, V1, result);
}

extern "C" void elementWiseCos(const int dim, const double V1[],  double result[])
{
    vdCos(dim, V1, result);
}

extern "C" void elementWiseSin(const int dim, const double V1[],  double result[])
{
    vdSin(dim, V1, result);
}

extern "C" void elementWiseTan(const int dim, const double V1[],  double result[])
{
    vdTan(dim, V1, result);
}

