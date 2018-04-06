#ifndef MKL_BLAS_FOR_JAVA_HPP
#define MKL_BLAS_FOR_JAVA_HPP

/* MKL Blas Level 1 */
extern "C" double sum(const int dataSize, const double data[]);

extern "C" void aV1_PV0(const int dim, const double a, const double V1[], double V0[]);

extern "C" void a_PV0(const int dim, const double a, double V0[]);

extern "C" void copyV1toV2(const int dim, const double V1[], double V0[]);

extern "C" double innerProductV1V2(const int dim, const double V1[], const double V2[]);

extern "C" double normV1(const int dim, const double V1[]);

extern "C" void aV0(const int dim, const double a, double V0[]);

extern "C" void swapV1V2(const int dim, double V1[], double V2[]);

extern "C" int indexOfMaxAbsV1(const int dim, const double V1[]);

extern "C" int indexOfMinAbsV1(const int dim, const double V1[]);

/* MKL Blas Level 2 */
extern "C" void aM1_TV1_PbV0(const char isTransposeM1, const int M1_row, const int M1_col, const double a,
                             const double M1[], const double V1[], const double b, double V0[]);

extern "C" void aV1_TtransV2_PM0(const int M0_row, const int M0_col, const double a,
                                 const double V1[], const double V2[], double M0[]);

extern "C" void aM1_TV1_PbV0_for_symmetry_M1(const char referenceUpOrLo, const int dim, const double a,
                             const double symmetryMatrixOfM1[], const double V1[], const double b, double V0[]);

extern "C" void aV1_TtransV1_PM0(const char referenceUpOrLo, const int dim, const double a,
                                 const double V1[], double symmetryM0[]);

extern "C" void aV1_TtransV2_PaV2_TtransV1_PM0(const char referenceUpOrLo, const int dim, const double a,
                                               const double V1[], const double V2[], double symmetryM0[]);

/* MKL Blas Level 3 */
extern "C" void aM1_TM2_PbM0(const int M1M0_row, const int M2M0_col, const int M1_colM2_row,
                             const double a, const double M1[], const double M2[], const double b, double M0[]);

extern "C" void aM1_TM2_PbM0_for_symmetry_M1(const char referenceUpOrLo, const int M2M0_row, const int M2M0_col,
                             const double a, const double symmetryM1[], const double M2[],
                             const double b, double M0[]);

extern "C" void aM2_TM1_PbM0(const char referenceUpOrLo, const int M2M0_row, const int M2M0_col,
                             const double a, const double symmetryM1[], const double M2[],
                             const double b, double M0[]);

extern "C" void aM1_TtransM1_PbM0(const char referenceUpOrLo, const int M1_rowM0_dim, const int M1_col,
                                  const double a, const double M1[], const double b, double symmetryM0[]);

extern "C" void aM1_TtransM2_PaM2_TtransM1_PbM0(const char referenceUpOrLo,
                                                const int M1M2_rowM0_dim, const int M1M2_col,
                                                const double a, const double M1[], const double M2[],
                                                const double b, double symmetryM0[]);

/* MKL CBLAS */
extern "C" void aV1_ElementwiseMultiplyV2_PbV0(const int dim, const double a, const double V1[],
                                               const double V2[], const double b, double V0[]);

/* MKL Blas-like Extension */
extern "C" void copyatransM1ToM2(const int M2_row, const int M2_col,
                                 const double a, const double M1[], double M2[]);

/* MKL LAPACKE */
extern "C" int doLUDecompositionWithLAPACKE_dgetrf(int M0_row, int M0_col, double M0[], int ipiv[]);

extern "C" int calcInverseWithLAPACKE_dgetri(int dim, double LUMatrix[], const int ipiv[]);

/* Vector Mathematical Functions */
extern "C" void elementWiseAdd(const int dim, const double V1[], const double V2[], double result[]);

extern "C" void elementWiseSub(const int dim, const double V1[], const double V2[], double result[]);

extern "C" void elementWiseMul(const int dim, const double V1[], const double V2[], double result[]);

extern "C" void elementWiseInv(const int dim, const double V1[], double result[]);

extern "C" void elementWiseDiv(const int dim, const double V1[], const double V2[], double result[]);

extern "C" void elementWiseSqrt(const int dim, const double V1[], double result[]);

extern "C" void elementWiseInvSqrt(const int dim, const double V1[],  double result[]);

extern "C" void elementWiseExp(const int dim, const double V1[],  double result[]);

extern "C" void elementWiseLn(const int dim, const double V1[],  double result[]);

extern "C" void elementWiseCos(const int dim, const double V1[],  double result[]);

extern "C" void elementWiseSin(const int dim, const double V1[],  double result[]);

extern "C" void elementWiseTan(const int dim, const double V1[],  double result[]);

#endif
