import com.sun.jna.Library;
import com.sun.jna.Native;

/**
 * Intel Math Kernel LibraryのJavaラッパー
 */
public interface mklFunction extends Library {
    mklFunction INSTANCE = (mklFunction) Native.loadLibrary("mkljava", mklFunction.class);
    /* MKL Blas Level 1 */
    double sum(final int dataSize, final double data[]);

    void aV1_PV0(final int dim, final double a, final double V1[], double V0[]);

    void a_PV0(final int dim, final double a, double V0[]);

    void copyV1toV2(final int dim, final double V1[], double V0[]);

    double innerProductV1V2(final int dim, final double V1[], final double V2[]);

    double normV1(final int dim, final double V1[]);

    void aV0(final int dim, final double a, double V0[]);

    void swapV1V2(final int dim, double V1[], double V2[]);

    int indexOfMaxAbsV1(final int dim, final double V1[]);

    int indexOfMinAbsV1(final int dim, final double V1[]);

    /* MKL Blas Level 2 */
    void aM1_TV1_PbV0(final char isTransposeM1, final int M1_row, final int M1_col, final double a,
                      final double M1[], final double V1[], final double b, double V0[]);

    void aV1_TtransV2_PM0(final int M0_row, final int M0_col, final double a,
                          final double V1[], final double V2[], double M0[]);

    void aM1_TV1_PbV0_for_symmetry_M1(final char referenceUpOrLo, final int dim, final double a,
                                      final double symmetryMatrixOfM1[], final double V1[], final double b, double V0[]);

    void aV1_TtransV1_PM0(final char referenceUpOrLo, final int dim, final double a,
                          final double V1[], double symmetryM0[]);

    void aV1_TtransV2_PaV2_TtransV1_PM0(final char referenceUpOrLo, final int dim, final double a,
                                        final double V1[], final double V2[], double symmetryM0[]);

    /* MKL Blas Level 3 */
    void aM1_TM2_PbM0(final int M1M0_row, final int M2M0_col, final int M1_colM2_row,
                      final double a, final double M1[], final double M2[], final double b, double M0[]);

    void aM1_TM2_PbM0_for_symmetry_M1(final char referenceUpOrLo, final int M2M0_row, final int M2M0_col,
                                      final double a, final double symmetryM1[], final double M2[],
                                      final double b, double M0[]);

    void aM2_TM1_PbM0(final char referenceUpOrLo, final int M2M0_row, final int M2M0_col,
                      final double a, final double symmetryM1[], final double M2[],
                      final double b, double M0[]);

    void aM1_TtransM1_PbM0(final char referenceUpOrLo, final int M1_rowM0_dim, final int M1_col,
                           final double a, final double M1[], final double b, double symmetryM0[]);

    void aM1_TtransM2_PaM2_TtransM1_PbM0(final char referenceUpOrLo,
                                         final int M1M2_rowM0_dim, final int M1M2_col,
                                         final double a, final double M1[], final double M2[],
                                         final double b, double symmetryM0[]);

    /* MKL CBLAS */
    void aV1_ElementwiseMultiplyV2_PbV0(final int dim, final double a, final double V1[],
                                        final double V2[], final double b, double V0[]);

    /* MKL Blas-like Extension */
    void copyatransM1ToM2(final int M2_row, final int M2_col,
                          final double a, final double M1[], double M2[]);

    /* MKL LAPACKE */
    int doLUDecompositionWithLAPACKE_dgetrf(int M0_row, int M0_col, double M0[], int ipiv[]);

    int calcInverseWithLAPACKE_dgetri(int dim, double LUMatrix[], final int ipiv[]);

    /* Vector Mathematical Functions */
    void elementWiseAdd(final int dim, final double V1[], final double V2[], double result[]);

    void elementWiseSub(final int dim, final double V1[], final double V2[], double result[]);

    void elementWiseMul(final int dim, final double V1[], final double V2[], double result[]);

    void elementWiseInv(final int dim, final double V1[], double result[]);

    void elementWiseDiv(final int dim, final double V1[], final double V2[], double result[]);

    void elementWiseSqrt(final int dim, final double V1[], double result[]);

    void elementWiseInvSqrt(final int dim, final double V1[],  double result[]);

    void elementWiseExp(final int dim, final double V1[],  double result[]);

    void elementWiseLn(final int dim, final double V1[],  double result[]);

    void elementWiseCos(final int dim, final double V1[],  double result[]);

    void elementWiseSin(final int dim, final double V1[],  double result[]);

    void elementWiseTan(final int dim, final double V1[],  double result[]);
}
