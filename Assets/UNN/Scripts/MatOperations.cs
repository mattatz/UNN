using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class MatOperations {

        public static void CopyMM(ComputeShader compute, Signal C, Signal B)
        {
            OpMM(compute, compute.FindKernel("CopyMM"), C, B);
        }

        public static void SqrtMM(ComputeShader compute, Signal C, Signal B)
        {
            OpMM(compute, compute.FindKernel("SqrtMM"), C, B);
        }

        public static void AddMM(ComputeShader compute, Signal C, Signal B)
        {
            OpMM(compute, compute.FindKernel("AddMM"), C, B);
        }

        public static void SubMM(ComputeShader compute, Signal C, Signal B)
        {
            OpMM(compute, compute.FindKernel("SubMM"), C, B);
        }

        public static void MulMM(ComputeShader compute, Signal C, Signal B)
        {
            OpMM(compute, compute.FindKernel("MulMM"), C, B);
        }


        public static void AddMV(ComputeShader compute, Signal C, Signal B)
        {
            OpMV(compute, compute.FindKernel("AddMV"), C, B);
        }

        public static void SubMV(ComputeShader compute, Signal C, Signal B)
        {
            OpMV(compute, compute.FindKernel("SubMV"), C, B);
        }

        public static void DivMV(ComputeShader compute, Signal C, Signal B)
        {
            OpMV(compute, compute.FindKernel("DivMV"), C, B);
        }

        public static void SumVM(ComputeShader compute, Signal C, Signal B)
        {
            OpVM(compute, compute.FindKernel("SumVM"), C, B);
        }

        public static void MeanVM(ComputeShader compute, Signal C, Signal B)
        {
            OpVM(compute, compute.FindKernel("MeanVM"), C, B);
        } 

        public static void VarianceVM(ComputeShader compute, Signal C, Signal B)
        {
            OpVM(compute, compute.FindKernel("VarianceVM"), C, B);
        } 

        public static void Multiply(ComputeShader compute, Signal A, Signal B, Signal C)
        {
            Multiply(
                compute, 
                A.Buffer, B.Buffer, C.Buffer, 
                A.Rows, A.Columns,
                B.Rows, B.Columns,
                C.Rows, C.Columns
            );
        }

        public static void MultiplyTM(ComputeShader compute, Signal A, Signal B, Signal C)
        {
            MultiplyTM(compute, A.Buffer, B.Buffer, C.Buffer, A.Rows, A.Columns, B.Rows, B.Columns, C.Rows, C.Columns);
        }

        public static void MultiplyMT(ComputeShader compute, Signal A, Signal B, Signal C)
        {
            MultiplyMT(compute, A.Buffer, B.Buffer, C.Buffer, A.Rows, A.Columns, B.Rows, B.Columns, C.Rows, C.Columns);
        }

        public static void Multiply(ComputeShader compute, ComputeBuffer A, ComputeBuffer B, ComputeBuffer C, int aRows, int aCols, int bRows, int bCols, int cRows, int cCols)
        {
            if(aCols != bRows)
            {
                Debug.LogWarning("A cols != B rows");
            }

            compute.SetInt("_ARows", aRows); compute.SetInt("_ACols", aCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            var kernel = compute.FindKernel("MatMul");
            compute.SetBuffer(kernel, "_A", A);
            compute.SetBuffer(kernel, "_B", B);
            compute.SetBuffer(kernel, "_C", C);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        public static void MultiplyTM(ComputeShader compute, ComputeBuffer A, ComputeBuffer B, ComputeBuffer C, int aRows, int aCols, int bRows, int bCols, int cRows, int cCols)
        {
            if(aRows != bRows)
            {
                Debug.LogWarning("A cols != B rows");
            }

            compute.SetInt("_ARows", aRows); compute.SetInt("_ACols", aCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            var kernel = compute.FindKernel("MatMulTM");
            compute.SetBuffer(kernel, "_A", A);
            compute.SetBuffer(kernel, "_B", B);
            compute.SetBuffer(kernel, "_C", C);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        public static void MultiplyMT(ComputeShader compute, ComputeBuffer A, ComputeBuffer B, ComputeBuffer C, int aRows, int aCols, int bRows, int bCols, int cRows, int cCols)
        {
            if(aCols != bCols)
            {
                Debug.LogWarning("A cols != B cols");
            }

            compute.SetInt("_ARows", aRows); compute.SetInt("_ACols", aCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            var kernel = compute.FindKernel("MatMulMT");
            compute.SetBuffer(kernel, "_A", A);
            compute.SetBuffer(kernel, "_B", B);
            compute.SetBuffer(kernel, "_C", C);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        protected static void OpMM(ComputeShader compute, int kernel, Signal C, Signal B)
        {
            int cRows = C.Rows, cCols = C.Columns;
            int bRows = B.Rows, bCols = B.Columns;

            if(cRows != bRows || cCols != bCols)
            {
                Debug.LogWarning("B & C does not have same dimensions.");
            }

            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);

            compute.SetBuffer(kernel, "_C", C.Buffer);
            compute.SetBuffer(kernel, "_B", B.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        protected static void OpMV(ComputeShader compute, int kernel, Signal C, Signal B)
        {
            int cRows = C.Rows, cCols = C.Columns;
            int bRows = B.Rows, bCols = B.Columns;

            if(bRows != 1)
            {
                Debug.LogWarning("B rows must be 1.");
            }

            if(cCols != bCols)
            {
                Debug.LogWarning("B & C does not same columns.");
            }

            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);

            compute.SetBuffer(kernel, "_C", C.Buffer);
            compute.SetBuffer(kernel, "_B", B.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        protected static void OpVM(ComputeShader compute, int kernel, Signal C, Signal B)
        {
            int cRows = C.Rows, cCols = C.Columns;
            int bRows = B.Rows, bCols = B.Columns;

            if(cRows != 1)
            {
                Debug.LogWarning("C rows must be 1.");
            }

            if(cCols != bCols)
            {
                Debug.LogWarning("B & C does not same columns.");
            }

            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);

            compute.SetBuffer(kernel, "_C", C.Buffer);
            compute.SetBuffer(kernel, "_B", B.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }



    }

}


