using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class MatOperations {

        public static void CopyMM(ComputeShader compute, Signal B, Signal C)
        {
            OpMM(compute, compute.FindKernel("CopyMM"), B, C);
        }

        public static void SqrtMM(ComputeShader compute, Signal B, Signal C)
        {
            OpMM(compute, compute.FindKernel("SqrtMM"), B, C);
        }

        public static void AddMM(ComputeShader compute, Signal B, Signal C)
        {
            OpMM(compute, compute.FindKernel("AddMM"), B, C);
        }

        public static void SubMM(ComputeShader compute, Signal B, Signal C)
        {
            OpMM(compute, compute.FindKernel("SubMM"), B, C);
        }

        public static void MulMM(ComputeShader compute, Signal B, Signal C)
        {
            OpMM(compute, compute.FindKernel("MulMM"), B, C);
        }

        public static void DivMM(ComputeShader compute, Signal B, Signal C)
        {
            OpMM(compute, compute.FindKernel("DivMM"), B, C);
        }

        public static void MulMMM(ComputeShader compute, Signal A, Signal B, Signal C)
        {
            OpMMM(compute, compute.FindKernel("MulMMM"), A, B, C);
        }

        public static void DivMMM(ComputeShader compute, Signal A, Signal B, Signal C)
        {
            OpMMM(compute, compute.FindKernel("DivMMM"), A, B, C);
        }

        public static void DivMVM(ComputeShader compute, Signal A, Signal B, Signal C)
        {
            OpMVM(compute, compute.FindKernel("DivMVM"), A, B, C);
        }

        public static void AddVM(ComputeShader compute, Signal B, Signal C)
        {
            OpVM(compute, compute.FindKernel("AddVM"), B, C);
        }

        public static void SubVM(ComputeShader compute, Signal B, Signal C)
        {
            OpVM(compute, compute.FindKernel("SubVM"), B, C);
        }

        public static void DivVM(ComputeShader compute, Signal B, Signal C)
        {
            OpVM(compute, compute.FindKernel("DivVM"), B, C);
        }

        public static void SumMV(ComputeShader compute, Signal B, Signal C)
        {
            OpMV(compute, compute.FindKernel("SumMV"), B, C);
        }

        public static void MeanMV(ComputeShader compute, Signal B, Signal C)
        {
            OpMV(compute, compute.FindKernel("MeanMV"), B, C);
        } 

        public static void VarianceMV(ComputeShader compute, Signal B, Signal C)
        {
            OpMV(compute, compute.FindKernel("VarianceMV"), B, C);
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

        protected static void OpMM(ComputeShader compute, int kernel, Signal B, Signal C)
        {
            int bRows = B.Rows, bCols = B.Columns;
            int cRows = C.Rows, cCols = C.Columns;

            if(cRows != bRows || cCols != bCols)
            {
                Debug.LogWarning("B & C does not have same dimensions.");
            }

            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            compute.SetBuffer(kernel, "_B", B.Buffer);
            compute.SetBuffer(kernel, "_C", C.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        protected static void OpMMM(ComputeShader compute, int kernel, Signal A, Signal B, Signal C)
        {
            int aRows = A.Rows, aCols = A.Columns;
            int bRows = B.Rows, bCols = B.Columns;
            int cRows = C.Rows, cCols = C.Columns;

            if(aRows != cRows || aCols != cCols)
            {
                Debug.LogWarning("A & C does not have same dimensions.");
            }

            if(cRows != bRows || cCols != bCols)
            {
                Debug.LogWarning("B & C does not have same dimensions.");
            }

            compute.SetInt("_ARows", aRows); compute.SetInt("_ACols", aCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            compute.SetBuffer(kernel, "_A", A.Buffer);
            compute.SetBuffer(kernel, "_B", B.Buffer);
            compute.SetBuffer(kernel, "_C", C.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        protected static void OpMVM(ComputeShader compute, int kernel, Signal A, Signal B, Signal C)
        {
            int aRows = A.Rows, aCols = A.Columns;
            int bRows = B.Rows, bCols = B.Columns;
            int cRows = C.Rows, cCols = C.Columns;

            if(aCols != bCols)
            {
                Debug.LogWarning("A & B does not have same columns.");
            }

            if(bRows != 1)
            {
                Debug.LogWarning("B rows must be 1.");
            }

            if(aRows != cRows || aCols != cCols)
            {
                Debug.LogWarning("A & C does not have same dimensions.");
            }

            compute.SetInt("_ARows", aRows); compute.SetInt("_ACols", aCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            compute.SetBuffer(kernel, "_A", A.Buffer);
            compute.SetBuffer(kernel, "_B", B.Buffer);
            compute.SetBuffer(kernel, "_C", C.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }


        protected static void OpVM(ComputeShader compute, int kernel, Signal B, Signal C)
        {
            int bRows = B.Rows, bCols = B.Columns;
            int cRows = C.Rows, cCols = C.Columns;

            if(bRows != 1)
            {
                Debug.LogWarning("B rows must be 1. (B must be vector)");
            }

            if(cCols != bCols)
            {
                Debug.LogWarning("B & C does not same columns.");
            }

            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            compute.SetBuffer(kernel, "_B", B.Buffer);
            compute.SetBuffer(kernel, "_C", C.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        protected static void OpMV(ComputeShader compute, int kernel, Signal B, Signal C)
        {
            int bRows = B.Rows, bCols = B.Columns;
            int cRows = C.Rows, cCols = C.Columns;

            if(cRows != 1)
            {
                Debug.LogWarning("C rows must be 1.");
            }

            if(cCols != bCols)
            {
                Debug.LogWarning("B & C does not same columns.");
            }

            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);

            compute.SetBuffer(kernel, "_B", B.Buffer);
            compute.SetBuffer(kernel, "_C", C.Buffer);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }



    }

}


