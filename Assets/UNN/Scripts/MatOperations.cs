using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class MatOperations {

        public static void Add(ComputeShader compute, ComputeBuffer C, ComputeBuffer B, int cRows, int cCols, int bRows, int bCols)
        {
            compute.SetInt("_CRows", cRows); compute.SetInt("_CCols", cCols);
            compute.SetInt("_BRows", bRows); compute.SetInt("_BCols", bCols);

            var kernel = compute.FindKernel("MatAdd");
            compute.SetBuffer(kernel, "_C", C);
            compute.SetBuffer(kernel, "_B", B);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);
        }

        public static void Sum(ComputeShader compute, Signal X, Signal Y)
        {
            Sum(compute, X.Buffer, Y.Buffer, X.Rows, X.Columns, Y.Rows, Y.Columns);
        }

        public static void Sum(ComputeShader compute, ComputeBuffer X, ComputeBuffer Y, int xRows, int xCols, int yRows, int yCols)
        {
            if(xCols != yCols)
            {
                Debug.Log("X cols != Y cols");
            }

            if(yRows > 1)
            {
                Debug.Log("Y rows must be one.");
            }

            compute.SetInt("_Rows", xRows); compute.SetInt("_Cols", xCols);

            var kernel = compute.FindKernel("Sum");
            compute.SetBuffer(kernel, "_X", X);
            compute.SetBuffer(kernel, "_Y", Y);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)yCols - 1) / tx) + 1, (int)ty, (int)tz);
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



    }

}


