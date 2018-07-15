using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class CrossEntropyError {

        public static float Loss(ComputeShader compute, Signal input, Signal answer)
        {
            int batchSize = input.Rows;

            Signal temp = new Signal(answer.Rows, answer.Columns);

            var kernel = compute.FindKernel("Log");
            int rows = temp.Rows, columns = temp.Columns;

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.SetInt("_Rows", rows);
            compute.SetInt("_Cols", columns);
            compute.SetBuffer(kernel, "_X", input.Buffer);
            compute.SetBuffer(kernel, "_T", answer.Buffer);
            compute.SetBuffer(kernel, "_Y", temp.Buffer);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)columns - 1) / tx) + 1, Mathf.FloorToInt(((int)rows - 1) / ty) + 1, (int)tz);

            float[,] log = temp.GetData();

            float sum = 0f;
            for(int y = 0; y < rows; y++)
            {
                for(int x = 0; x < columns; x++)
                {
                    sum += log[y, x];
                }
            }

            temp.Dispose();

            return - sum / batchSize;
        }

    }

}


