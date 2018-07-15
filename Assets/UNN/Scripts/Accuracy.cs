using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class Accuracy {


        public static float Calculate(ComputeShader compute, Signal input, Signal output, Signal answer)
        {
            var batchSize = input.Rows;

            var tmp = new Signal(batchSize, 1);

            var kernel = compute.FindKernel("Accuracy");
            compute.SetBuffer(kernel, "_X", output.Buffer);
            compute.SetBuffer(kernel, "_T", answer.Buffer);
            compute.SetBuffer(kernel, "_Y", tmp.Buffer);
            compute.SetInt("_Rows", batchSize);
            compute.SetInt("_Cols", answer.Columns);

            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)batchSize - 1) / tx) + 1, (int)ty, (int)tz);

            var data = tmp.GetData();
            float acc = 0f;
            for(int x = 0; x < batchSize; x++)
            {
                acc += data[x, 0];
            }

            // tmp.Log();
            // Debug.Log(acc);

            tmp.Dispose();

            return acc / batchSize;
        }

    }

}


