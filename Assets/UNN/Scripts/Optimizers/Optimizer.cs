using System;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;

namespace UNN
{

    public abstract class Optimizer : IDisposable {

        public Optimizer(AffineLayer layer) { }

        public abstract void Update(ComputeShader compute, float rate, Signal weights, Signal dW, Signal biases, Signal dB);

        protected void Dispatch(ComputeShader compute, int kernel, int rows, int columns)
        {
            uint tx, ty, tz;
            compute.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            compute.SetInt("_Rows", rows);
            compute.SetInt("_Cols", columns);
            compute.Dispatch(kernel, Mathf.FloorToInt(((int)columns - 1) / tx) + 1, Mathf.FloorToInt(((int)rows - 1) / ty) + 1, (int)tz);
        }

        public abstract void Dispose();

    }

}


