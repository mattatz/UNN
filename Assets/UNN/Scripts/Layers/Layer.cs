using System;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;

namespace UNN
{


    [System.Serializable]
    public abstract class Layer : IDisposable {

        public Layer()
        {
        }

        public abstract Signal Forward(ComputeShader compute, Signal x, bool train);
        public abstract Signal Backward(ComputeShader compute, Signal dout);

        protected Signal Refresh(Signal src, Signal dst)
        {
            return Refresh(src.Rows, src.Columns, dst);
        }

        protected Signal Refresh(int rows, int cols, Signal dst)
        {
            if(dst == null)
            {
                dst = new Signal(rows, cols);
            } else if(dst.Rows != rows || dst.Columns != cols)
            {
                dst.Dispose();
                dst = new Signal(rows, cols);
            }
            return dst;
        }

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


