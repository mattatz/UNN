using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;

namespace UNN
{


    [System.Serializable]
    public class ReLULayer : Layer
    {

        protected Signal mask;

        public ReLULayer() : base()
        {
        }

        public override Signal Forward(ComputeShader compute, Signal x)
        {
            mask = Refresh(x, mask);

            var kernel = compute.FindKernel("ReLU");
            compute.SetBuffer(kernel, "_X", x.Buffer);
            compute.SetBuffer(kernel, "_Y", mask.Buffer);
            Dispatch(compute, kernel, mask.Rows, mask.Columns);

            return mask;
        }

        public Signal Backward(ComputeShader compute, Signal dout)
        {
            var kernel = compute.FindKernel("ReLUBackward");
            compute.SetBuffer(kernel, "_X", mask.Buffer);
            compute.SetBuffer(kernel, "_Y", dout.Buffer);
            Dispatch(compute, kernel, dout.Rows, dout.Columns);

            // mask.Log();
            // dout.Log();

            return dout;
        }

        public override void Dispose()
        {
            if(mask != null)
            {
                mask.Dispose();
                mask = null;
            }
        }

    }

}


