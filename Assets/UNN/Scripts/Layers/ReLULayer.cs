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

        public override Signal Forward(ComputeShader compute, Signal x, bool train)
        {
            mask = Refresh(x, mask);

            var kernel = compute.FindKernel("ReLU");
            compute.SetBuffer(kernel, "_X", x.Buffer);
            compute.SetBuffer(kernel, "_Y", mask.Buffer);
            Dispatch(compute, kernel, mask.Rows, mask.Columns);

            var output = new Signal(mask);
            MatOperations.CopyMM(compute, mask, output);
            return output;
        }

        public override Signal Backward(ComputeShader compute, Signal dout)
        {
            var output = new Signal(dout);

            var kernel = compute.FindKernel("ReLUBackward");
            compute.SetBuffer(kernel, "_X", dout.Buffer);
            compute.SetBuffer(kernel, "_T", mask.Buffer);
            compute.SetBuffer(kernel, "_Y", output.Buffer);
            Dispatch(compute, kernel, output.Rows, output.Columns);

            // mask.Log();
            // dout.Log();

            return output;
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


