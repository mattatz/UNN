using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    [System.Serializable]
    public class SoftmaxLayer : OutputLayer {

        protected Signal y;

        public SoftmaxLayer() : base()
        {
        }

        public override Signal Forward(ComputeShader compute, Signal x, bool train)
        {
            var temp = new Signal(x.Rows, x.Columns);

            var kernel = compute.FindKernel("Exp");
            compute.SetBuffer(kernel, "_X", x.Buffer);
            compute.SetBuffer(kernel, "_Y", temp.Buffer);
            Dispatch(compute, kernel, temp.Rows, temp.Columns);

            y = Refresh(x, y);
            kernel = compute.FindKernel("Softmax");
            compute.SetBuffer(kernel, "_X", temp.Buffer);
            compute.SetBuffer(kernel, "_Y", y.Buffer);
            Dispatch(compute, kernel, y.Rows, y.Columns);

            temp.Dispose();

            return y;
        }

        public override Signal Backward(ComputeShader compute, Signal answer)
        {
            int batchSize = answer.Rows;
            var output = new Signal(answer.Rows, answer.Columns);
            var kernel = compute.FindKernel("SoftmaxBackward");
            compute.SetBuffer(kernel, "_X", y.Buffer);
            compute.SetBuffer(kernel, "_T", answer.Buffer);
            compute.SetBuffer(kernel, "_Y", output.Buffer);
            compute.SetInt("_BatchSize", batchSize);
            Dispatch(compute, kernel, output.Rows, output.Columns);

            return output;
        }

        public override void Dispose()
        {
            if(y != null)
            {
                y.Dispose();
                y = null;
            }
        }

    }

}

