using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    [System.Serializable]
    public class SGDOptimizer : Optimizer {

        public SGDOptimizer(AffineLayer layer) : base(layer)
        {
        }

        public override void Update(ComputeShader compute, float rate, Signal weights, Signal dW, Signal biases, Signal dB)
        {
            var kernel = compute.FindKernel("SGD");

            compute.SetFloat("_LearningRate", rate);
            compute.SetBuffer(kernel, "_X", dW.Buffer);
            compute.SetBuffer(kernel, "_Y", weights.Buffer);
            Dispatch(compute, kernel, weights.Rows, weights.Columns);

            compute.SetBuffer(kernel, "_X", dB.Buffer);
            compute.SetBuffer(kernel, "_Y", biases.Buffer);
            Dispatch(compute, kernel, biases.Rows, biases.Columns);
        }

        public override void Dispose()
        {
            // do nothing
        }

    }

}


