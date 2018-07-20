using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    [System.Serializable]
    public class SGDOptimizer : Optimizer {

        public SGDOptimizer() : base()
        {
        }

        public override void Update(ComputeShader compute, float rate, Signal gamma, Signal dGamma)
        {
            var kernel = compute.FindKernel("SGD");

            compute.SetFloat("_LearningRate", rate);
            compute.SetBuffer(kernel, "_X", dGamma.Buffer);
            compute.SetBuffer(kernel, "_Y", gamma.Buffer);
            Dispatch(compute, kernel, gamma.Rows, gamma.Columns);
        }

        public override void Dispose()
        {
            // do nothing
        }


    }

}


