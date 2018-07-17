using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class MomentumOptimizer : Optimizer
    {

        protected Signal vW, vB;
        protected float momentum;

        public MomentumOptimizer(AffineLayer layer, float momentum = 0.9f) : base(layer)
        {
            this.momentum = momentum;
            vW = new Signal(layer.Weights);
            vW.Init(0f);

            vB = new Signal(layer.Biases);
            vB.Init(0f);
        }

        public override void Update(ComputeShader compute, float rate, Signal weights, Signal dW, Signal biases, Signal dB)
        {
            var kernel = compute.FindKernel("Momentum");

            compute.SetFloat("_LearningRate", rate);
            compute.SetFloat("_Momentum", momentum);

            compute.SetBuffer(kernel, "_T", dW.Buffer);
            compute.SetBuffer(kernel, "_X", vW.Buffer);
            compute.SetBuffer(kernel, "_Y", weights.Buffer);
            Dispatch(compute, kernel, weights.Rows, weights.Columns);

            compute.SetBuffer(kernel, "_T", dB.Buffer);
            compute.SetBuffer(kernel, "_X", vB.Buffer);
            compute.SetBuffer(kernel, "_Y", biases.Buffer);
            Dispatch(compute, kernel, biases.Rows, biases.Columns);
        }

        public override void Dispose()
        {
            if(vW != null)
            {
                vW.Dispose();
                vB.Dispose();
                vW = vB = null;
            }
        }

    }

}


