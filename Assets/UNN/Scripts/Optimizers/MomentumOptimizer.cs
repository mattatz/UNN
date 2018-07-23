using System.Linq;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    [System.Serializable]
    public class MomentumOptimizer : Optimizer
    {

        protected Dictionary<Signal, Signal> velocity;
        [SerializeField] protected float momentum;

        public MomentumOptimizer(float m = 0.9f) : base()
        {
            momentum = m;
        }
        
        public override void Update(ComputeShader compute, float rate, Signal gamma, Signal dGamma)
        {
            var kernel = compute.FindKernel("Momentum");

            compute.SetFloat("_LearningRate", rate);
            compute.SetFloat("_Momentum", momentum);

            Signal vGamma;
            if(velocity == null) velocity = new Dictionary<Signal, Signal>();
            if(!velocity.ContainsKey(gamma))
            {
                vGamma = new Signal(gamma);
                vGamma.Init(0f);
                velocity.Add(gamma, vGamma);
            } else
            {
                vGamma = velocity[gamma];
            }
            compute.SetBuffer(kernel, "_X", vGamma.Buffer);
            compute.SetBuffer(kernel, "_T", dGamma.Buffer);
            compute.SetBuffer(kernel, "_Y", gamma.Buffer);
            Dispatch(compute, kernel, gamma.Rows, gamma.Columns);
        }

        public override void Dispose()
        {
            if(velocity != null)
            {
                velocity.Values.ToList().ForEach(v => v.Dispose());
                velocity.Clear();
                velocity = null;
            }
        }

    }

}


