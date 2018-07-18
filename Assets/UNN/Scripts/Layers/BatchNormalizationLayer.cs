using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class BatchNormalizationLayer : Layer
    {

        [SerializeField] protected float gamma, beta, momentum;

        protected Signal runningMean, runningVar;

        public BatchNormalizationLayer(float gamma, float beta, float momentum = 0.9f)
        {
            this.gamma = gamma;
            this.beta = beta;
            this.momentum = momentum;
        }

        public override Signal Forward(ComputeShader compute, Signal x)
        {

            if(runningMean == null)
            {
                runningMean = new Signal(1, x.Columns);
                runningMean.Init(0f);

                runningVar = new Signal(1, x.Columns);
                runningVar.Init(0f);
            }

            var mu = new Signal(1, x.Columns);
            Mean(compute, x, mu);

            var xc = new Signal(x); // xc = x - mu
            MatOperations.Copy(compute, xc, x);
            MatOperations.SubMV(compute, xc, mu);

            var variance = new Signal(1, xc.Columns);
            Variance(compute, xc, variance);

            var std = new Signal(variance);
            MatOperations.Sqrt(compute, std, variance);

            var output = new Signal(x);

            return output;
        }

        protected void Mean(ComputeShader compute, Signal x, Signal y)
        {
            if(y.Rows != 1)
            {
                Debug.LogWarning("y.Rows is not 1.");
            }

            if(x.Columns != y.Columns)
            {
                Debug.LogWarning("x.Columns is not equal to y.Columns.");
            }

            var meanKer = compute.FindKernel("Mean");
            compute.SetBuffer(meanKer, "_X", x.Buffer);
            compute.SetBuffer(meanKer, "_Y", y.Buffer);
            Dispatch(compute, meanKer, x.Rows, x.Columns);
        }

        protected void Variance(ComputeShader compute, Signal x, Signal y)
        {
            if(y.Rows != 1)
            {
                Debug.LogWarning("y.Rows is not 1.");
            }

            if(x.Columns != y.Columns)
            {
                Debug.LogWarning("x.Columns is not equal to y.Columns.");
            }

            var meanKer = compute.FindKernel("Mean");
            compute.SetBuffer(meanKer, "_X", x.Buffer);
            compute.SetBuffer(meanKer, "_Y", y.Buffer);
            Dispatch(compute, meanKer, x.Rows, x.Columns);
        }

        public override void Dispose()
        {
        }


    }

}


