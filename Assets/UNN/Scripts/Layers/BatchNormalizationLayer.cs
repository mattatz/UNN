using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class BatchNormalizationLayer : Layer
    {

        [SerializeField] protected float gamma, beta, momentum;

        [SerializeField] protected int batchSize;

        protected Signal xc, xn, std;
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
            MatOperations.MeanVM(compute, x, mu);

            xc = Refresh(x, xc); // xc = x - mu
            MatOperations.CopyMM(compute, xc, x);
            MatOperations.SubMV(compute, xc, mu);

            var variance = new Signal(1, xc.Columns);
            MatOperations.VarianceVM(compute, xc, variance);

            std = Refresh(variance, std);
            MatOperations.SqrtMM(compute, std, variance);

            xn = Refresh(xc, xn);
            MatOperations.CopyMM(compute, xn, xc);
            MatOperations.DivMV(compute, xn, std);

            batchSize = x.Rows;

            Momentum(compute, runningMean, mu);
            Momentum(compute, runningVar, variance);

            mu.Dispose();
            variance.Dispose();

            var output = new Signal(x);

            var kernel = compute.FindKernel("BNForward");
            compute.SetBuffer(kernel, "_Y", output.Buffer);
            compute.SetFloat("_Gamma", gamma);
            compute.SetFloat("_Beta", beta);
            Dispatch(compute, kernel, output.Rows, output.Columns);

            return output;
        }

        public Signal Backward(ComputeShader compute, Signal dout)
        {
            var dbeta = new Signal(1, dout.Columns);
            var dgamma = new Signal(1, dout.Columns);

            MatOperations.MulMM(compute, xn, dout);
            MatOperations.SumVM(compute, dgamma, xn);


            dbeta.Dispose();
            dgamma.Dispose();

            var dx = new Signal(dout);

            return dx;
        }

        protected void Momentum(ComputeShader compute, Signal Y, Signal X)
        {
            var kernel = compute.FindKernel("BNMomentum");
            compute.SetBuffer(kernel, "_Y", Y.Buffer);
            compute.SetBuffer(kernel, "_X", X.Buffer);
            compute.SetFloat("_Momentum", momentum);
            Dispatch(compute, kernel, Y.Rows, Y.Columns);
        }

        public override void Dispose()
        {
            if(xc != null)
            {
                xc.Dispose();
                xn.Dispose();
                std.Dispose();
                runningMean.Dispose();
                runningVar.Dispose();

                xc = xn = std = runningMean = runningVar = null;
            }
        }


    }

}


