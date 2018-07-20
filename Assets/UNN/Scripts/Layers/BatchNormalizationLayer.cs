using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class BatchNormalizationLayer : Layer
    {

        [SerializeField] protected Signal gamma, beta;
        [SerializeField] protected float momentum;

        [SerializeField] protected int batchSize;

        protected Signal xc, xn, std;
        protected Signal runningMean, runningVar;

        public BatchNormalizationLayer(float momentum = 0.9f)
        {
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

            if(gamma == null)
            {
                gamma = new Signal(output);
                gamma.Init(1f);

                beta = new Signal(output);
                beta.Init(0f);
            }

            var kernel = compute.FindKernel("BNForward");
            compute.SetBuffer(kernel, "_Y", output.Buffer);
            compute.SetBuffer(kernel, "_Gamma", gamma.Buffer);
            compute.SetBuffer(kernel, "_Beta", beta.Buffer);
            Dispatch(compute, kernel, output.Rows, output.Columns);

            return output;
        }

        public Signal Backward(ComputeShader compute, Signal dout)
        {
            var dbeta = new Signal(1, dout.Columns);
            MatOperations.SumVM(compute, dbeta, dout);

            var dgamma = new Signal(1, dout.Columns);
            MatOperations.MulMM(compute, xn, dout);
            MatOperations.SumVM(compute, dgamma, xn);

            var dxn = new Signal(dout);
            MatOperations.MulMMM(compute, dout, gamma, dxn);

            var dxc = new Signal(dxn);
            MatOperations.CopyMM(compute, dxc, dxn);
            MatOperations.DivMV(compute, dxc, std);

            var dxn_x_xc = new Signal(dxn);
            MatOperations.MulMMM(compute, dxn, xc, dxn_x_xc);

            var std_x_std = new Signal(std);
            MatOperations.MulMMM(compute, std, std, std_x_std);

            var dxn_x_xc_div_std_x_std = new Signal(std);
            MatOperations.DivMVM(compute, dxn_x_xc, std_x_std, dxn_x_xc_div_std_x_std);

            var dstd = new Signal(std);
            MatOperations.SumVM(compute, dstd, dxn_x_xc_div_std_x_std);

            dbeta.Dispose();
            dgamma.Dispose();
            dxn.Dispose();
            dxn_x_xc.Dispose();
            std_x_std.Dispose();
            dxn_x_xc_div_std_x_std.Dispose();
            dstd.Dispose();

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
            if(gamma != null)
            {
                gamma.Dispose();
                beta.Dispose();

                gamma = beta = null;
            }

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


