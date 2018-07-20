using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class BatchNormalizationLayer : InnerLayer
    {

        [SerializeField] protected Signal gamma, beta;
        [SerializeField] protected float momentum;

        [SerializeField] protected int batchSize;

        protected Signal xc, xn, std;
        protected Signal runningMean, runningVar;

        public BatchNormalizationLayer(int rows, int columns, float momentum = 0.9f) : base(rows, columns)
        {
            this.momentum = momentum;

            gamma = new Signal(1, columns);
            gamma.Init(1f);

            beta = new Signal(1, columns);
            beta.Init(0f);

            runningMean = new Signal(1, columns);
            runningMean.Init(0f);

            runningVar = new Signal(1, columns);
            runningVar.Init(0f);
        }

        public override Signal Forward(ComputeShader compute, Signal x)
        {

            var mu = new Signal(1, x.Columns);
            MatOperations.MeanMV(compute, x, mu);

            xc = Refresh(x, xc); // xc = x - mu
            MatOperations.SubMVM(compute, x, mu, xc);

            var variance = new Signal(1, xc.Columns);
            MatOperations.VarianceMV(compute, xc, variance);

            std = Refresh(variance, std);
            MatOperations.SqrtMM(compute, variance, std);

            xn = Refresh(xc, xn); // xn = xc / std
            MatOperations.DivMVM(compute, xc, std, xn);

            batchSize = x.Rows;

            Momentum(compute, mu, runningMean);
            Momentum(compute, variance, runningVar);

            var output = new Signal(x);

            var kernel = compute.FindKernel("BNForward");
            compute.SetBuffer(kernel, "_Y", output.Buffer);
            compute.SetBuffer(kernel, "_Gamma", gamma.Buffer);
            compute.SetBuffer(kernel, "_Beta", beta.Buffer);
            Dispatch(compute, kernel, output.Rows, output.Columns);

            mu.Dispose();
            variance.Dispose();

            return output;
        }

        public override Signal Backward(ComputeShader compute, Signal dout)
        {
            var dbeta = new Signal(1, dout.Columns);
            MatOperations.SumMV(compute, dout, dbeta);

            var dgamma = new Signal(1, dout.Columns);
            MatOperations.MulMM(compute, dout, xn);
            MatOperations.SumMV(compute, xn, dgamma);

            var dxn = new Signal(dout);
            MatOperations.MulMMM(compute, dout, gamma, dxn);

            var dxc = new Signal(dxn);
            MatOperations.CopyMM(compute, dxn, dxc);
            MatOperations.DivVM(compute, std, dxc);

            var dxn_x_xc = new Signal(dxn);
            MatOperations.MulMMM(compute, dxn, xc, dxn_x_xc);

            var std_x_std = new Signal(std);
            MatOperations.MulMMM(compute, std, std, std_x_std);

            var dxn_x_xc_div_std_x_std = new Signal(std);
            MatOperations.DivMVM(compute, dxn_x_xc, std_x_std, dxn_x_xc_div_std_x_std);

            var dstd = new Signal(std);
            MatOperations.SumMV(compute, dxn_x_xc_div_std_x_std, dstd);

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

        protected void Momentum(ComputeShader compute, Signal X, Signal Y)
        {
            var kernel = compute.FindKernel("BNMomentum");
            compute.SetBuffer(kernel, "_X", X.Buffer);
            compute.SetBuffer(kernel, "_Y", Y.Buffer);
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


