using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    [System.Serializable]
    public class BatchNormalizationLayer : TrainLayer
    {

        [SerializeField] protected Signal gamma, beta;
        protected Signal dgamma, dbeta;

        [SerializeField] protected float momentum;

        [SerializeField] protected int batchSize;

        protected Signal xc, xn, std;
        [SerializeField] protected Signal runningMean, runningVar;

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

        public override Signal Forward(ComputeShader compute, Signal x, bool train)
        {
            if(train)
            {
                var mu = new Signal(1, x.Columns);
                MatOperations.MeanMV(compute, x, mu);
                // x.Log("x");
                // mu.Log("mu");

                xc = Refresh(x, xc); // xc = x - mu
                MatOperations.SubMVM(compute, x, mu, xc);
                // xc.Log("xc");

                var variance = new Signal(1, xc.Columns);
                MatOperations.VarianceMV(compute, xc, variance);
                // variance.Log("variance");

                std = Refresh(variance, std);
                MatOperations.SqrtMM(compute, variance, std);
                // std.Log("std");

                xn = Refresh(xc, xn); // xn = xc / std
                MatOperations.DivMVM(compute, xc, std, xn);

                batchSize = x.Rows;

                Momentum(compute, mu, runningMean);
                Momentum(compute, variance, runningVar);

                mu.Dispose();
                variance.Dispose();
            } else
            {
                xc = Refresh(x, xc); // xc = x - runningMean
                MatOperations.SubMVM(compute, x, runningMean, xc);
                // x.Log("x");
                // runningMean.Log("runningMean");
                // xn.Log("xn");

                xn = Refresh(xc, xn); // xn = xc / sqrt(runningVar + epsilon)
                Xn(compute, xc, runningVar, xn);
            }

            var output = new Signal(xn);

            var kernel = compute.FindKernel("BNForward");
            compute.SetBuffer(kernel, "_X", xn.Buffer);
            compute.SetBuffer(kernel, "_Gamma", gamma.Buffer);
            compute.SetBuffer(kernel, "_Beta", beta.Buffer);
            compute.SetBuffer(kernel, "_Y", output.Buffer);
            Dispatch(compute, kernel, output.Rows, output.Columns);

            return output;
        }

        public override Signal Backward(ComputeShader compute, Signal dout)
        {
            dbeta = Refresh(1, dout.Columns, dbeta);
            MatOperations.SumMV(compute, dout, dbeta);

            dgamma = Refresh(1, dout.Columns, dgamma);
            MatOperations.MulMM(compute, dout, xn);
            MatOperations.SumMV(compute, xn, dgamma);

            var dxn = new Signal(dout);
            MatOperations.MulMVM(compute, dout, gamma, dxn);

            var dxc = new Signal(dxn);
            MatOperations.DivMVM(compute, dxn, std, dxc);

            var dxn_x_xc = new Signal(dxn);
            MatOperations.MulMMM(compute, dxn, xc, dxn_x_xc);
            dxn.Dispose();

            var std_x_std = new Signal(std);
            MatOperations.MulMMM(compute, std, std, std_x_std);

            var dxn_x_xc_div_std_x_std = new Signal(dxn_x_xc);
            MatOperations.DivMVM(compute, dxn_x_xc, std_x_std, dxn_x_xc_div_std_x_std);
            dxn_x_xc.Dispose();
            std_x_std.Dispose();

            var dstd = new Signal(std);
            MatOperations.SumMV(compute, dxn_x_xc_div_std_x_std, dstd);
            dxn_x_xc_div_std_x_std.Dispose();

            var dvar = new Signal(dstd);
            DVar(compute, dstd, std, dvar);
            dstd.Dispose();

            DXc(compute, xc, dvar, dxc, 2f / batchSize);
            dvar.Dispose();

            var dmu = new Signal(1, dxc.Columns);
            MatOperations.SumMV(compute, dxc, dmu);

            var dx = new Signal(dout);
            DX(compute, dxc, dmu, dx, 1f / batchSize);
            dxc.Dispose();
            dmu.Dispose();

            return dx;
        }

        public override void Learn(Optimizer optimizer, ComputeShader compute, float rate)
        {
            optimizer.Update(compute, rate, gamma, dgamma);
            optimizer.Update(compute, rate, beta, dbeta);
        }

        protected void Momentum(ComputeShader compute, Signal X, Signal Y)
        {
            var kernel = compute.FindKernel("BNMomentum");
            compute.SetBuffer(kernel, "_X", X.Buffer);
            compute.SetBuffer(kernel, "_Y", Y.Buffer);
            compute.SetFloat("_Momentum", momentum);
            Dispatch(compute, kernel, Y.Rows, Y.Columns);
        }

        protected void Xn(ComputeShader compute, Signal X, Signal T, Signal Y)
        {
            var kernel = compute.FindKernel("BNXn");
            compute.SetBuffer(kernel, "_X", X.Buffer);
            compute.SetBuffer(kernel, "_T", T.Buffer);
            compute.SetBuffer(kernel, "_Y", Y.Buffer);
            Dispatch(compute, kernel, Y.Rows, Y.Columns);
        }

        protected void DVar(ComputeShader compute, Signal X, Signal T, Signal Y)
        {
            var kernel = compute.FindKernel("BNDVar");
            compute.SetBuffer(kernel, "_X", X.Buffer);
            compute.SetBuffer(kernel, "_T", T.Buffer);
            compute.SetBuffer(kernel, "_Y", Y.Buffer);
            Dispatch(compute, kernel, Y.Rows, Y.Columns);
        }

        protected void DXc(ComputeShader compute, Signal X, Signal T, Signal Y, float sigma)
        {
            var kernel = compute.FindKernel("BNDXc");
            compute.SetBuffer(kernel, "_X", X.Buffer);
            compute.SetBuffer(kernel, "_T", T.Buffer);
            compute.SetBuffer(kernel, "_Y", Y.Buffer);
            compute.SetFloat("_Sigma", sigma);
            Dispatch(compute, kernel, Y.Rows, Y.Columns);
        }

        protected void DX(ComputeShader compute, Signal X, Signal T, Signal Y, float sigma)
        {
            var kernel = compute.FindKernel("BNDX");
            compute.SetBuffer(kernel, "_X", X.Buffer);
            compute.SetBuffer(kernel, "_T", T.Buffer);
            compute.SetBuffer(kernel, "_Y", Y.Buffer);
            compute.SetFloat("_Sigma", sigma);
            Dispatch(compute, kernel, Y.Rows, Y.Columns);
        }

        public override void Dispose()
        {
            if(gamma != null)
            {
                gamma.Dispose();
                beta.Dispose();
            }

            if (dgamma != null)
            {
                dgamma.Dispose();
                dbeta.Dispose();
            }

            if(xc != null) { 
                xc.Dispose();
                xn.Dispose();
            }

            if(std != null)
            {
                std.Dispose();
            }

            if(runningMean != null)
            {
                runningMean.Dispose();
                runningVar.Dispose();
            }

            dgamma = dbeta = null;
            gamma = beta = null;
            xc = xn = std = runningMean = runningVar = null;
        }

    }

}


