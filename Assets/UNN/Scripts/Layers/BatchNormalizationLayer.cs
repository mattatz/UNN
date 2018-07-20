using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public class BatchNormalizationLayer : TrainLayer
    {

        [SerializeField] protected Signal gamma, beta;
        protected Signal dgamma, dbeta;

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

            x.Log("x");
            mu.Log("mu");

            xc = Refresh(x, xc); // xc = x - mu
            MatOperations.SubMVM(compute, x, mu, xc);

            xc.Log("xc");

            var variance = new Signal(1, xc.Columns);
            MatOperations.VarianceMV(compute, xc, variance);

            variance.Log("variance");

            std = Refresh(variance, std);
            MatOperations.SqrtMM(compute, variance, std);

            std.Log("std");

            xn = Refresh(xc, xn); // xn = xc / std
            MatOperations.DivMVM(compute, xc, std, xn);

            xn.Log("xn");

            batchSize = x.Rows;

            Momentum(compute, mu, runningMean);
            Momentum(compute, variance, runningVar);

            var output = new Signal(xn);

            var kernel = compute.FindKernel("BNForward");
            compute.SetBuffer(kernel, "_X", xn.Buffer);
            compute.SetBuffer(kernel, "_Gamma", gamma.Buffer);
            compute.SetBuffer(kernel, "_Beta", beta.Buffer);
            compute.SetBuffer(kernel, "_Y", output.Buffer);
            Dispatch(compute, kernel, output.Rows, output.Columns);

            output.Log("output");

            mu.Dispose();
            variance.Dispose();

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
            MatOperations.CopyMM(compute, dxn, dxc);
            MatOperations.DivVM(compute, std, dxc);

            var dxn_x_xc = new Signal(dxn);
            MatOperations.MulMMM(compute, dxn, xc, dxn_x_xc);

            var std_x_std = new Signal(std);
            MatOperations.MulMMM(compute, std, std, std_x_std);

            var dxn_x_xc_div_std_x_std = new Signal(dxn_x_xc);
            MatOperations.DivMVM(compute, dxn_x_xc, std_x_std, dxn_x_xc_div_std_x_std);

            var dstd = new Signal(std);
            MatOperations.SumMV(compute, dxn_x_xc_div_std_x_std, dstd);

            var dvar = new Signal(dstd);
            DVar(compute, dstd, std, dvar);

            DXc(compute, xc, dvar, dxc, 2f / batchSize);

            var dmu = new Signal(1, dxc.Columns);
            MatOperations.SumMV(compute, dxc, dmu);

            var dx = new Signal(dout);
            DX(compute, dxc, dmu, dx, 1f / batchSize);

            dxn.Dispose();
            dxc.Dispose();
            dxn_x_xc.Dispose();
            std_x_std.Dispose();
            dxn_x_xc_div_std_x_std.Dispose();
            dstd.Dispose();
            dvar.Dispose();
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

                gamma = beta = null;
            }

            if(dgamma != null)
            {
                dgamma.Dispose();
                dbeta.Dispose();

                xc.Dispose();
                xn.Dispose();
                std.Dispose();
                runningMean.Dispose();
                runningVar.Dispose();

                dgamma = dbeta = null;
                xc = xn = std = runningMean = runningVar = null;
            }
        }

    }

}


