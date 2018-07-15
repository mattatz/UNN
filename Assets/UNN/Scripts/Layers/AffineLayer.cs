using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;

namespace UNN
{

    public class AffineLayer : Layer
    {

        protected Signal weights, biases;

        protected int rows, columns;

        protected Signal x;
        protected Signal dW, dB;

        public AffineLayer(int rows, int columns, float weight_std = 0.01f) : base()
        {
            this.rows = rows;
            this.columns = columns;

            var weights = new float[rows, columns];
            var biases = new float[columns];

            for(int y = 0; y < rows; y++)
            {
                for(int x = 0; x < columns; x++)
                {
                    weights[y, x] = weight_std * Gaussian.SampleRandom();
                }
            }

            for (int x = 0; x < columns; x++)
            {
                biases[x] = 0f;
            }

            this.weights = new Signal(weights);
            this.biases = new Signal(biases);
        }

        public override Signal Forward(ComputeShader compute, Signal x)
        {
            this.x = x;

            var output = new Signal(x.Rows, weights.Columns);

            // matmul M = input * weights
            MatOperations.Multiply(compute, x, weights, output);

            // matplus M´ = M + biases
            MatOperations.Add(compute, output.Buffer, biases.Buffer, output.Rows, output.Columns, biases.Rows, biases.Columns);

            // output.Log();

            return output;
        }
        
        public Signal Backward(ComputeShader compute, Signal dout)
        {
            var dx = new Signal(dout.Rows, weights.Rows);
            MatOperations.MultiplyMT(compute, dout, weights, dx);
            // dx.Log();

            dW = Refresh(x.Columns, dout.Columns, dW);
            MatOperations.MultiplyTM(compute, x, dout, dW);

            dB = Refresh(1, dout.Columns, dB);
            MatOperations.Sum(compute, dout, dB);

            // dx.Log();
            // dw.Log();
            // db.Log();

            return dx;
        }

        public void Learn(ComputeShader compute, float rate)
        {
            var kernel = compute.FindKernel("Grad");

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
            if(weights != null)
            {
                weights.Dispose();
                biases.Dispose();
                weights = biases = null;
            }

            if(x != null)
            {
                x.Dispose();
                x = null;
            }

            if(dW != null)
            {
                dW.Dispose();
                dB.Dispose();
            }

        }

    }

}


