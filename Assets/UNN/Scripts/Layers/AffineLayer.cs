using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;

namespace UNN
{

    [System.Serializable]
    public class AffineLayer : InnerLayer
    {
        public Signal Weights { get { return weights; } }
        public Signal Biases { get { return biases; } }

        [SerializeField] protected Signal weights, biases;

        protected Signal x;
        protected Signal dW, dB;

        public AffineLayer(int rows, int columns, float weight_std = 0.01f) : base(rows, columns)
        {
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
            this.x = Refresh(x, this.x);
            MatOperations.CopyMM(compute, x, this.x);

            var output = new Signal(x.Rows, weights.Columns);

            // matmul M = input * weights
            MatOperations.Multiply(compute, x, weights, output);

            // matplus M´ = M + biases 
            MatOperations.AddVM(compute, biases, output);

            // output.Log();

            return output;
        }
        
        public override Signal Backward(ComputeShader compute, Signal dout)
        {
            var dx = new Signal(dout.Rows, weights.Rows);
            MatOperations.MultiplyMT(compute, dout, weights, dx);
            // dx.Log();

            dW = Refresh(x.Columns, dout.Columns, dW);
            MatOperations.MultiplyTM(compute, x, dout, dW);

            dB = Refresh(1, dout.Columns, dB);
            MatOperations.SumMV(compute, dout, dB);

            // dx.Log();
            // dw.Log();
            // db.Log();

            return dx;
        }

        public void Learn(Optimizer optimizer, ComputeShader compute, float rate)
        {
            optimizer.Update(compute, rate, weights, dW, biases, dB);
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


