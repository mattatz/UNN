using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;

namespace UNN.Test
{


    [System.Serializable]
    public class MNISTNetwork : Network
    {

        [SerializeField] protected int inputSize, hiddenSize, outputSize;

        [SerializeField] protected AffineLayer affine1;
        [SerializeField] protected ReLULayer relu;
        [SerializeField] protected AffineLayer affine2;
        [SerializeField] protected SoftmaxLayer softmax;

        [SerializeField] protected MomentumOptimizer optim1, optim2;

        public MNISTNetwork(int inputSize, int hiddenSize, int outputSize) : base()
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            affine1 = new AffineLayer(inputSize, hiddenSize);
            relu = new ReLULayer();
            affine2 = new AffineLayer(hiddenSize, outputSize);
            softmax = new SoftmaxLayer();

            optim1 = new MomentumOptimizer(affine1, 0.9f);
            optim2 = new MomentumOptimizer(affine2, 0.9f);
        }

        public override Signal Predict(ComputeShader compute, Signal input)
        {
            var layers = new List<Layer>() {
                affine1, relu, affine2,
            };

            layers.ForEach(layer =>
            {
                var tmp = input;
                input = layer.Forward(compute, input);
                tmp.Dispose();
            });

            return input;
        }

        public override float Loss(ComputeShader compute, Signal signal, Signal answer)
        {
            var predictSig = Predict(compute, signal);

            var softmaxSig = softmax.Forward(compute, predictSig);
            predictSig.Dispose();

            return CrossEntropyError.Loss(compute, softmaxSig, answer);
        }

        public override float Accuracy(ComputeShader compute, Signal input, Signal answer)
        {
            var output = Predict(compute, input);
            float acc = UNN.Accuracy.Calculate(compute, input, output, answer);
            output.Dispose();
            return acc;
        }

        public override void Gradient(ComputeShader compute, Signal input, Signal answer)
        {
            Loss(compute, input, answer);

            var layers = new List<Layer>() {
                affine1, relu, affine2, softmax
            };

            layers.Reverse();

            var signal = answer;
            layers.ForEach(layer =>
            {
                var tmp = signal;
                signal = layer.Backward(compute, tmp);
                tmp.Dispose();
            });
            signal.Dispose();
        }

        public override void Learn(ComputeShader compute, float rate = 0.1f)
        {
            affine1.Learn(optim1, compute, rate);
            affine2.Learn(optim2, compute, rate);
        }

        public override void Dispose()
        {
            affine1.Dispose();
            relu.Dispose();
            affine2.Dispose();
            softmax.Dispose();

            optim1.Dispose();
            optim2.Dispose();
        }

    }

}


