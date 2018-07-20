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

        [SerializeField] protected MomentumOptimizer optimizer;

        public MNISTNetwork(int inputSize, int hiddenSize, int outputSize) : base()
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            affine1 = new AffineLayer(inputSize, hiddenSize);
            relu = new ReLULayer();
            affine2 = new AffineLayer(hiddenSize, outputSize);
            softmax = new SoftmaxLayer();

            optimizer = new MomentumOptimizer(0.9f);
        }

        public override Signal Predict(ComputeShader compute, Signal input, bool train)
        {
            var layers = new List<Layer>() {
                affine1, relu, affine2,
            };

            layers.ForEach(layer =>
            {
                var tmp = input;
                input = layer.Forward(compute, tmp, train);
                tmp.Dispose();
            });

            return input;
        }

        public override float Loss(ComputeShader compute, Signal signal, Signal answer, bool train)
        {
            var predictSig = Predict(compute, signal, train);

            var softmaxSig = softmax.Forward(compute, predictSig, train);
            predictSig.Dispose();

            return CrossEntropyError.Loss(compute, softmaxSig, answer);
        }

        public override float Accuracy(ComputeShader compute, Signal input, Signal answer)
        {
            var output = Predict(compute, input, false);
            float acc = UNN.Accuracy.Calculate(compute, input, output, answer);
            output.Dispose();
            return acc;
        }

        public override void Gradient(ComputeShader compute, Signal input, Signal answer)
        {
            Loss(compute, input, answer, true);

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
            affine1.Learn(optimizer, compute, rate);
            affine2.Learn(optimizer, compute, rate);
        }

        public override void Dispose()
        {
            affine1.Dispose();
            relu.Dispose();
            affine2.Dispose();
            softmax.Dispose();

            optimizer.Dispose();
        }

    }

}


