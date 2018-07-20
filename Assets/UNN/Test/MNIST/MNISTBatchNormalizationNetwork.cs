using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN.Test
{

    public class MNISTBatchNormalizationNetwork : Network
    {

        [SerializeField] protected int inputSize, hiddenSize, outputSize;

        [SerializeField] protected AffineLayer affine1;
        [SerializeField] protected BatchNormalizationLayer bn1;
        [SerializeField] protected ReLULayer relu1;

        [SerializeField] protected AffineLayer affine2;
        [SerializeField] protected BatchNormalizationLayer bn2;
        [SerializeField] protected ReLULayer relu2;

        [SerializeField] protected AffineLayer affine3;
        [SerializeField] protected SoftmaxLayer softmax;

        [SerializeField] protected MomentumOptimizer optimizer;

        public MNISTBatchNormalizationNetwork(int inputSize, int hiddenSize, int outputSize) : base()
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;


            affine1 = new AffineLayer(inputSize, hiddenSize);
            bn1 = new BatchNormalizationLayer(hiddenSize, hiddenSize);
            relu1 = new ReLULayer();

            affine2 = new AffineLayer(hiddenSize, hiddenSize);
            bn2 = new BatchNormalizationLayer(hiddenSize, hiddenSize);
            relu2 = new ReLULayer();

            affine3 = new AffineLayer(hiddenSize, outputSize);
            softmax = new SoftmaxLayer();

            optimizer = new MomentumOptimizer(0.9f);
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
                affine1, bn1, relu1,
                affine2, bn2, relu2,
                affine3,
                softmax,
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

        public override void Learn(ComputeShader compute, float rate = 0.1F)
        {
            affine1.Learn(optimizer, compute, rate);
            bn1.Learn(optimizer, compute, rate);

            affine2.Learn(optimizer, compute, rate);
            bn2.Learn(optimizer, compute, rate);

            affine3.Learn(optimizer, compute, rate);
        }

        public override float Loss(ComputeShader compute, Signal signal, Signal answer)
        {
            var predictSig = Predict(compute, signal);

            var softmaxSig = softmax.Forward(compute, predictSig);
            predictSig.Dispose();

            return CrossEntropyError.Loss(compute, softmaxSig, answer);
        }

        public override Signal Predict(ComputeShader compute, Signal input)
        {
            var layers = new List<Layer>() {
                affine1, bn1, relu1,
                affine2, bn2, relu2,
                affine3,
            };

            layers.ForEach(layer =>
            {
                var tmp = input;
                input = layer.Forward(compute, tmp);
                tmp.Dispose();
            });

            return input;
        }

        public override void Dispose()
        {
            affine1.Dispose();
            bn1.Dispose();
            relu1.Dispose();

            affine2.Dispose();
            bn2.Dispose();
            relu2.Dispose();

            affine3.Dispose();

            softmax.Dispose();

            optimizer.Dispose();
        }


    }

}


