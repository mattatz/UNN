using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN.Test
{

    [System.Serializable]
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

            affine1 = new AffineLayer(inputSize, hiddenSize, Mathf.Sqrt(2.0f / inputSize));
            bn1 = new BatchNormalizationLayer(hiddenSize, hiddenSize);
            relu1 = new ReLULayer();

            affine2 = new AffineLayer(hiddenSize, hiddenSize, Mathf.Sqrt(2.0f / hiddenSize));
            bn2 = new BatchNormalizationLayer(hiddenSize, hiddenSize);
            relu2 = new ReLULayer();

            affine3 = new AffineLayer(hiddenSize, outputSize, Mathf.Sqrt(2.0f / hiddenSize));
            softmax = new SoftmaxLayer();

            optimizer = new MomentumOptimizer(0.9f);
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

                if(signal.IsNaN()) signal.Log(layer.GetType().ToString());
            });
            signal.Dispose();
        }

        public override void Learn(ComputeShader compute, float rate = 0.1f)
        {
            var layers = new List<TrainLayer>() {
                affine1, bn1,
                affine2, bn2,
                affine3,
            };
            layers.ForEach(trLayer => trLayer.Learn(optimizer, compute, rate));
        }

        public override float Loss(ComputeShader compute, Signal signal, Signal answer, bool train)
        {
            var predictSig = Predict(compute, signal, train);

            var softmaxSig = softmax.Forward(compute, predictSig, train);
            predictSig.Dispose();

            return CrossEntropyError.Loss(compute, softmaxSig, answer);
        }

        public override Signal Predict(ComputeShader compute, Signal input, bool train)
        {
            var layers = new List<Layer>() {
                affine1, bn1, relu1,
                affine2, bn2, relu2,
                affine3,
            };

            // Debug.Log("Predict");

            layers.ForEach(layer =>
            {
                // input.Log(layer.GetType().ToString());

                var tmp = input;
                input = layer.Forward(compute, tmp, train);
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

            affine1 = affine2 = null;
            bn1 = bn2 = null;
            relu1 = relu2 = null;

            softmax = null;
            optimizer = null;
        }

    }

}


