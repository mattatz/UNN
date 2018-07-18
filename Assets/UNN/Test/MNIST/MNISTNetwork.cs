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

        public Signal Predict(ComputeShader compute, Signal input)
        {
            var s0 = affine1.Forward(compute, input);
            var s1 = relu.Forward(compute, s0);
            var s2 = affine2.Forward(compute, s1);

            s0.Dispose();

            return s2;
        }

        protected float Loss(ComputeShader compute, Signal signal, Signal answer)
        {
            var predictSig = Predict(compute, signal);

            var softmaxSig = softmax.Forward(compute, predictSig);
            predictSig.Dispose();

            return CrossEntropyError.Loss(compute, softmaxSig, answer);
        }

        public void Train(ComputeShader compute, DigitDataset dataset, int batchSize, float learningRate)
        {
            Signal input, answer;
            dataset.GetSubSignals(batchSize, out input, out answer);
            Gradient(compute, input, answer);
            Learn(compute, learningRate);

            input.Dispose();
            answer.Dispose();
        }

        public float Accuracy(ComputeShader compute, Signal input, Signal answer)
        {
            var output = Predict(compute, input);
            float acc = UNN.Accuracy.Calculate(compute, input, output, answer);
            output.Dispose();
            return acc;
        }

        protected void Gradient(ComputeShader compute, Signal input, Signal answer)
        {
            Loss(compute, input, answer);

            var softmaxSig = softmax.Backward(compute, answer);
            // softmaxSig.Log();

            var a2Sig = affine2.Backward(compute, softmaxSig);
            a2Sig = relu.Backward(compute, a2Sig);
            var a1Sig = affine1.Backward(compute, a2Sig);

            softmaxSig.Dispose();
            a2Sig.Dispose();
            a1Sig.Dispose();
        }

        protected void Learn(ComputeShader compute, float rate = 0.1f)
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


