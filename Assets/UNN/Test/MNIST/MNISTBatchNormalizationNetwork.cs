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

        [SerializeField] protected MomentumOptimizer optim1, optim2, optim3;

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

            optim1 = new MomentumOptimizer(affine1, 0.9f);
            optim2 = new MomentumOptimizer(affine2, 0.9f);
            optim3 = new MomentumOptimizer(affine2, 0.9f);
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

            var softmaxSig = softmax.Backward(compute, answer);
            // softmaxSig.Log();

            var a3Sig = affine3.Backward(compute, softmaxSig); softmaxSig.Dispose();
            
            var r2Sig = relu2.Backward(compute, a3Sig); a3Sig.Dispose();
            var bn2Sig = bn2.Backward(compute, r2Sig); r2Sig.Dispose();
            var a2Sig = affine2.Backward(compute, bn2Sig); bn2Sig.Dispose();

            var r1Sig = relu1.Backward(compute, a2Sig); a2Sig.Dispose();
            var bn1Sig = bn1.Backward(compute, r1Sig); r1Sig.Dispose();
            var a1Sig = affine1.Backward(compute, bn1Sig); bn1Sig.Dispose();

            a1Sig.Dispose();
        }

        public override void Learn(ComputeShader compute, float rate = 0.1F)
        {
            affine1.Learn(optim1, compute, rate);
            affine2.Learn(optim2, compute, rate);
            affine3.Learn(optim3, compute, rate);
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
            var s0 = affine1.Forward(compute, input);
            var s1 = bn1.Forward(compute, s0);
            var s2 = relu1.Forward(compute, s1);
            var s3 = affine2.Forward(compute, s2);

            s0.Dispose();
            s1.Dispose();
            s2.Dispose();

            return s3;
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

            optim1.Dispose();
            optim2.Dispose();
            optim3.Dispose();
        }


    }

}


