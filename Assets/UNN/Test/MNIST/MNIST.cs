using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using Random = UnityEngine.Random;

namespace UNN.Test
{

    public class MNIST : MonoBehaviour {

        [SerializeField] protected ComputeShader compute;
        [SerializeField, Range(1000, 10000)] protected int iterations = 10000;
        [SerializeField, Range(32, 128)] protected int batchSize = 100;
        [SerializeField, Range(0.01f, 0.1f)] protected float learningRate = 0.1f;
        
        protected DigitDataset trainDataset, testDataset;
        protected List<Texture2D> images;

        protected Vector2 scrollPosition = Vector2.zero;

        AffineLayer affine1;
        ReLULayer relu;
        AffineLayer affine2;
        SoftmaxLayer softmax;

        protected int iter = 0;
        protected float accuracy = 0f;

        protected void Start () {
            var trainImagePath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "train-images.idx3-ubyte");
            var trainLabelPath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "train-labels.idx1-ubyte");
            var testImagePath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "t10k-images.idx3-ubyte");
            var testLabelPath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "t10k-labels.idx1-ubyte");

            trainDataset = Load(trainImagePath, trainLabelPath);
            testDataset = Load(testImagePath, testLabelPath);

            // images = dataset.Digits.Select(digit => digit.ToTexture(dataset.Rows, dataset.Columns)).ToList();

            var inputSize = trainDataset.Rows * trainDataset.Columns;
            var hiddenSize = 50;
            var outputSize = 10;

            affine1 = new AffineLayer(inputSize, hiddenSize);
            relu = new ReLULayer(hiddenSize, outputSize);
            affine2 = new AffineLayer(hiddenSize, outputSize);
            softmax = new SoftmaxLayer();
        }

        protected void Update()
        {
            if((++iter) <= iterations)
            {
                Train(trainDataset, batchSize, learningRate);

                if(iter % 100 == 0)
                {
                    Signal testInput, testAnswer;
                    testDataset.GetAllSignals(out testInput, out testAnswer);
                    Debug.Log(Accuracy(testInput, testAnswer));
                    testInput.Dispose();
                    testAnswer.Dispose();
                }
            }
        }

        public float[,] Evaluate(Signal input)
        {
            var output = Predict(input);
            float[,] result = output.GetData();
            output.Dispose();
            return result;
        }

        protected void Test()
        {
            float[,] A = new float[2, 5]
            {
                { 0, 1, 2, 1, 1 }, { 3, 4, 5, 1, 1 }
            };

            float[,] B = new float[2, 3]
            {
                { 0, 1, 2, }, { 3, 4, 5, }
            };

            var sigA = new Signal(A);
            var sigB = new Signal(B);
            var sigAB = new Signal(A.GetLength(1), B.GetLength(1));

            MatOperations.MultiplyTM(compute, sigA, sigB, sigAB);
            sigAB.Log();

            sigA.Dispose();
            sigB.Dispose();
            sigAB.Dispose();
        }

        protected Signal Predict(Signal input)
        {
            var s0 = affine1.Forward(compute, input);
            var s1 = relu.Forward(compute, s0);
            var s2 = affine2.Forward(compute, s1);

            s0.Dispose();

            return s2;
        }

        protected float Loss(Signal signal, Signal answer)
        {
            var predictSig = Predict(signal);

            var softmaxSig = softmax.Forward(compute, predictSig);
            predictSig.Dispose();

            return CrossEntropyError.Loss(compute, softmaxSig, answer);
        }

        protected void Train(DigitDataset dataset, int batchSize, float learningRate)
        {
            Signal input, answer;
            dataset.GetSubSignals(batchSize, out input, out answer);
            Gradient(input, answer);
            Learn(learningRate);

            input.Dispose();
            answer.Dispose();
        }

        protected float Accuracy(Signal input, Signal answer)
        {
            var output = Predict(input);
            float acc = UNN.Accuracy.Calculate(compute, input, output, answer);
            output.Dispose();
            return acc;
        }

        protected void Gradient(Signal input, Signal answer)
        {
            Loss(input, answer);

            var softmaxSig = softmax.Backward(compute, answer);
            // softmaxSig.Log();

            var a2Sig = affine2.Backward(compute, softmaxSig);
            a2Sig = relu.Backward(compute, a2Sig);
            var a1Sig = affine1.Backward(compute, a2Sig);

            softmaxSig.Dispose();
            a2Sig.Dispose();
            a1Sig.Dispose();
        }

        protected void Learn(float rate = 0.1f)
        {
            affine1.Learn(compute, rate);
            affine2.Learn(compute, rate);
        }

        protected DigitDataset Load(string imagePath, string labelPath, int limit = -1)
        {
            var labelsStream = new FileStream(labelPath, FileMode.Open);
            var imagesStream = new FileStream(imagePath, FileMode.Open);

            var labelsReader = new BinaryReader(labelsStream);
            var imagesReader = new BinaryReader(imagesStream);

#pragma warning disable 0219
            int magic1 = ReadBigInt32(imagesReader);
#pragma warning restore 0219

            int imagesCount = ReadBigInt32(imagesReader);
            int rows = ReadBigInt32(imagesReader);
            int cols = ReadBigInt32(imagesReader);

#pragma warning disable 0219
            int magic2 = ReadBigInt32(labelsReader);
#pragma warning restore 0219

            int labelsCount = ReadBigInt32(labelsReader);

            if(imagesCount != labelsCount)
            {
                Debug.LogWarning("the # of images and labels are not same.");
            }

            int count;
            if (limit <= 0)
            {
                count = imagesCount;
            } else
            {
                count = Mathf.Min(limit, imagesCount);
            }

            var digits = new Digit[count];
            float inv = 1f / 255f;

            for (int i = 0; i < count; i++)
            {
                float[] pixels = new float[rows * cols];
                for (int y = 0; y < rows; ++y)
                {
                    for (int x = 0; x < cols; ++x)
                    {
                        byte b = imagesReader.ReadByte();

                        // invert y direction
                        pixels[(rows - y - 1) * cols + x] = (int)b * inv;
                    }
                }

                byte label = labelsReader.ReadByte();

                var digit = new Digit(pixels, label);
                digits[i] = digit;
            } 

            imagesStream.Close();
            imagesReader.Close();
            labelsStream.Close();
            labelsReader.Close();

            return new DigitDataset(digits, rows, cols);
        }


        // https://stackoverflow.com/questions/49407772/reading-mnist-database
        protected int ReadBigInt32(BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        protected void OnDestroy()
        {
            affine1.Dispose();
            relu.Dispose();
            affine2.Dispose();
            softmax.Dispose();
        }

        protected void OnGUI()
        {
            GUI.Label(new Rect(30, 30, 180, 30), iter.ToString() + " / " + iterations.ToString());

            if (trainDataset == null || images == null) return;

            var cols = Mathf.CeilToInt(Screen.width / trainDataset.Columns);
            var n = trainDataset.Digits.Count();

            scrollPosition = GUI.BeginScrollView(new Rect(0, 0, Screen.width, Screen.height), scrollPosition, new Rect(0, 0, Screen.width, n / cols * trainDataset.Rows));

            for(int i = 0; i < n; i++)
            {
                int x = i % cols;
                int y = i / cols;
                GUI.DrawTexture(new Rect(x * trainDataset.Columns, y * trainDataset.Rows, trainDataset.Columns, trainDataset.Rows), images[i]);
            }

            GUI.EndScrollView();
        }

    }

}


