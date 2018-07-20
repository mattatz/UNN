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
        [SerializeField, Range(100, 500)] protected int measure = 500;
        [SerializeField, Range(0.01f, 0.1f)] protected float learningRate = 0.1f;

        [SerializeField] protected string filename = "MNISTNetwork.json";
        [SerializeField] protected bool load = true;

        [SerializeField] protected MNISTInput input;
        
        protected DigitDataset trainDataset, testDataset;
        protected List<Texture2D> images;

        protected Vector2 scrollPosition = Vector2.zero;

        protected MNISTNetwork network;
        protected string path;

        [SerializeField] protected bool training;
        protected int iter = 0;
        protected float accuracy = 0f;
        protected int lastLabel;

        protected void Start() {
            var trainImagePath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "train-images.idx3-ubyte");
            var trainLabelPath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "train-labels.idx1-ubyte");
            var testImagePath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "t10k-images.idx3-ubyte");
            var testLabelPath = Path.Combine(Path.Combine(Application.streamingAssetsPath, "MNIST"), "t10k-labels.idx1-ubyte");

            trainDataset = LoadDataset(trainImagePath, trainLabelPath);
            testDataset = LoadDataset(testImagePath, testLabelPath);

            // images = trainDataset.Digits.Take(128).Select(digit => digit.ToTexture(trainDataset.Rows, trainDataset.Columns)).ToList();

            var inputSize = trainDataset.Rows * trainDataset.Columns;

            path = Path.Combine(Application.persistentDataPath, filename);
            if(load && File.Exists(path))
            {
                Debug.Log("load " + path);
                network = LoadNetwork();
                training = false;
            } else
            {
                network = new MNISTNetwork(inputSize, 50, 10);
            }

            /*
            Signal input, answer;
            trainDataset.GetSignals(new List<Digit>() { trainDataset.Digits[0] }, out input, out answer);
            input.LogMNIST();
            input.Dispose();
            answer.Dispose();
            */
        }

        protected void Update()
        {
            if(training && iter < iterations)
            {
                iter++;
                network.Train(compute, trainDataset, batchSize, learningRate);
                if(iter % measure == 0)
                {
                    Signal testInput, testAnswer;
                    testDataset.GetAllSignals(out testInput, out testAnswer);
                    accuracy = network.Accuracy(compute, testInput, testAnswer);
                    testInput.Dispose();
                    testAnswer.Dispose();
                }
            }
        }

        public void Evaluate(Signal input)
        {
            var output = network.Predict(compute, input);
            // var output = Predict(input);
            float[,] result = output.GetData();
            output.Dispose();

            // int rows = result.GetLength(0);
            int cols = result.GetLength(1);

            int label = 0;
            float max = 0f;
            for(int x = 0; x < cols; x++)
            {
                var v = result[0, x];
                if(v > max)
                {
                    max = v;
                    label = x;
                }
            }

            lastLabel = label;
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

        protected void SaveNetwork()
        {
            var json = JsonUtility.ToJson(network);
            File.WriteAllText(path, json);
        }

        protected MNISTNetwork LoadNetwork()
        {
            var json = File.ReadAllText(path);
            var network = JsonUtility.FromJson(json, typeof(MNISTNetwork)) as MNISTNetwork;
            return network;
        }

        protected DigitDataset LoadDataset(string imagePath, string labelPath, int limit = -1)
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
            network.Dispose();
        }

        protected void OnGUI()
        {
            if (trainDataset != null && images != null)
            {
                DrawMNISTView();
            }

            GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), input.Buffer);

            using(new GUILayout.HorizontalScope())
            {
                GUILayout.Space(20f);
                using(new GUILayout.VerticalScope())
                {
                    GUILayout.Space(20f);

                    training = GUILayout.Toggle(training, "training");
                    GUILayout.Label("iterations : " + iter.ToString() + " / " + iterations.ToString());

                    GUILayout.Label("accuracy : " + accuracy.ToString("0.00"));
                    GUILayout.Label("result : " + lastLabel);

                    GUILayout.Space(10f);
                    if(GUILayout.Button("Save"))
                    {
                        SaveNetwork();
                    }
                }
            }

        }

        protected void DrawMNISTView()
        {
            var n = images.Count;

            var cols = Mathf.CeilToInt(Screen.width / trainDataset.Columns);
            scrollPosition = GUI.BeginScrollView(new Rect(0, 0, Screen.width, Screen.height), scrollPosition, new Rect(0, 0, Screen.width, n / cols * trainDataset.Rows));
            for(int i = 0; i < n; i++)
            {
                int x = i % cols;
                int y = i / cols;
                GUI.DrawTexture(new Rect(x * trainDataset.Columns, y * trainDataset.Rows, trainDataset.Columns, trainDataset.Rows), images[i]);
                // GUI.Label(new Rect(x * trainDataset.Columns, y * trainDataset.Rows, trainDataset.Columns, trainDataset.Rows), trainDataset.Digits[i].Label.ToString());
            }

            GUI.EndScrollView();
        }

    }

}


