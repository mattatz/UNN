using System.IO;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;

namespace UNN.Test
{

    public class MNISTBatchNormalizationTest : MNISTTest {

        protected override void SetupNetwork()
        {
            var inputSize = trainDataset.Rows * trainDataset.Columns;

            path = Path.Combine(Application.persistentDataPath, filename);
            if(load && File.Exists(path))
            {
                network = LoadMNISTBatchNormalizationNetwork();
                if(network == null) {
                    network = new MNISTBatchNormalizationNetwork(inputSize, 100, 10);
                } else
                {
                    Debug.Log("load " + path);
                    training = false;
                    Measure();
                }
            } else
            {
                network = new MNISTBatchNormalizationNetwork(inputSize, 100, 10);
            }
        }

        protected MNISTBatchNormalizationNetwork LoadMNISTBatchNormalizationNetwork()
        {
            var json = File.ReadAllText(path);
            return JsonUtility.FromJson(json, typeof(MNISTBatchNormalizationNetwork)) as MNISTBatchNormalizationNetwork;
        }

        protected override void Update()
        {
            if(training && iter < iterations)
            // if(training && iter < 1)
            {
                iter++;
                Signal input, answer;
                trainDataset.GetSubSignals(batchSize, out input, out answer);
                network.Gradient(compute, input, answer);
                network.Learn(compute, learningRate);

                input.Dispose();
                answer.Dispose();
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



    }

}


