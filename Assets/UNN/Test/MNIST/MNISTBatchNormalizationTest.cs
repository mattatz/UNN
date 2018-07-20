using System.IO;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;

namespace UNN.Test
{

    public class MNISTBatchNormalizationTest : MNIST {

        protected override void SetupNetwork()
        {
            var inputSize = trainDataset.Rows * trainDataset.Columns;

            /*
            path = Path.Combine(Application.persistentDataPath, filename);
            if(load && File.Exists(path))
            {
                Debug.Log("load " + path);
                network = LoadNetwork();
                training = false;
            } else
            {
            }
            */

            network = new MNISTBatchNormalizationNetwork(inputSize, 100, 10);
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


