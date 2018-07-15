using System.Linq;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using Random = UnityEngine.Random;

namespace UNN.Test
{

    public class DigitDataset {

        public Digit[] Digits { get { return digits; } }
        public int Count { get { return digits.Length; } }
        public int Rows { get { return rows; } }
        public int Columns { get { return columns; } }

        protected Digit[] digits;
        protected int rows, columns;

        public DigitDataset(Digit[] digits, int rows, int columns)
        {
            this.digits = digits;
            this.rows = rows;
            this.columns = columns;
        }

        public void GetAllSignals(out Signal input, out Signal answer)
        {
            GetSignals(digits.ToList(), out input, out answer);
        }

        public void GetSubSignals(int batchSize, out Signal input, out Signal answer)
        {
            var indices = Enumerable.Range(0, digits.Length).ToList();
            var source = new List<Digit>();

            for(int iter = 0; iter < batchSize; iter++)
            {
                int idx = Random.Range(0, indices.Count);
                int i = indices[idx];
                indices.RemoveAt(idx);

                var digit = digits[i];
                source.Add(digit);
            }

            GetSignals(source, out input, out answer);
        }

        protected void GetSignals(List<Digit> source, out Signal input, out Signal answer)
        {
            int inputSize = rows * columns;
            const int outputSize = 10;

            float[,] pixels = new float[source.Count, inputSize];
            float[,] labels = new float[source.Count, outputSize];

            for(int y = 0; y < source.Count; y++)
            {
                var digit = source[y];
                for(int x = 0; x < inputSize; x++)
                {
                    pixels[y, x] = digit.Pixels[x];
                }

                var onehot = digit.GetOneHotLabel();
                for(int x = 0; x < outputSize; x++)
                {
                    labels[y, x] = onehot[x];
                }
            }

            input = new Signal(pixels);
            answer = new Signal(labels);
        }

    }

}


