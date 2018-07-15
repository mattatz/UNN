using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using Random = UnityEngine.Random;

namespace UNN
{

    public class Gaussian {

        public static float SampleRandom(float mean = 0f, float stddev = 1f)
        {
            var x1 = Random.value;
            var x2 = Random.value;
            var y1 = Mathf.Sqrt(-2.0f * Mathf.Log(x1)) * Mathf.Cos(2.0f * Mathf.PI * x2);
            return y1 * stddev + mean;
        }

    }

}


