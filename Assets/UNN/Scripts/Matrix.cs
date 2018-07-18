using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    [System.Serializable]
    public class Matrix {

        [SerializeField] protected float[] column;

        public Matrix(int count)
        {
            column = new float[count];
        }

        public float GetValue(int index)
        {
            return column[index];
        }

        public void SetValue(int index, float v)
        {
            column[index] = v;
        }

    }

}


