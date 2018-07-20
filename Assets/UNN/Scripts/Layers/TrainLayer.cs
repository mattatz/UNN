using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public abstract class TrainLayer : Layer
    {

        [SerializeField] protected int rows, columns;

        public TrainLayer(int rows, int columns) : base()
        {
            this.rows = rows;
            this.columns = columns;
        }

        public abstract void Learn(Optimizer optimizer, ComputeShader compute, float rate);

    }

}


