using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN
{

    public abstract class InnerLayer : Layer
    {

        [SerializeField] protected int rows, columns;

        public InnerLayer(int rows, int columns) : base()
        {
            this.rows = rows;
            this.columns = columns;
        }

    }

}


