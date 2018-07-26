using System.Text;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;

namespace UNN.Test
{

    public class Digit {

        public float[] Pixels { get { return pixels; } }
        public int Label { get { return label; } }


        protected float[] pixels;
        protected int label;

        public Digit(float[] pixels, int label)
        {
            this.pixels = pixels;
            this.label = label;
        }

        public float[] GetOneHotLabel()
        {
            const int n = 10;
            var labels = new float[n];
            for(int i = 0; i < n; i++)
            {
                labels[i] = 0f;
            }
            labels[label] = 1f;
            return labels;
        }

        public Texture2D ToTexture(int rows, int cols)
        {
            var tex = new Texture2D(rows, cols);
            for(int y = 0; y < rows; y++)
            {
                for(int x = 0; x < cols; x++)
                {
                    var v = pixels[y * cols + x];
                    tex.SetPixel(x, y, Color.white * v);
                }
            }
            tex.Apply();
            return tex;
        }

    }

}


