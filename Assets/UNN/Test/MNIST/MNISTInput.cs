using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN.Test
{

    public class MNISTInput : MonoBehaviour {

        public RenderTexture Buffer { get { return buffers[read]; } }

        [SerializeField] protected MNISTTest mnist;
        [SerializeField] protected Material input;
        [SerializeField] protected ComputeShader converter;
        [SerializeField, Range(0.05f, 0.1f)] protected float size = 0.1f;

        protected bool dragging;

        [SerializeField] protected RenderTexture[] buffers;
        protected int read = 0;
        protected int write = 1;

        protected int lastLabel;

        void Start () {
            const int resolution = 128;
            buffers = new RenderTexture[2];
            buffers[0] = Create(resolution, resolution);
            buffers[1] = Create(resolution, resolution);
        }
        
        void Update () {

            if(Input.GetMouseButtonDown(0))
            {
                dragging = true;
            } else if(Input.GetMouseButtonUp(0))
            {
                Evaluate();

                // clear
                Graphics.Blit(null, buffers[write], input, 1);
                Swap();

                dragging = false;
            }

            if(dragging)
            {
                var p = Camera.main.ScreenToViewportPoint(Input.mousePosition);

                input.SetTexture("_Source", buffers[read]);
                input.SetVector("_Point", p);
                input.SetFloat("_Size", size);

                Graphics.Blit(null, buffers[write], input, 0);
                Swap();
            }
        }

        protected void Swap()
        {
            var tmp = read;
            read = write;
            write = tmp;
        }

        protected void Evaluate()
        {
            const int rows = 28;
            const int columns = 28;
            var signal = new Signal(1, rows * columns);

            var kernel = converter.FindKernel("Convert");
            converter.SetTexture(kernel, "_Input", buffers[read]);
            converter.SetBuffer(kernel, "_Digit", signal.Buffer);

            converter.SetInt("_Cols", columns);
            converter.SetInt("_Rows", rows);
            converter.SetVector("_TexelSize", new Vector2((1f / columns) * 0.5f, (1f / rows) * 0.5f));

            uint tx, ty, tz;
            converter.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            converter.Dispatch(kernel, Mathf.FloorToInt(((int)columns - 1) / tx) + 1, Mathf.FloorToInt(((int)rows - 1) / ty) + 1, (int)tz);

            // signal.LogMNIST();

            mnist.Evaluate(signal);
            signal.Dispose();
        }


        protected RenderTexture Create(int width, int height)
        {
            var tex = new RenderTexture(width, height, 0);
            tex.filterMode = FilterMode.Bilinear;
            tex.enableRandomWrite = true;
            tex.Create();
            return tex;
        }

        protected void OnDestroy()
        {
            buffers[0].Release();
            buffers[1].Release();
        }

    }

}


