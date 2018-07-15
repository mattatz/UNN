using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN.Test
{

    public class MNISTInput : MonoBehaviour {

        [SerializeField] protected MNIST network;
        [SerializeField] protected Material input;
        [SerializeField] protected ComputeShader converter;
        [SerializeField, Range(0.05f, 0.1f)] protected float size = 0.1f;

        protected bool dragging;

        [SerializeField] protected RenderTexture[] buffers;
        protected int read = 0;
        protected int write = 1;

        void Start () {
            buffers = new RenderTexture[2];
            buffers[0] = Create(128, 128);
            buffers[1] = Create(128, 128);
        }
        
        void Update () {

            if(Input.GetMouseButtonDown(0))
            {
                dragging = true;
            } else if(Input.GetMouseButtonUp(0))
            {
                Test();

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

        protected void Test()
        {
            var signal = new Signal(1, 28 * 28);

            var kernel = converter.FindKernel("Convert");
            converter.SetTexture(kernel, "_Input", buffers[read]);
            converter.SetBuffer(kernel, "_Digit", signal.Buffer);

            converter.SetInt("_Cols", signal.Columns);
            converter.SetInt("_Rows", signal.Rows);
            converter.SetVector("_TexelSize", new Vector2((1f / signal.Columns) * 0.5f, (1f / signal.Rows) * 0.5f));

            uint tx, ty, tz;
            converter.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);
            converter.Dispatch(kernel, Mathf.FloorToInt(((int)signal.Columns - 1) / tx) + 1, Mathf.FloorToInt(((int)signal.Rows - 1) / ty) + 1, (int)tz);

            // signal.Log();

            var result = network.Evaluate(signal);
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

            Debug.Log(label);

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

        protected void OnGUI()
        {
            var texture = buffers[0];
            GUI.DrawTexture(new Rect(0, 0, Screen.width, Screen.height), texture);
        }

    }

}


