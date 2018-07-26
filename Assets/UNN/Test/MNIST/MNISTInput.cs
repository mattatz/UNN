using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace UNN.Test
{

    public class MNISTInput : MonoBehaviour {

        [SerializeField] protected MNISTTest mnist;
        [SerializeField] protected Material input;
        [SerializeField] protected ComputeShader converter;
        [SerializeField, Range(0.05f, 0.1f)] protected float size = 0.1f;

        protected MaterialPropertyBlock block;
        new protected Renderer renderer;
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

            renderer = GetComponent<Renderer>();
            block = new MaterialPropertyBlock();
            renderer.GetPropertyBlock(block);
            block.SetTexture("_Input", buffers[0]);
            renderer.SetPropertyBlock(block);
        }

        protected Vector3 offset = new Vector3(0.5f, 0.5f, 0f);
        
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
                var ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                var pl = new Plane(-Camera.main.transform.forward, transform.position);
                float enter;
                if(pl.Raycast(ray, out enter))
                {
                    var p = ray.direction * enter + ray.origin;
                    p = transform.InverseTransformPoint(p) + offset;
                    if(0f <= p.x && p.x < 1f && 0f < p.y && p.y < 1f)
                    {
                        input.SetTexture("_Source", buffers[read]);
                        input.SetVector("_Point", p);
                        input.SetFloat("_Size", size);

                        Graphics.Blit(null, buffers[write], input, 0);
                        Swap();
                    }
                }
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


