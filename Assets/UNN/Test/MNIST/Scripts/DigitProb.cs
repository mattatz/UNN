using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.UI;

namespace UNN.Test
{

    [RequireComponent (typeof(RawImage))]
    public class DigitProb : MonoBehaviour {

        [SerializeField] protected Color color;
        [SerializeField, Range(0f, 1f)] protected float t = 1f;
        protected Material material;

        protected Coroutine co;

        protected void Start () {
            var img = GetComponent<RawImage>();
            material = new Material(img.material); // clone
            img.material = material;
        }
        
        protected void Update () {
            material.SetFloat("_T", t);
        }

        public void SetProbability(float v)
        {
            v = Mathf.Clamp01(v);

            if(co != null)
            {
                StopCoroutine(co);
            }
            StartCoroutine(IAnimate(v, 0.25f));
        }

        protected IEnumerator IAnimate(float v, float duration)
        {
            yield return 0;

            var from = t;

            var time = 0f;
            while(time < duration)
            {
                yield return 0;
                time += Time.deltaTime;

                var k = time / duration;
                t = Mathf.Lerp(from, v, k * (2f - k)); // easing with quad out
            }

            t = v;
        }

        protected void OnDestroy()
        {
            Destroy(material);
        }

    }

}


