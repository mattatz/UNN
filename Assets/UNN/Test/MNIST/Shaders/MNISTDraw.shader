Shader "UNN/Test/MNISTDraw"
{

  Properties
  {
    _Source ("", 2D) = "" {}
    _Size ("Size", Float) = 0.1
    _Alpha ("Alpha", Range(0.0, 1.0)) = 0.1
  }

  CGINCLUDE

  struct appdata
  {
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
  };

  struct v2f
  {
    float2 uv : TEXCOORD0;
    float4 vertex : SV_POSITION;
  };

  v2f vert(appdata v)
  {
    v2f o;
    o.vertex = UnityObjectToClipPos(v.vertex);
    o.uv = v.uv;
    return o;
  }

  sampler2D _Source;
  float2 _Point;
  float _Size, _Alpha;

  ENDCG

  SubShader
  {
    Cull Off ZWrite Off ZTest Always

    // paint
    Pass 
    {
      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag

      #include "UnityCG.cginc"

      fixed4 frag(v2f i) : SV_Target
      {
        fixed4 col = tex2D(_Source, i.uv);
        float d = distance(_Point, i.uv);
        col.rgb += saturate(1.0 - d / _Size);
        col.rgb = saturate(col.rgb);
        col.a = _Alpha;
        return col;
      }

      ENDCG
    }

    // clear
    Pass 
    {
      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag

      #include "UnityCG.cginc"

      fixed4 frag(v2f i) : SV_Target
      {
        return float4(0, 0, 0, _Alpha);
      }

      ENDCG
    }

  }

}
