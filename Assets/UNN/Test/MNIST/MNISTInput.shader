Shader "UNN/Test/MNISTInput"
{

  Properties
  {
    _Input ("", 2D) = "" {}
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

  ENDCG

  SubShader
  {
    Cull Off ZWrite Off ZTest Always

    Pass
    {
      CGPROGRAM
      #pragma vertex vert
      #pragma fragment frag

      #include "UnityCG.cginc"

      sampler2D _Input;

      fixed4 frag(v2f i) : SV_Target
      {
        fixed4 col = tex2D(_Input, i.uv);
        col.rgb = (1.0).xxx - col.rgb;
        return col;
      }

      ENDCG
    }

  }

}
