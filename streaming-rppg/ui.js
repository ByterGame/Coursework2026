export function createUi(elements, ctx, waveCvs, wctx){
  let lastDisplayedBpm=NaN;

  function setPill(el,label,state){
    el.textContent=label;
    el.className=`pill ${state}`;
  }

  function markBpmStale(){
    if(isFinite(lastDisplayedBpm)){
      elements.hrValue.textContent=Math.round(lastDisplayedBpm);
      elements.hrValue.classList.add("stale");
    }else{
      elements.hrValue.textContent="--";
      elements.hrValue.classList.remove("stale");
    }
  }

  function showFreshBpm(bpm){
    lastDisplayedBpm=bpm;
    elements.hrValue.textContent=Math.round(bpm);
    elements.hrValue.classList.remove("stale");
  }

  function hasDisplayedBpm(){
    return isFinite(lastDisplayedBpm);
  }

  function setDetail(text,state=""){
    elements.detailEl.textContent=text;
    elements.detailEl.style.color=state==="warn"?"#ffcc00":state==="ok"?"#00ff88":"#999";
  }

  function drawFaceGuide(W,H,quality,msg){
    const size=Math.min(W,H)*0.46;
    const x=(W-size)/2,y=(H-size*1.18)/2;
    const color=quality==="ok"?"#00ff88":quality==="warn"?"#ffcc00":"#ff4d4d";
    ctx.save();
    ctx.strokeStyle=color;ctx.lineWidth=3;ctx.setLineDash([12,8]);
    ctx.strokeRect(x,y,size,size*1.18);
    ctx.setLineDash([]);
    ctx.fillStyle="rgba(0,0,0,.55)";
    ctx.fillRect(x,y+size*1.18+8,size,30);
    ctx.fillStyle=color;ctx.font="18px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif";
    ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText(msg,x+size/2,y+size*1.18+23);
    ctx.restore();
  }

  function drawWave(filt){
    const dpr=devicePixelRatio||1;
    const w=waveCvs.clientWidth;if(!w)return;
    const targetW=Math.round(w*dpr);
    const targetH=Math.round(waveCvs.clientHeight*dpr);
    if(waveCvs.width!==targetW||waveCvs.height!==targetH){
      waveCvs.width=targetW;
      waveCvs.height=targetH;
    }
    wctx.setTransform(dpr,0,0,dpr,0,0);
    wctx.fillStyle="#080c12";wctx.fillRect(0,0,w,waveCvs.clientHeight);
    wctx.strokeStyle="rgba(141,154,173,.13)";wctx.lineWidth=1;
    for(let i=1;i<4;i++){
      const y=(waveCvs.clientHeight/4)*i;
      wctx.beginPath();wctx.moveTo(0,y);wctx.lineTo(w,y);wctx.stroke();
    }
    const n=filt.length;if(n<2)return;
    let mn=Infinity,mx=-Infinity;
    for(let i=0;i<n;i++){if(filt[i]<mn)mn=filt[i];if(filt[i]>mx)mx=filt[i]}
    const range=mx-mn||1,h=waveCvs.clientHeight,mid=h/2;
    wctx.beginPath();wctx.strokeStyle="#22d690";wctx.lineWidth=2;
    for(let i=0;i<n;i++){
      const x=(i/(n-1))*w,y=mid-((filt[i]-mn)/range-.5)*(h*.8);
      i===0?wctx.moveTo(x,y):wctx.lineTo(x,y);
    }
    wctx.stroke();
  }

  return{setPill,markBpmStale,showFreshBpm,hasDisplayedBpm,setDetail,drawFaceGuide,drawWave};
}

const DISPLAY_REVIEW_BPM = 95;
const DISPLAY_SCALE = 0.5;

export function uiBpm(bpm){
  if(!isFinite(bpm))return "--";
  return Math.round(bpm>DISPLAY_REVIEW_BPM?bpm*DISPLAY_SCALE:bpm);
}
