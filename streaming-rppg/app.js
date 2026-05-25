import {
  FilesetResolver, FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs";
import {
  FS, WIN_SEC, POS_L, HR_LOW, HR_HIGH, MIN_GOOD_SEC, PRIOR_TTL_SEC,
  posSignal, chromSignal, detrend, butterBpf, peakFreq, selectStableEstimate,
  resampleUniform, medianArr, needsEstimateReview, normalizeDisplayEstimate,
} from "./signal.js";
import {createUi, uiBpm} from "./ui.js";

const video=document.getElementById("video");
const display=document.getElementById("display");
const overlay=document.getElementById("overlay");
const startBtn=document.getElementById("start-btn");
const waveCvs=document.getElementById("waveform");
const elements={
  hrValue:document.getElementById("hr-value"),
  statusEl:document.getElementById("status"),
  detailEl:document.getElementById("detail"),
  facePill:document.getElementById("face-pill"),
  lightPill:document.getElementById("light-pill"),
  fpsPill:document.getElementById("fps-pill"),
  signalPill:document.getElementById("signal-pill"),
  bufferMetric:document.getElementById("buffer-metric"),
  methodMetric:document.getElementById("method-metric"),
};

const ctx=display.getContext("2d",{willReadFrequently:true});
const wctx=waveCvs.getContext("2d");
const ui=createUi(elements,ctx,waveCvs,wctx);
const ext=document.createElement("canvas");
const ectx=ext.getContext("2d",{willReadFrequently:true});

let landmarker=null;
let rawBuf=[];
let firstHr=true;
let audioCtx=null;
let hrHistory=[];
let prevVts=null;
let fpsEma=30;
let totalFrames=0;
let lastAnalysisAt=0;
let lastFaceCenter=null;
let lastGoodBpm=NaN;
let lastGoodAt=0;
let pendingBpm=NaN;
let pendingAt=0;
let smoothMask=null;

function beep(){
  if(!audioCtx) audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  const o=audioCtx.createOscillator(),g=audioCtx.createGain();
  o.connect(g);g.connect(audioCtx.destination);
  o.type="sine";o.frequency.value=880;
  g.gain.setValueAtTime(.3,audioCtx.currentTime);
  g.gain.exponentialRampToValueAtTime(.001,audioCtx.currentTime+.25);
  o.start();o.stop(audioCtx.currentTime+.25);
}

function setEstimateDetail(pos=null, chromPeak=null, snr=NaN, state=""){
  const posText=pos&&isFinite(pos.bpm)?`${uiBpm(pos.bpm)} bpm`:"--";
  const chromText=chromPeak&&isFinite(chromPeak.bpm)?`${uiBpm(chromPeak.bpm)} bpm`:"--";
  const snrText=isFinite(snr)?snr.toFixed(1):"--";
  ui.setDetail(`POS ${posText} | CHROM ${chromText} | SNR ${snrText}`,state);
}

function smoothPoly(poly){
  if(!smoothMask||smoothMask.length!==poly.length){
    smoothMask=poly.map(p=>({x:p.x,y:p.y}));
    return smoothMask;
  }
  const a=0.28;
  for(let i=0;i<poly.length;i++){
    smoothMask[i].x=smoothMask[i].x*(1-a)+poly[i].x*a;
    smoothMask[i].y=smoothMask[i].y*(1-a)+poly[i].y*a;
  }
  return smoothMask;
}

function assessFrame(lm,rgbStats,W,H){
  let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity;
  for(const p of lm){
    const x=p.x*W,y=p.y*H;
    if(x<xMin)xMin=x;if(x>xMax)xMax=x;if(y<yMin)yMin=y;if(y>yMax)yMax=y;
  }
  const faceW=xMax-xMin,faceH=yMax-yMin;
  const cx=(xMin+xMax)/2,cy=(yMin+yMax)/2;
  const guideSize=Math.min(W,H)*0.46;
  const ratio=faceW/guideSize;
  const centered=Math.abs(cx-W/2)<W*0.12 && Math.abs(cy-H*0.48)<H*0.16;
  let faceState="ok",faceMsg="hold still";
  if(ratio<0.72){faceState="bad";faceMsg="move closer"}
  else if(ratio>1.18){faceState="bad";faceMsg="move farther"}
  else if(!centered){faceState="warn";faceMsg="center face"}

  let motion=0;
  if(lastFaceCenter) motion=Math.hypot(cx-lastFaceCenter.x,cy-lastFaceCenter.y)/Math.max(1,faceH);
  lastFaceCenter={x:cx,y:cy};
  if(motion>0.035){faceState="warn";faceMsg="hold still"}

  const lightMean=(rgbStats.r+rgbStats.g+rgbStats.b)/3;
  let lightState="ok",lightMsg="light ok";
  if(lightMean<45){lightState="bad";lightMsg="more light"}
  else if(lightMean>225 || rgbStats.satRatio>0.025){lightState="bad";lightMsg="too bright"}
  else if(lightMean<70 || lightMean>205){lightState="warn";lightMsg="adjust light"}

  const fpsState=fpsEma>=27?"ok":fpsEma>=22?"warn":"bad";
  const ok=faceState!=="bad" && lightState!=="bad" && fpsState!=="bad";
  return{ok,faceState,faceMsg,lightState,lightMsg,fpsState,ratio,motion};
}

function foreheadMask(lm,w,h){
  let xMin=Infinity,xMax=-Infinity;
  for(const p of lm){const x=p.x*w;if(x<xMin)xMin=x;if(x>xMax)xMax=x}
  const fw=xMax-xMin;
  const yBrow=(lm[105].y+lm[334].y)/2*h;
  const yChin=lm[152].y*h;
  const fh=Math.max(1,yChin-yBrow)*.55;
  const topY=yBrow-fh*.50,botY=yBrow-fh*.20;
  const cx=(xMin+xMax)/2;
  return[
    {x:cx-fw*.30,y:topY},{x:cx+fw*.30,y:topY},
    {x:cx+fw*.30,y:botY},{x:cx-fw*.30,y:botY}
  ];
}

function extractRGB(img,poly,w,h){
  const d=img.data;
  const xs=poly.map(p=>Math.round(p.x)),ys=poly.map(p=>Math.round(p.y));
  const x0=Math.max(0,Math.min(...xs)),x1=Math.min(w-1,Math.max(...xs));
  const y0=Math.max(0,Math.min(...ys)),y1=Math.min(h-1,Math.max(...ys));
  const ccx=(x0+x1)/2,ccy=(y0+y1)/2;
  const rx2=Math.max(1,((x1-x0)/2)**2),ry2=Math.max(1,((y1-y0)/2)**2);
  let sr=0,sg=0,sb=0,cnt=0,sat=0;
  for(let y=y0;y<=y1;y++)for(let x=x0;x<=x1;x++){
    if(((x-ccx)**2)/rx2+((y-ccy)**2)/ry2>1)continue;
    const i=(y*w+x)*4,r=d[i],g=d[i+1],b=d[i+2];
    sr+=r;sg+=g;sb+=b;cnt++;
    if(r>248||g>248||b>248||r<4||g<4||b<4)sat++;
  }
  if(!cnt)return{r:0,g:0,b:0,satRatio:1};
  return{r:sr/cnt,g:sg/cnt,b:sb/cnt,satRatio:sat/cnt};
}

function processFrame(info){
  if(!landmarker)return;
  try{
    const W=video.videoWidth,H=video.videoHeight;
    if(!W){video.requestVideoFrameCallback(processFrame);return}
    if(display.width!==W||display.height!==H){display.width=W;display.height=H;ext.width=W;ext.height=H}

    ctx.setTransform(-1,0,0,1,W,0);
    ctx.drawImage(video,0,0,W,H);
    ctx.setTransform(1,0,0,1,0,0);
    ectx.drawImage(video,0,0,W,H);

    const vts=info.mediaTime;
    if(prevVts!==null){
      const dt=vts-prevVts;
      if(dt>0) fpsEma=fpsEma*0.95+(1/dt)*0.05;
    }
    prevVts=vts;

    const res=landmarker.detectForVideo(ext,performance.now());
    if(!res.faceLandmarks||!res.faceLandmarks.length){
      ui.markBpmStale();
      elements.statusEl.textContent=ui.hasDisplayedBpm()?"no face - last value may be outdated":"no face - look at camera";
      setEstimateDetail();
      ui.setPill(elements.facePill,"no face","bad");
      ui.setPill(elements.lightPill,"light","warn");
      ui.setPill(elements.fpsPill,`${fpsEma.toFixed(1)} fps`,fpsEma>=27?"ok":fpsEma>=22?"warn":"bad");
      ui.setPill(elements.signalPill,"signal","bad");
      ui.drawFaceGuide(W,H,"bad","look at camera");
      lastFaceCenter=null;pendingBpm=NaN;smoothMask=null;
      video.requestVideoFrameCallback(processFrame);return;
    }

    const lm=res.faceLandmarks[0];
    const mask=smoothPoly(foreheadMask(lm,W,H));
    const rgb=extractRGB(ectx.getImageData(0,0,W,H),mask,W,H);
    const frameQuality=assessFrame(lm,rgb,W,H);
    const guideQuality=frameQuality.faceState==="bad"||frameQuality.lightState==="bad"||frameQuality.fpsState==="bad"
      ?"bad":frameQuality.faceState==="warn"||frameQuality.lightState==="warn"||frameQuality.fpsState==="warn"?"warn":"ok";
    ui.drawFaceGuide(W,H,guideQuality,frameQuality.faceMsg);

    const roiColor=frameQuality.ok?"#00ff88":"#ffcc00";
    ctx.fillStyle=frameQuality.ok?"rgba(0,255,136,.22)":"rgba(255,204,0,.20)";
    ctx.strokeStyle=roiColor;ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(W-mask[0].x,mask[0].y);
    for(let i=1;i<mask.length;i++)ctx.lineTo(W-mask[i].x,mask[i].y);
    ctx.closePath();ctx.fill();ctx.stroke();

    const sampleTime=performance.now()/1000;
    if(frameQuality.ok){
      rawBuf.push({t:sampleTime,r:rgb.r,g:rgb.g,b:rgb.b});
      const keepAfter=sampleTime-(WIN_SEC+3);
      while(rawBuf.length && rawBuf[0].t<keepAfter)rawBuf.shift();
    }else{
      rawBuf=[];hrHistory=[];pendingBpm=NaN;smoothMask=null;lastGoodBpm=NaN;lastGoodAt=0;
      ui.markBpmStale();
      setEstimateDetail();
    }
    totalFrames++;

    ui.setPill(elements.facePill,frameQuality.faceMsg,frameQuality.faceState);
    ui.setPill(elements.lightPill,frameQuality.lightMsg,frameQuality.lightState);
    ui.setPill(elements.fpsPill,`${fpsEma.toFixed(1)} fps`,frameQuality.fpsState);

    const bufferedSec=rawBuf.length>1?rawBuf[rawBuf.length-1].t-rawBuf[0].t:0;
    elements.bufferMetric.textContent=`${Math.min(WIN_SEC,bufferedSec).toFixed(1)}s`;
    elements.statusEl.textContent=frameQuality.ok
      ?`collecting stable frames... ${Math.min(WIN_SEC,bufferedSec).toFixed(1)}/${WIN_SEC}s`
      :ui.hasDisplayedBpm()
        ?`${frameQuality.faceMsg}; ${frameQuality.lightMsg} - last value may be outdated`
        :`${frameQuality.faceMsg}; ${frameQuality.lightMsg}`;
    ui.setPill(elements.signalPill,bufferedSec>=MIN_GOOD_SEC?"ready":"collecting",bufferedSec>=MIN_GOOD_SEC?"ok":"warn");

    if(frameQuality.ok && bufferedSec>=MIN_GOOD_SEC && sampleTime-lastAnalysisAt>=1){
      lastAnalysisAt=sampleTime;
      const series=resampleUniform(rawBuf,FS,Math.min(WIN_SEC,bufferedSec));
      if(!series){video.requestVideoFrameCallback(processFrame);return}
      const R=series.R,G=series.G,B=series.B;

      let gMean=0;for(const v of G)gMean+=v;gMean/=G.length;
      let gVar=0;for(const v of G)gVar+=(v-gMean)**2;gVar/=G.length;
      const gCv=gMean>0?Math.sqrt(gVar)/gMean:0;

      const pulse=posSignal(R,G,B,POS_L);
      const f=butterBpf(detrend(pulse),HR_LOW,HR_HIGH,FS);
      const recentPrior=sampleTime-lastGoodAt<PRIOR_TTL_SEC?lastGoodBpm:NaN;
      const pos=peakFreq(f,FS,HR_LOW,HR_HIGH,recentPrior);

      const chrom=chromSignal(R,G,B);
      const chromF=butterBpf(detrend(chrom),HR_LOW,HR_HIGH,FS);
      const chromPeak=peakFreq(chromF,FS,HR_LOW,HR_HIGH,recentPrior);
      const chosen=selectStableEstimate(pos,chromPeak,recentPrior);
      let bpm=chosen.bpm,snr=chosen.snr;
      bpm=normalizeDisplayEstimate(bpm);
      elements.methodMetric.textContent="POS";
      setEstimateDetail(pos,chromPeak,snr);

      const chromReliable=isFinite(chromPeak.bpm)&&chromPeak.snr>=4;
      const chromSupports=chromReliable && Math.abs(chromPeak.bpm-bpm)<=12;
      const chromSupportsAlternate=chromReliable && (
        Math.abs(chromPeak.bpm*2-bpm)<=10 ||
        Math.abs(chromPeak.bpm-bpm*2)<=10
      );
      const posBattle=pos.ambiguous&&isFinite(pos.rivalBpm)?`${Math.round(pos.bpm)} vs ${Math.round(pos.rivalBpm)} bpm`:"";

      ui.drawWave(f);

      const jumpOk=!isFinite(lastGoodBpm)||Math.abs(bpm-lastGoodBpm)<28||hrHistory.length<3;
      const firstNeedsConfirm=!isFinite(recentPrior);
      const pendingOk=isFinite(pendingBpm)&&Math.abs(bpm-pendingBpm)<=10&&sampleTime-pendingAt<18;
      const firstHighNeedsStabilityCheck=firstNeedsConfirm&&!chosen.stabilized&&uiBpm(bpm)>=HR_LOW*60&&needsEstimateReview(bpm);
      const ambiguous=!chosen.stabilized && (pos.ambiguous || (chromSupportsAlternate && !isFinite(recentPrior)) || firstHighNeedsStabilityCheck);
      const ensembleOk=chosen.stabilized || chromSupports || !chromReliable || snr>chromPeak.snr*1.7;
      const canAccept=(!firstNeedsConfirm||pendingOk) && !ambiguous && ensembleOk;

      if(isFinite(bpm)&&bpm>=42&&bpm<=95&&snr>5&&gCv>0.001&&gCv<0.08&&jumpOk&&canAccept){
        hrHistory.push(bpm);
        if(hrHistory.length>15) hrHistory.shift();
        const stable=Math.round(medianArr(hrHistory));
        lastGoodBpm=stable;lastGoodAt=sampleTime;pendingBpm=NaN;
        ui.showFreshBpm(stable);
        elements.statusEl.textContent=`heart rate: ${stable} bpm  (chrom:${uiBpm(chromPeak.bpm)} snr:${snr.toFixed(1)})`;
        setEstimateDetail(pos,chromPeak,snr,"ok");
        ui.setPill(elements.signalPill,`snr ${snr.toFixed(1)}`,"ok");
        if(firstHr){firstHr=false;beep()}
      }else{
        if(firstNeedsConfirm&&isFinite(bpm)&&!ambiguous&&ensembleOk){
          pendingBpm=bpm;pendingAt=sampleTime;
        }
        let reason=!jumpOk?"unstable":snr<=5?"low signal":"waiting";
        let signalLabel=reason;
        if(ambiguous){
          const battle=needsEstimateReview(pos.bpm)?`${uiBpm(pos.bpm)} bpm stability check`:(posBattle||`${uiBpm(pos.bpm)} bpm stability check`);
          const chromText=isFinite(chromPeak.bpm)?`pos ${battle}, chrom ${uiBpm(chromPeak.bpm)} bpm`:`pos ${battle}`;
          reason=`ambiguous: ${chromText}`;
          setEstimateDetail(pos,chromPeak,snr,"warn");
          signalLabel="ambiguous";
        }else if(firstNeedsConfirm&&!pendingOk&&isFinite(bpm)){
          reason=`confirming ${Math.round(bpm)} bpm`;
          setEstimateDetail(pos,chromPeak,snr,"warn");
          signalLabel="confirming";
        }else if(!ensembleOk&&isFinite(chromPeak.bpm)){
          reason=`POS/CHROM disagree: ${uiBpm(pos.bpm)} vs ${uiBpm(chromPeak.bpm)} bpm`;
          setEstimateDetail(pos,chromPeak,snr,"warn");
          signalLabel="disagree";
        }else{
          setEstimateDetail(pos,chromPeak,snr,"warn");
        }
        elements.statusEl.textContent=`${reason}... (snr:${snr.toFixed(1)} cv:${(gCv*100).toFixed(2)}% ${fpsEma.toFixed(1)}fps)`;
        ui.setPill(elements.signalPill,signalLabel,"warn");
      }
    }
  }catch(e){console.warn("frame error:",e)}
  video.requestVideoFrameCallback(processFrame);
}

async function start(){
  startBtn.disabled=true;startBtn.textContent="loading...";
  try{
    const vis=await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
    landmarker=await FaceLandmarker.createFromOptions(vis,{
      baseOptions:{
        modelAssetPath:"https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate:"GPU"
      },
      runningMode:"VIDEO",numFaces:1
    });
    const stream=await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:"user",width:{ideal:640},height:{ideal:480},frameRate:{ideal:30}}});
    video.srcObject=stream;await video.play();
    overlay.classList.add("hidden");
    elements.statusEl.textContent="face detected - collecting frames...";
    video.requestVideoFrameCallback(processFrame);
  }catch(e){
    overlay.classList.remove("hidden");
    startBtn.textContent="Start";startBtn.disabled=false;
    elements.statusEl.textContent="error: "+e.message;console.error(e);
  }
}

startBtn.addEventListener("click",start);
