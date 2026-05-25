export const FS = 30;
export const WIN_SEC = 10;
export const POS_SEC = 1.6;
export const POS_L = Math.round(FS * POS_SEC);
export const HR_LOW = 0.7;
export const HR_HIGH = 3.3;
export const MIN_GOOD_SEC = 8;
export const PRIOR_TTL_SEC = 45;
const ESTIMATE_REVIEW_BPM = 95;
const SECONDARY_SCALE = 0.5;

export function needsEstimateReview(bpm){
  return isFinite(bpm)&&bpm>ESTIMATE_REVIEW_BPM;
}

export function normalizeDisplayEstimate(bpm){
  if(!isFinite(bpm))return bpm;
  return needsEstimateReview(bpm)?bpm*SECONDARY_SCALE:bpm;
}

export function posSignal(R,G,B,L){
  const n=R.length;if(n<L)return new Float64Array(n);
  const H=new Float64Array(n);
  const Wsum=new Float64Array(n);
  for(let nn=L-1;nn<n;nn++){
    const m=nn-L+1;
    let mR=0,mG=0,mB=0;
    for(let i=m;i<=nn;i++){mR+=R[i];mG+=G[i];mB+=B[i]}
    mR/=L;mG/=L;mB/=L;
    if(mR===0)mR=1;if(mG===0)mG=1;if(mB===0)mB=1;
    let v1=0,v2=0;
    const s1=new Float64Array(L),s2=new Float64Array(L);
    for(let i=0;i<L;i++){
      const cR=R[m+i]/mR,cG=G[m+i]/mG,cB=B[m+i]/mB;
      s1[i]=cG-cB;s2[i]=-2*cR+cG+cB;
      v1+=s1[i];v2+=s2[i];
    }
    const mean1=v1/L,mean2=v2/L;
    let sd1=0,sd2=0;
    for(let i=0;i<L;i++){sd1+=(s1[i]-mean1)**2;sd2+=(s2[i]-mean2)**2}
    sd1=Math.sqrt(sd1/L);sd2=Math.sqrt(sd2/L);
    const alpha=sd2>0?sd1/sd2:0;
    for(let i=0;i<L;i++){H[m+i]+=s1[i]+alpha*s2[i];Wsum[m+i]++}
  }
  for(let i=0;i<n;i++) if(Wsum[i]>0) H[i]/=Wsum[i];
  return H;
}

export function chromSignal(R,G,B){
  const n=R.length;
  const out=new Float64Array(n);
  if(n<2)return out;
  let mR=0,mG=0,mB=0;
  for(let i=0;i<n;i++){mR+=R[i];mG+=G[i];mB+=B[i]}
  mR/=n;mG/=n;mB/=n;
  if(mR===0)mR=1;if(mG===0)mG=1;if(mB===0)mB=1;
  const X=new Float64Array(n),Y=new Float64Array(n);
  let mx=0,my=0;
  for(let i=0;i<n;i++){
    const rn=R[i]/mR,gn=G[i]/mG,bn=B[i]/mB;
    X[i]=3*rn-2*gn;
    Y[i]=1.5*rn+gn-1.5*bn;
    mx+=X[i];my+=Y[i];
  }
  mx/=n;my/=n;
  let sx=0,sy=0;
  for(let i=0;i<n;i++){sx+=(X[i]-mx)**2;sy+=(Y[i]-my)**2}
  sx=Math.sqrt(sx/n);sy=Math.sqrt(sy/n);
  const alpha=sy>0?sx/sy:0;
  for(let i=0;i<n;i++)out[i]=X[i]-alpha*Y[i];
  return out;
}

export function detrend(x){
  const n=x.length;if(n<2)return x.slice();
  let sx=0,st=0,stt=0,sxt=0;
  for(let i=0;i<n;i++){sx+=x[i];st+=i;stt+=i*i;sxt+=i*x[i]}
  const d=n*stt-st*st;if(Math.abs(d)<1e-12)return x.slice();
  const a=(n*sxt-st*sx)/d,b=(stt*sx-st*sxt)/d;
  const y=new Float64Array(n);
  for(let i=0;i<n;i++)y[i]=x[i]-(a*i+b);
  return y;
}

export function butterBpf(x,lo,hi,fs){
  const n=x.length;if(n<8)return x.slice();
  const w0=Math.PI*(lo+hi)/fs,bw=Math.PI*(hi-lo)/fs;
  const sinB=Math.sin(bw),cosW=Math.cos(w0),alpha=sinB/2;
  const a0=1+alpha,a1=-2*cosW,a2=1-alpha;
  const s={b0:alpha/a0,b1:0,b2:-alpha/a0,a1:a1/a0,a2:a2/a0};
  function biquad(v){
    const y=new Float64Array(n);let z1=0,z2=0;
    for(let i=0;i<n;i++){
      const t=v[i]-s.a1*z1-s.a2*z2;
      y[i]=s.b0*t+s.b1*z1+s.b2*z2;z2=z1;z1=t;
    }return y;
  }
  let y=biquad(x);
  const rev=new Float64Array(n);for(let i=0;i<n;i++)rev[i]=y[n-1-i];
  y=biquad(rev);
  const out=new Float64Array(n);for(let i=0;i<n;i++)out[i]=y[n-1-i];
  return out;
}

function fft(re,im){
  const N=re.length;
  for(let i=1,j=0;i<N;i++){
    let b=N>>1;for(;j&b;b>>=1)j^=b;j^=b;
    if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]]}
  }
  for(let len=2;len<=N;len*=2){
    const ang=-2*Math.PI/len,wr=Math.cos(ang),wi=Math.sin(ang);
    for(let i=0;i<N;i+=len){
      let cr=1,ci=0;
      for(let j=0;j<len/2;j++){
        const p=i+j,q=p+len/2;
        const ur=cr*re[q]-ci*im[q],ui=cr*im[q]+ci*re[q];
        re[q]=re[p]-ur;im[q]=im[p]-ui;
        re[p]+=ur;im[p]+=ui;
        const tr=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=tr;
      }
    }
  }
}

export function peakFreq(sig,fs,lo,hi,prevBpm=NaN){
  const N=1<<Math.ceil(Math.log2(sig.length));
  const re=new Float64Array(N),im=new Float64Array(N);
  for(let i=0;i<sig.length;i++){
    const win=0.5-0.5*Math.cos((2*Math.PI*i)/(sig.length-1));
    re[i]=sig[i]*win;
  }
  fft(re,im);
  const res=fs/N;
  const iLo=Math.max(1,Math.ceil(lo/res));
  const iHi=Math.min(N/2-1,Math.floor(hi/res));
  if(iLo>=iHi)return{bpm:NaN,snr:0};
  const powers=[];
  for(let i=iLo;i<=iHi;i++)powers.push(re[i]*re[i]+im[i]*im[i]);
  const sortedPowers=powers.slice().sort((a,b)=>a-b);
  const medianPow=sortedPowers[Math.floor(sortedPowers.length/2)]||1e-12;
  function goertzel(f){
    const w=2*Math.PI*f/fs,coeff=2*Math.cos(w);
    let s0=0,s1=0,s2=0;
    for(let i=0;i<sig.length;i++){s0=sig[i]+coeff*s1-s2;s2=s1;s1=s0}
    return s1*s1+s2*s2-coeff*s1*s2;
  }
  function normAutocorr(testBpm){
    const n=sig.length;
    const lag=Math.round(fs*60/testBpm);
    if(lag<2||lag>=n)return 0;
    let ac=0,var_=0,m=0;
    for(let i=0;i<n;i++)m+=sig[i];m/=n;
    for(let i=0;i<n;i++)var_+=(sig[i]-m)**2;
    if(var_<=0)return 0;
    for(let i=0;i<n-lag;i++) ac+=(sig[i]-m)*(sig[i+lag]-m);
    return ac/var_;
  }
  function refineBin(bin){
    let fl=Math.max(lo,(bin-1)*res),fh=Math.min(hi,(bin+1)*res);
    for(let k=0;k<12;k++){
      const m1=fl+(fh-fl)/3,m2=fh-(fh-fl)/3;
      if(goertzel(m1)>goertzel(m2))fh=m2;else fl=m1;
    }
    return (fl+fh)/2;
  }
  const peaks=[];
  for(let i=iLo+1;i<iHi;i++){
    const p=re[i]*re[i]+im[i]*im[i];
    const pl=re[i-1]*re[i-1]+im[i-1]*im[i-1];
    const pr=re[i+1]*re[i+1]+im[i+1]*im[i+1];
    if(p>=pl&&p>=pr)peaks.push({bin:i,power:p});
  }
  if(!peaks.length){
    for(let i=iLo;i<=iHi;i++)peaks.push({bin:i,power:re[i]*re[i]+im[i]*im[i]});
  }
  peaks.sort((a,b)=>b.power-a.power);
  const byBpm=new Map();
  function addCandidate(freq,source){
    if(freq<lo||freq>hi)return;
    const bpm=freq*60;
    const key=Math.round(bpm);
    const power=goertzel(freq);
    const ac=normAutocorr(bpm);
    const prevPenalty=isFinite(prevBpm)?Math.min(Math.abs(bpm-prevBpm)/35,2):0;
    const score=Math.log(Math.max(power,1e-12)/medianPow)+2.5*Math.max(0,ac)-prevPenalty;
    const cand={bpm,power,ac,score,source};
    const old=byBpm.get(key);
    if(!old||cand.score>old.score)byBpm.set(key,cand);
  }
  for(const p of peaks.slice(0,8)){
    const f=refineBin(p.bin);
    addCandidate(f,"peak");
    const secondaryFreq=f*SECONDARY_SCALE;
    addCandidate(secondaryFreq,"secondary");
  }
  const candidates=[...byBpm.values()];
  for(const high of candidates){
    if(high.bpm<ESTIMATE_REVIEW_BPM)continue;
    const low=candidates.find(c=>Math.abs(c.bpm*2-high.bpm)<6);
    if(!low)continue;
    const lowSupport=low.ac>=high.ac*0.65||low.power>=high.power*0.22;
    if(lowSupport){
      high.score-=1.0;
      low.score+=0.7;
    }
  }
  candidates.sort((a,b)=>b.score-a.score);
  const best=candidates[0];
  if(!best)return{bpm:NaN,snr:0};
  const rival=candidates.find(c=>c!==best && (
    Math.abs(c.bpm*2-best.bpm)<6 ||
    Math.abs(c.bpm-best.bpm*2)<6 ||
    Math.abs(c.bpm-best.bpm)<12
  ));
  const ambiguous=!!rival && rival.score>best.score-1.15;
  return{bpm:best.bpm,snr:best.power/medianPow,candidates:candidates.slice(0,4),ambiguous,rivalBpm:rival?rival.bpm:NaN};
}

export function selectStableEstimate(pos, chromPeak, recentPrior){
  const out={bpm:pos.bpm,snr:pos.snr,stabilized:false,rawBpm:pos.bpm,alternateBpm:NaN,reason:"POS"};
  if(!needsEstimateReview(pos.bpm)||!pos.candidates)return out;
  const expected=pos.bpm*SECONDARY_SCALE;
  const alternate=pos.candidates
    .filter(c=>c.bpm>=42&&c.bpm<=95&&Math.abs(c.bpm-expected)<=7)
    .sort((a,b)=>b.score-a.score)[0];
  if(!alternate)return out;
  const chromBpm=chromPeak&&isFinite(chromPeak.bpm)?chromPeak.bpm:NaN;
  const chromSupportsAlternate=isFinite(chromBpm)&&Math.abs(chromBpm-alternate.bpm)<=12;
  const priorSupportsAlternate=isFinite(recentPrior)&&Math.abs(recentPrior-alternate.bpm)<=14;
  const autocorrSupport=alternate.ac>=0.02 || alternate.ac>=Math.max(0,pos.candidates[0]?.ac||0)*0.45;
  const powerSupport=alternate.power>=Math.max(1e-12,pos.candidates[0]?.power||1e-12)*0.12;
  const scoreClose=alternate.score>=(pos.candidates[0]?.score??0)-2.2;
  const hasAnyConfidence=chromSupportsAlternate||priorSupportsAlternate||autocorrSupport||powerSupport||scoreClose;
  if(hasAnyConfidence){
    out.bpm=alternate.bpm;
    out.snr=pos.snr;
    out.stabilized=true;
    out.alternateBpm=alternate.bpm;
    out.reason="stability filter";
  }
  return out;
}

export function resampleUniform(samples, fs, seconds){
  if(samples.length<2)return null;
  const end=samples[samples.length-1].t;
  const start=end-seconds;
  if(samples[0].t>start+0.25)return null;
  const n=Math.round(seconds*fs);
  const R=new Float64Array(n),G=new Float64Array(n),B=new Float64Array(n);
  let j=0;
  for(let i=0;i<n;i++){
    const t=start+i/fs;
    while(j<samples.length-2 && samples[j+1].t<t)j++;
    const a=samples[j],b=samples[Math.min(j+1,samples.length-1)];
    const span=Math.max(1e-6,b.t-a.t);
    const u=Math.min(1,Math.max(0,(t-a.t)/span));
    R[i]=a.r+(b.r-a.r)*u;
    G[i]=a.g+(b.g-a.g)*u;
    B[i]=a.b+(b.b-a.b)*u;
  }
  return{R,G,B};
}

export function medianArr(arr){
  const s=arr.slice().sort((a,b)=>a-b);
  const m=s.length>>1;
  return s.length%2?s[m]:(s[m-1]+s[m])/2;
}
