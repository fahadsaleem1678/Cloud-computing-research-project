// Reproducible stylized controller study. Node.js 20+; no external packages.
// All compute times, activation sizes, and link traces below are assumptions,
// not measurements of MobileNetV2, ResNet18, CIFAR-10, or physical hardware.
const fs = require('fs');
const path = require('path');

const OUT = __dirname;
const REPS = 40;
const N = 300;
const DEADLINE_MS = 120;
const EPOCH_SECONDS = 1;
const SWITCH_COST_MS = 22;
const H = 0.08, K = 3, D = 8;
const CUTS = {
  edge:  {edgeMs:115, serverMs:0,  megabytes:0,    rttMs:0},
  split: {edgeMs:55,  serverMs:20, megabytes:0.24, rttMs:12},
  cloud: {edgeMs:0,   serverMs:50, megabytes:0.95, rttMs:12}
};
const names = Object.keys(CUTS);
const policyNames = ['edge-only','cloud-only','fixed-split','greedy','H','K','D','HK','HD','KD','HKD'];

function rng(seed) { let a=seed>>>0; return () => {a+=0x6D2B79F5; let t=a; t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;}; }
function normal(rand) { return Math.sqrt(-2*Math.log(Math.max(rand(),1e-12)))*Math.cos(2*Math.PI*rand()); }
function percentile(a,q) {const s=[...a].sort((x,y)=>x-y);const x=(s.length-1)*q;const lo=Math.floor(x),hi=Math.ceil(x);return s[lo]+(s[hi]-s[lo])*(x-lo);}
function cost(cut,bw) {const x=CUTS[cut];return x.edgeMs+x.serverMs+x.rttMs+8000*x.megabytes/bw;}
function bandwidth(trace,t,rep) {
  const phase=(rep%7)*0.71;
  if (trace==='steady') return 100*(1+0.08*Math.sin(t/13+phase));
  if (trace==='step') return t<150 ? 15 : 150;
  if (trace==='bursty') return t%40<6 ? 12 : 150;
  if (trace==='periodic') return Math.floor(t/25)%2===0 ? 25 : 180;
  throw new Error(trace);
}
function controls(policy) {return {h:policy.includes('H')?H:0,k:policy.includes('K')?K:1,d:policy.includes('D')?D:0};}
function simulate(trace,rep,policy) {
  const rand=rng(4000+rep*97+['steady','step','bursty','periodic'].indexOf(trace)*1009);
  const values=[], actual=[], observed=[], jitter=[];
  for(let t=0;t<N;t++) {actual.push(bandwidth(trace,t,rep));observed.push(Math.max(1,actual[t]*Math.exp(normal(rand)*0.24)));jitter.push(normal(rand)*5);}
  let current='edge', switches=0, lastSwitch=-10000, candidate=null, consecutive=0;
  let firstStable=null, stepAt=150;
  const cfg=controls(policy);
  for(let t=0;t<N;t++) {
    let changed=false;
    if(policy==='edge-only') current='edge';
    else if(policy==='cloud-only') current='cloud';
    else if(policy==='fixed-split') current='split';
    else {
      const best=names.reduce((a,b)=>cost(a,observed[t])<cost(b,observed[t])?a:b);
      const gain=(cost(current,observed[t])-cost(best,observed[t]))/cost(current,observed[t]);
      const supported=best!==current && gain>=cfg.h;
      if(supported) {if(candidate===best)consecutive++;else {candidate=best;consecutive=1;}}
      else {candidate=null;consecutive=0;}
      if(supported && consecutive>=cfg.k && t-lastSwitch>=cfg.d) {
        current=best;lastSwitch=t;switches++;changed=true;candidate=null;consecutive=0;
      }
    }
    values.push(Math.max(0,cost(current,actual[t])+jitter[t]+(changed?SWITCH_COST_MS:0)));
    if(trace==='step' && t>=stepAt && firstStable===null && current==='split') firstStable=t-stepAt;
  }
  return {trace,rep,policy,n:N,mean_ms:values.reduce((a,b)=>a+b,0)/N,p95_ms:percentile(values,.95),p99_ms:percentile(values,.99),miss_pct:100*values.filter(v=>v>DEADLINE_MS).length/N,switches_per_min:switches/(N*EPOCH_SECONDS/60),adapt_delay_s:firstStable};
}
const runs=[];
for(const trace of ['steady','step','bursty','periodic'])for(let rep=0;rep<REPS;rep++)for(const policy of policyNames)runs.push(simulate(trace,rep,policy));
function csv(rows,fields) {return [fields.join(','),...rows.map(r=>fields.map(f=>r[f]===null?'':typeof r[f]==='number'?r[f].toFixed(4):r[f]).join(','))].join('\n')+'\n';}
const fields=['trace','rep','policy','n','mean_ms','p95_ms','p99_ms','miss_pct','switches_per_min','adapt_delay_s'];
fs.writeFileSync(path.join(OUT,'simulation_runs.csv'),csv(runs,fields));
const summary=[];
for(const trace of ['steady','step','bursty','periodic'])for(const policy of policyNames) {
  const a=runs.filter(r=>r.trace===trace&&r.policy===policy);
  const avg=k=>a.reduce((s,r)=>s+r[k],0)/a.length;
  const delays=a.map(r=>r.adapt_delay_s).filter(v=>v!==null);
  const adaptive=!['edge-only','cloud-only','fixed-split'].includes(policy);
  summary.push({trace,policy,repetitions:a.length,mean_p95_ms:avg('p95_ms'),mean_p99_ms:avg('p99_ms'),mean_miss_pct:avg('miss_pct'),mean_switches_per_min:avg('switches_per_min'),mean_adapt_delay_s:trace==='step'&&adaptive&&delays.length===a.length?delays.reduce((x,y)=>x+y,0)/delays.length:null});
}
fs.writeFileSync(path.join(OUT,'simulation_summary.csv'),csv(summary,['trace','policy','repetitions','mean_p95_ms','mean_p99_ms','mean_miss_pct','mean_switches_per_min','mean_adapt_delay_s']));
const paired=[];
for(const trace of ['steady','step','bursty','periodic'])for(const metric of ['p95_ms','miss_pct','switches_per_min']) {
  const diffs=[];
  for(let rep=0;rep<REPS;rep++) {
    const hkd=runs.find(r=>r.trace===trace&&r.rep===rep&&r.policy==='HKD');
    const greedy=runs.find(r=>r.trace===trace&&r.rep===rep&&r.policy==='greedy');
    diffs.push(hkd[metric]-greedy[metric]);
  }
  const mean=diffs.reduce((a,b)=>a+b,0)/REPS;
  const sd=Math.sqrt(diffs.reduce((a,b)=>a+(b-mean)**2,0)/(REPS-1));
  const half=1.96*sd/Math.sqrt(REPS);
  paired.push({trace,metric,n:REPS,hkd_minus_greedy:mean,ci_low:mean-half,ci_high:mean+half});
}
fs.writeFileSync(path.join(OUT,'simulation_paired_effects.csv'),csv(paired,['trace','metric','n','hkd_minus_greedy','ci_low','ci_high']));
console.log(summary.filter(r=>['greedy','HKD','edge-only','fixed-split'].includes(r.policy)).map(r=>`${r.trace.padEnd(8)} ${r.policy.padEnd(12)} P95=${r.mean_p95_ms.toFixed(1)} miss=${r.mean_miss_pct.toFixed(1)}% switches/min=${r.mean_switches_per_min.toFixed(1)} delay=${r.mean_adapt_delay_s===null?'--':r.mean_adapt_delay_s.toFixed(1)}`).join('\n'));
console.log('\nPaired HKD minus greedy (95% normal intervals across replications):');
console.log(paired.map(r=>`${r.trace} ${r.metric}: ${r.hkd_minus_greedy.toFixed(2)} [${r.ci_low.toFixed(2)}, ${r.ci_high.toFixed(2)}]`).join('\n'));
