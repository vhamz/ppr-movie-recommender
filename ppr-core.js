/**
 * TensorFlow.js power iteration shared by global and personalized PageRank.
 * Node order: contiguous training users, then contiguous training movies.
 * Probability matrices have shape [nodes, restart columns]. No padding node.
 * The MovieLens kernel uses the two R blocks and never builds a square tensor.
 */

/** Create disposable tensors and one propagation operator. */
function graphPowerState(graph, restartNodes, alpha, opts = {}) {
  if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
    throw new Error('alpha must be a link-following probability in [0, 1].');
  }
  const spec = opts.transition;
  const n = spec ? spec.Q.length : graph && graph.nodeCount;
  if (!n) throw new Error('A nonempty graph or explicit transition is required.');
  const nodes = restartNodes;
  if (nodes && nodes.some(i => !Number.isInteger(i) || i < 0 || i >= n)) {
    throw new Error('Unknown restart node index: ' + nodes.join(','));
  }
  const width = nodes ? nodes.length : 1;
  if (!width) throw new Error('At least one restart column is required.');
  const tensors = tf.tidy(() => {
    const v = nodes ? tf.oneHot(tf.tensor1d(nodes, 'int32'), n).transpose()
      : tf.fill([n, 1], 1 / n);
    const mask = spec ? Array.from(spec.danglingMask, Number)
      : Array.from(graph.degree, d => d === 0 ? 1 : 0);
    const dangling = tf.tensor1d(mask).expandDims(1);
    if (spec) return {v, dangling, Q: tf.tensor2d(spec.Q)};
    return {v, dangling, R: tf.tensor2d(graph.R, [graph.nUsers, graph.nMovies]),
      invU: tf.tensor1d(Array.from(graph.degree.slice(0, graph.nUsers), d => d ? 1/d : 0)).expandDims(1),
      invM: tf.tensor1d(Array.from(graph.degree.slice(graph.nUsers), d => d ? 1/d : 0)).expandDims(1)};
  });
  const step = p => tf.tidy(() => {
    let propagated;
    if (spec) propagated = tf.matMul(tensors.Q, p, true, false);
    else {
      const u = p.slice([0, 0], [graph.nUsers, width]).mul(tensors.invU);
      const m = p.slice([graph.nUsers, 0], [graph.nMovies, width]).mul(tensors.invM);
      propagated = tf.concat([tf.matMul(tensors.R, m), tf.matMul(tensors.R, u, true, false)], 0);
    }
    const danglingMass = p.mul(tensors.dangling).sum(0);
    return propagated.mul(alpha).add(tensors.v.mul(danglingMass.mul(alpha).add(1-alpha)));
  });
  return {n, width, tensors, step, p:tensors.v.clone(),
    tol:opts.tol === undefined ? 1e-6 : opts.tol,
    maxIter:opts.maxIter === undefined ? 500 : opts.maxIter};
}

/** Synchronous one-column path for the supplied harness and interactive walks. */
function graphPowerSync(graph, restartNode, alpha, opts = {}) {
  const state = graphPowerState(graph, restartNode === null ? null : [restartNode], alpha, opts);
  let residual = Infinity, iterations = 0;
  try {
    for (let i = 1; i <= state.maxIter; i++) {
      const next = state.step(state.p);
      residual = tf.tidy(() => next.sub(state.p).abs().sum().dataSync()[0]);
      state.p.dispose(); state.p = next; iterations = i;
      if (residual < state.tol) break;
    }
    const p = Float64Array.from(state.p.dataSync());
    const extraResidual = tf.tidy(() => state.step(state.p).sub(state.p).abs().sum().dataSync()[0]);
    return {p, iterations, residual, converged:residual < state.tol,
      mass:p.reduce((sum, x) => sum+x, 0), extraResidual};
  } finally {state.p.dispose(); tf.dispose(state.tensors);}
}

/**
 * Asynchronous batched PPR. Each column stops on its own first L1 < tolerance.
 * Async reads happen outside tf.tidy. Yielding lets progress updates paint.
 */
async function graphPowerBatch(graph, restartNodes, alpha, opts = {}) {
  const state = graphPowerState(graph, restartNodes, alpha, opts);
  const done = new Array(state.width).fill(false);
  const iterations = new Array(state.width).fill(0);
  const residuals = new Array(state.width).fill(Infinity);
  const masses = new Array(state.width).fill(1);
  let maxMassError = 0, minScore = Infinity;
  try {
    for (let iteration = 1; iteration <= state.maxIter; iteration++) {
      const next = state.step(state.p);
      const metrics = tf.tidy(() => tf.stack([
        next.sub(state.p).abs().sum(0), next.sum(0), next.min(0)
      ]));
      const data = await metrics.data(); metrics.dispose();
      const previouslyDone = done.slice();
      for (let j = 0; j < state.width; j++) {
        if (done[j]) continue;
        iterations[j] = iteration; residuals[j] = data[j];
        masses[j] = data[state.width+j];
        maxMassError = Math.max(maxMassError, Math.abs(masses[j]-1));
        minScore = Math.min(minScore, data[2*state.width+j]);
        if (residuals[j] < state.tol) done[j] = true;
      }
      const kept = tf.tidy(() => {
        const frozen = tf.tensor1d(previouslyDone, 'bool').reshape([1,state.width]);
        return tf.where(frozen, state.p, next);
      });
      state.p.dispose(); next.dispose(); state.p = kept;
      if (opts.onProgress && iteration % 8 === 0) opts.onProgress(iteration, done.filter(Boolean).length, state.width);
      if (done.every(Boolean)) break;
      if (tf.getBackend() === "cpu" || iteration % 4 === 0) await new Promise(resolve => setTimeout(resolve, 0));
    }
    const extra = tf.tidy(() => state.step(state.p).sub(state.p).abs().sum(0));
    const extraData = await extra.data(); extra.dispose();
    const all = await state.p.data();
    return (restartNodes || [null]).map((node, j) => ({
      p:Float64Array.from({length:state.n}, (_, i) => all[i*state.width+j]),
      iterations:iterations[j], residual:residuals[j], converged:done[j],
      mass:masses[j], extraResidual:extraData[j], maxMassError, minScore
    }));
  } finally {state.p.dispose(); tf.dispose(state.tensors);}
}

/** Load the selected HW5 checkpoint, never train a different MF reference. */
async function hw6LoadPreviousMf() {
  const response = await fetch('mf-checkpoint.json');
  if (!response.ok) throw new Error('Cannot load the saved HW5 MF checkpoint: HTTP '+response.status);
  const saved = await response.json(), w = saved.weights;
  if (saved.counts.train !== HW6_SPLIT.train.length || saved.counts.val !== HW6_SPLIT.val.length ||
      saved.counts.test !== HW6_SPLIT.test.length) throw new Error('HW5 split counts differ.');
  if (w.uShape[0] <= Math.max(...HW6_STATE.trainUserIds) ||
      w.iShape[0] <= Math.max(...HW6_STATE.candidateIds)) throw new Error('MF cannot score all shared candidates.');
  hw6DisposeMf();
  HW6_MF_MODEL = tf.tidy(() => ({userFactors:tf.variable(tf.tensor2d(w.users,w.uShape)),
    itemFactors:tf.variable(tf.tensor2d(w.items,w.iShape)),globalMean:saved.mean}));
  Object.assign(MF_INFO,{factors:w.uShape[1],epochs:saved.opts.epochs,
    learningRate:saved.opts.lr,regularization:saved.opts.reg,batchSize:saved.opts.batchSize,
    seed:'HW5 initialization 42/43; shuffle 2026',backend:saved.backend,
    checkpoint:saved.best.epoch,validationRMSE:saved.best.valRMSE,
    trainedAt:'2026-10-14',history:saved.history,source:'Saved HW5 checkpoint; no HW6 retraining',
    scoring:'clip(globalMean + dot(user, movie), 1, 5)'});
  return HW6_MF_MODEL;
}

/** Prepare all shared-user walks once per alpha, cached by graph and node. */
async function hw6PreparePpr(alphas) {
  const graph = hw6ActiveGraph();
  if (!graph) throw new Error('Build the graph before running evaluation.');
  if (graph._preparing) return graph._preparing;
  const users = Array.from(new Set([...hw6SharedCohort('val'),...hw6SharedCohort('test')])).sort((a,b)=>a-b);
  graph._pprCache = graph._pprCache || new Map();
  graph._batchRuns = graph._batchRuns || [];
  const controls = Array.from(document.querySelectorAll('button,input,select')).map(el=>({el,disabled:el.disabled}));
  for (const {el} of controls) el.disabled = true;
  graph._preparing = (async () => {
    for (const alpha of alphas) {
      if (!graph._pprCache.has(alpha)) graph._pprCache.set(alpha,new Map());
      const cache = graph._pprCache.get(alpha);
      const missing = users.filter(id=>!cache.has(graph.nodeOfUser(id)));
      // The full batch uses matrix multiplication efficiently on WebGL and
      // remains small (2,554 x ~930 floats). Split to bound device memory.
      const batchSize = tf.getBackend() === "cpu" ? 16 : 128;
      for (let start = 0; start < missing.length; start += batchSize) {
        const ids = missing.slice(start,start+batchSize), nodes = ids.map(id=>graph.nodeOfUser(id));
        hw6SetStatus('Computing PPR: alpha '+alpha.toFixed(2)+', users '+(start+1)+'-'+Math.min(start+batchSize,missing.length)+' / '+missing.length+'...');
        console.log('PPR_PROGRESS',alpha,start,missing.length);
        await tf.nextFrame();
        const before = tf.memory().numTensors, t0 = performance.now();
        const results = await graphPowerBatch(graph,nodes,alpha,{onProgress:(iteration,done,total)=>hw6SetStatus('PPR alpha '+alpha.toFixed(2)+': users '+(start+1)+'–'+Math.min(start+batchSize,missing.length)+' / '+missing.length+', iteration '+iteration+'; converged '+done+'/'+total)});
        const fallback=[];
        const stalled=results.map((r,j)=>r.converged?-1:j).filter(j=>j>=0);
        if(stalled.length&&tf.getBackend()!=='cpu'){
          // Float32 WebGL may enter a two-cycle just above the requested L1
          // tolerance at high alpha. Retry only those seeds from p0=v using
          // TF.js CPU accumulation, with the SAME tolerance and iteration cap.
          const backend=tf.getBackend();
          hw6SetStatus('Checking '+stalled.length+' high-alpha walks on the TensorFlow.js CPU backend...');
          try {
            await tf.setBackend('cpu');await tf.ready();
            const retried=await graphPowerBatch(graph,stalled.map(j=>nodes[j]),alpha,{onProgress:(iteration,done,total)=>hw6SetStatus('CPU precision check, alpha '+alpha+': iteration '+iteration+'; converged '+done+'/'+total)});
            for(let k=0;k<stalled.length;k++){
              const j=stalled[k];
              fallback.push({userId:ids[j],initialBackend:backend,initialIterations:results[j].iterations,
                initialResidual:results[j].residual,finalBackend:'cpu',finalResidual:retried[k].residual,
                finalIterations:retried[k].iterations,converged:retried[k].converged});
              results[j]={...retried[k],backend:'cpu'};
            }
          }finally{await tf.setBackend(backend);await tf.ready();}
        }
        results.forEach((r,j)=>cache.set(nodes[j],r));
        const after = tf.memory().numTensors;
        graph._batchRuns.push({alpha,users:ids,seconds:(performance.now()-t0)/1000,
          tensorsBefore:before,tensorsAfter:after,maxMassError:Math.max(...results.map(r=>r.maxMassError)),
          minScore:Math.min(...results.map(r=>r.minScore)),fallback});
        if (after !== before) throw new Error('PPR batch leaked tensors: '+before+' -> '+after);
        if (results.some(r=>!r.converged)) {
          console.log('PPR_NOT_CONVERGED',JSON.stringify(results.map((r,j)=>({userId:ids[j],residual:r.residual,mass:r.mass,converged:r.converged})).filter(r=>!r.converged)));
          throw new Error('PPR hit the iteration cap; inspect convergence before reporting metrics.');
        }
      }
    }
  })();
  try { await graph._preparing; }
  finally {graph._preparing = null; for (const {el,disabled} of controls) el.disabled=disabled;}
}

/** Additional checks beyond the supplied harness, using the actual kernel. */
async function hw6ExtraChecks() {
  const graph = hw6ActiveGraph(), before = tf.memory().numTensors;
  const fixture = {Q:[[0,1,0],[1,0,0],[0,0,0]],danglingMask:[0,0,1]};
  const capped = graphPowerSync(null,0,.95,{transition:fixture,maxIter:1});
  if (capped.converged) throw new Error('Iteration cap falsely marked converged.');
  let unknownThrows=false;
  try { personalizedPageRank(graph.nodeCount+99,.85); } catch (_) {unknownThrows=true;}
  if (!unknownThrows) throw new Error('Unknown restart node was accepted.');
  const seedA=graph.nodeOfUser(1), seedB=graph.nodeOfUser(2);
  const single = graphPowerSync(graph,seedA,.5);
  const batch = await graphPowerBatch(graph,[seedA,seedB],.5);
  const repeat = await graphPowerBatch(graph,[seedA,seedB],.5);
  let maxSingleBatch=0, maxRepeat=0;
  for(let i=0;i<graph.nodeCount;i++){
    maxSingleBatch=Math.max(maxSingleBatch,Math.abs(single.p[i]-batch[0].p[i]));
    maxRepeat=Math.max(maxRepeat,Math.abs(batch[0].p[i]-repeat[0].p[i]));
  }
  const indexResult=hw6RunPPR(graph,1,.5);
  const maxSeedIndexError=Math.max(...Array.from(single.p,(v,i)=>Math.abs(v-indexResult.scores[i])));
  const filtering={valOnlyEligible:false,testExcluded:true};
  const valIds=HW6_STATE.pairsBySplit.val.get(1), trainIds=HW6_STATE.pairsBySplit.train.get(1);
  for(const id of valIds||[])if(!trainIds.has(id)&&HW6_STATE.candidateSet.has(id)){
    filtering.valOnlyEligible=!hw6ExclusionSet('val',1).has(id);
    filtering.testExcluded=hw6ExclusionSet('test',1).has(id);break;
  }
  const after=tf.memory().numTensors;
  if(after!==before||maxSingleBatch>2e-6||maxRepeat>1e-7||maxSeedIndexError>2e-6||
    !filtering.valOnlyEligible||!filtering.testExcluded)throw new Error('Extra verification failed.');
  return {cappedConverged:capped.converged,unknownThrows,maxSingleBatch,maxRepeat,maxSeedIndexError,
    filtering,tensorsBefore:before,tensorsAfter:after,batchRuns:graph._batchRuns,
    passed:true};
}


// UI actions precompute expensive walks asynchronously before the synchronous
// rendering/scoring adapters consume them. One action at a time avoids backend
// switches and graph mutations racing with an evaluation.
let hw6UiBusy = false;
async function hw6UiWalk(userId, alpha) {
  const graph = hw6ActiveGraph();
  const node = userId === null ? null : graph.nodeOfUser(userId);
  if (userId === null && HW6_STATE.pageRank?.graph === graph && HW6_STATE.pageRank.alpha === alpha) return;
  if (userId !== null && graph._pprCache?.get(alpha)?.has(node)) return;
  hw6SetStatus('Computing ' + (userId === null ? 'global PageRank' : 'PPR for user '+userId) + ' at alpha '+alpha+'…');
  await new Promise(resolve => setTimeout(resolve, 20));
  let result = (await graphPowerBatch(graph, node === null ? null : [node], alpha))[0];
  if (!result.converged && tf.getBackend() !== 'cpu') {
    const backend = tf.getBackend();
    try { await tf.setBackend('cpu'); result = (await graphPowerBatch(graph,node === null ? null : [node],alpha))[0]; }
    finally { await tf.setBackend(backend); }
  }
  if (!result.converged) throw new Error('Walk did not converge. Lower alpha and retry.');
  if (node === null) HW6_STATE.pageRank = {...hw6NormalizePropagation(result,graph.nodeCount),graph,alpha};
  else {
    graph._pprCache ||= new Map();
    if (!graph._pprCache.has(alpha)) graph._pprCache.set(alpha,new Map());
    graph._pprCache.get(alpha).set(node,result);
  }
}
async function hw6UiAction(button, handler, targetId) {
  if (hw6UiBusy) return;
  hw6UiBusy = true;
  const controls = Array.from(document.querySelectorAll('button,input,select')).map(el=>({el,disabled:el.disabled}));
  const label = button.textContent;
  controls.forEach(({el})=>el.disabled=true);
  button.textContent = label+' — running…';
  document.getElementById('action-status')?.setAttribute('aria-busy','true');
  hw6SetStatus(label+' — preparing…');
  try {
    await new Promise(resolve=>setTimeout(resolve,30));
    if (!HW6_STATE.loaded) throw new Error('Dataset is still loading. Please retry once loading finishes.');
    if (!hw6ActiveGraph()) hw6HandleBuildGraph();
    const alpha=hw6ReadAlpha(), method=hw6ReadMethod();
    const seed=hw6ReadNumber('seed-user-select',hw6ReadNumber('user-select',1));
    const user=hw6ReadNumber('user-select',1);
    if (handler===hw6HandleTwoUserDemo) {
      const users=hw6SharedCohort('test').slice(0,2);
      for(const a of [.85,.5]) for(const id of users) await hw6UiWalk(id,a);
    }
    if (handler===hw6HandleRunPageRank || (handler===hw6RenderRecommendations && method==='pagerank')) await hw6UiWalk(null,alpha);
    if (handler===hw6HandleRunPPR) await hw6UiWalk(seed,alpha);
    if (handler===hw6RenderRecommendations && method==='ppr') await hw6UiWalk(user,alpha);
    // Graph recolouring must also use a prepared score vector.
    if ([hw6HandleBuildGraph,hw6HandleRunPageRank,hw6HandleRunPPR].includes(handler)) {
      if(method==='ppr') await hw6UiWalk(seed,alpha);
      if(method==='pagerank') await hw6UiWalk(null,alpha);
    }
    await handler();
    if(handler===hw6RenderRecommendations) hw6SetStatus('Top-10 recommendations ready for user '+user+'.');
    const target=document.getElementById(targetId);
    if(target) target.scrollIntoView({behavior:'smooth',block:'center'});
  } catch(error) {
    console.error(error);
    hw6SetStatus(label+' failed: '+error.message);
  } finally {
    controls.forEach(({el,disabled})=>el.disabled=disabled);
    button.textContent=label;
    document.getElementById('action-status')?.setAttribute('aria-busy','false');
    hw6UiBusy=false;
  }
}


async function hw6RecallAsync(method, K, split) {
  const cohort=hw6SharedCohort(split),perUser=[];
  for(let i=0;i<cohort.length;i++) {
    const userId=cohort[i],excluded=hw6ExclusionSet(split,userId),relevant=new Set();
    HW6_STATE.gradesBySplit[split].get(userId).forEach((rating,movieId)=>{
      if(rating>=4&&HW6_STATE.candidateSet.has(movieId)&&!excluded.has(movieId)) relevant.add(movieId);
    });
    const recs=recommendTopK(method,userId,K,split);
    perUser.push({userId,relevant:relevant.size,hit:recs.reduce((n,r)=>n+Number(relevant.has(r.movieId)),0)});
    if(i%16===0) {
      hw6SetStatus('Evaluating '+method+' / '+split+' / alpha '+hw6ReadAlpha()+': user '+(i+1)+' of '+cohort.length+'…');
      await new Promise(resolve=>setTimeout(resolve,0));
    }
  }
  const excluded=hw6CohortExclusions(split);
  return {recall:perUser.length?perUser.reduce((n,r)=>n+r.hit/r.relevant,0)/perUser.length:0,
    cohort,perUser,excludedUnseen:excluded.excludedUnseen,excludedNoEligible:excluded.excludedNoEligible};
}
async function hw6MeanPopularityAsync(method,cohort,split,alpha) {
  let total=0,users=0;
  for(let i=0;i<cohort.length;i++) {
    const mean=hw6MeanPopularity(recommendTopK(method,cohort[i],HW6_K,split));
    if(mean!==null){total+=mean;users++;}
    if(i%16===0) await new Promise(resolve=>setTimeout(resolve,0));
  }
  return users?total/users:null;
}
async function hw6RunAlphaSweepAsync() {
  const rows=[];let best=null,bestRecall=-Infinity;
  const previous=hw6ReadAlpha();
  try {
    for(const alpha of ALPHA_GRID) {
      hw6SetAlpha(alpha);
      const report=await hw6RecallAsync('ppr',HW6_K,'val');
      rows.push({alpha,recall:report.recall,cohortSize:report.cohort.length,error:null});
      if(report.recall>bestRecall){bestRecall=report.recall;best=alpha;}
    }
  }finally{hw6SetAlpha(previous);}
  HW6_STATE.alphaSweepRows=rows;HW6_STATE.alphaSweepNote='';
  return {rows,best,note:''};
}
