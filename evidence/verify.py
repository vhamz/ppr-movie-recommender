"""Independent SciPy/NumPy oracle, including all shared users and four alphas."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy import sparse,linalg
from collections import Counter
work=Path(__file__).resolve().parent
root=work.parent/'ppr-movie-recommender'
if not root.exists():root=work.parent
t0=time.perf_counter()
ratings=np.loadtxt(root/'u.data',dtype=np.int64)
train=[];val=[];test=[]
for uid in sorted(set(ratings[:,0])):
 rows=ratings[ratings[:,0]==uid];rows=rows[np.lexsort((rows[:,1],rows[:,3]))]
 nt=int(.8*len(rows));nv=int(.1*len(rows))
 train.extend(rows[:nt]);val.extend(rows[nt:nt+nv]);test.extend(rows[nt+nv:])
train,val,test=map(np.array,[train,val,test])
users=np.unique(train[:,0]);items=np.unique(train[:,1]);U=len(users);M=len(items);N=U+M
uindex={int(u):i for i,u in enumerate(users)};iindex={int(i):j for j,i in enumerate(items)}
pos=train[train[:,2]>=4];edges={(uindex[int(r[0])],iindex[int(r[1])]) for r in pos}
ei=np.array(sorted(edges));R=sparse.csr_matrix((np.ones(len(edges)),(ei[:,0],ei[:,1])),shape=(U,M))
du=np.asarray(R.sum(axis=1)).ravel();dm=np.asarray(R.sum(axis=0)).ravel();degree=np.r_[du,dm]
inv=np.divide(1.,degree,out=np.zeros(N),where=degree>0)
A=sparse.bmat([[None,R],[R.T,None]],format='csr');Q=sparse.diags(inv)@A
row_sums=np.asarray(Q.sum(axis=1)).ravel();assert np.max(np.abs(row_sums[degree>0]-1))<1e-12
seen_train={int(u):set(train[train[:,0]==u,1]) for u in users}
seen_val={int(u):set(val[val[:,0]==u,1]) for u in users}
pool=set(items);positive_users=set(pos[:,0]);cohorts={};relevant={};exclusions={}
for name,rows in [('val',val),('test',test)]:
 rel={};unseen=[];no_positive=[];no_relevant=[]
 for u in users:
  held=rows[(rows[:,0]==u)&(rows[:,2]>=4)]
  unseen.extend(int(i) for i in held[:,1] if i not in pool)
  seen=seen_train[int(u)]|(seen_val[int(u)] if name=='test' else set())
  rel[int(u)]=set(held[:,1])&pool-seen
  if u not in positive_users:no_positive.append(int(u))
  if not rel[int(u)]:no_relevant.append(int(u))
 cohorts[name]=[int(u) for u in users if u in positive_users and rel[int(u)]]
 relevant[name]=rel
 exclusions[name]={'cohort':len(cohorts[name]),'unseenRatings':len(unseen),'unseenDistinct':len(set(unseen)),
  'noPositive':no_positive,'noRelevant':no_relevant,'excludedUnion':len(set(no_positive)|set(no_relevant))}
saved=json.loads((root/'mf-checkpoint.json').read_text());w=saved['weights']
p=np.array(w['users'],np.float64).reshape(w['uShape']);q=np.array(w['items'],np.float64).reshape(w['iShape'])
mf=np.clip(saved['mean']+p[users]@q[items].T,1,5)
def recs(scores,u,name):
 seen=seen_train[u]|(seen_val[u] if name=='test' else set())
 eligible=np.array([j for j,i in enumerate(items) if i not in seen])
 order=np.lexsort((items[eligible],-scores[eligible]))[:10]
 return [int(i) for i in items[eligible[order]]]
def metric(scores,name):
 per=[];mean_pop=[];lists={}
 for u in cohorts[name]:
  vec=scores[uindex[u]] if scores.ndim==2 else scores
  ids=recs(vec,u,name);rel=relevant[name][u];hit=len(set(ids)&rel)
  per.append({'userId':u,'hit':hit,'relevant':len(rel),'recall':hit/len(rel)})
  mean_pop.append(np.mean([dm[iindex[i]] for i in ids]));lists[u]=ids
 return {'recall':float(np.mean([r['recall'] for r in per])),'meanPopularity':float(np.mean(mean_pop)),'perUser':per,'lists':lists}
allPpr={};sweeps=[];sampleVectors=[]
# For positive seeds, disconnected zero-degree nodes never receive mass.
# Eliminate movie probabilities to solve an independent user-block system:
# (I - alpha^2 R D_m^-1 R^T D_u^-1) p_u = (1-alpha) e_u.
inv_u=inv[:U];inv_m=inv[U:]
S=(R@sparse.diags(inv_m)@R.T@sparse.diags(inv_u)).toarray()
for alpha in [.5,.7,.85,.95]:
 pu=linalg.solve(np.eye(U)-alpha**2*S,(1-alpha)*np.eye(U),assume_a='gen')
 pm=alpha*(R.T@(inv_u[:,None]*pu))
 scores=pm.T
 allPpr[alpha]=scores
 sweep=metric(scores,'val');sweeps.append({'alpha':alpha,'recall':sweep['recall']})
 print('REFERENCE_ALPHA',alpha,sweep['recall'],flush=True)
 for u in cohorts['test'][:2]:
  col=uindex[u];vector=np.r_[pu[:,col],pm[:,col]]
  nxt=alpha*(Q.T@vector)+(1-alpha)*np.eye(1,N,col).ravel()
  assert np.sum(np.abs(nxt-vector))<1e-11
  sampleVectors.append({'alpha':alpha,'userId':u,'scores':vector.tolist()})
best=sorted(sweeps,key=lambda r:(-r['recall'],r['alpha']))[0]['alpha']
v=np.ones(N)/N;pg=v.copy();mask=degree==0
for it in range(10000):
 nxt=best*(Q.T@pg+pg[mask].sum()*v)+(1-best)*v
 if np.abs(nxt-pg).sum()<1e-13:pg=nxt;break
 pg=nxt
methods={'popularity':metric(dm,'test'),'pagerank':metric(pg[U:],'test'),
 'ppr':metric(allPpr[best],'test'),'mf':metric(mf,'test')}
result={'split':{'train':len(train),'val':len(val),'test':len(test)},'graph':{'users':U,'movies':M,'nodes':N,'positiveEdges':len(edges),
 'zeroUsers':[int(u) for u,d in zip(users,du) if d==0],'zeroMovies':[int(i) for i,d in zip(items,dm) if d==0]},
 'sparsity':1-len(ratings)/(len(np.unique(ratings[:,0]))*1682),'exclusions':exclusions,'sweep':sweeps,'selectedAlpha':best,
 'methods':methods,'rowNormalizationMaxError':float(np.max(np.abs(row_sums[degree>0]-1))),
 'hashes':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['u.data','u.item']}}
measured=work/'experiment.json' if (work/'experiment.json').exists() else work/'results.json'
if measured.exists():
 js=json.loads(measured.read_text());assert js['selectedAlpha']==best
 checks=[]
 for row in js['comparison']:
  py=methods[row['method']];diff=abs(row['recall']-py['recall'])
  checks.append({'method':row['method'],'recallDifference':diff,'meanPopularityDifference':abs(row['meanPopularity']-py['meanPopularity'])})
  assert diff<1e-10,(row['method'],row['recall'],py['recall'])
 for row in js['sweep']:assert abs(row['recall']-next(r['recall'] for r in sweeps if r['alpha']==row['alpha']))<1e-10
 max_vector_error=0
 for sample in js['sampleVectors']:
  py=next(x for x in sampleVectors if x['userId']==sample['userId'] and x['alpha']==sample['alpha'])
  error=float(np.max(np.abs(np.array(py['scores'])-sample['scores'])))
  max_vector_error=max(max_vector_error,error);assert error<2e-6
 for row in js['lists']:
  ids=[r['movieId'] for r in row['recs']];assert ids==methods[row['method']]['lists'][row['userId']],(row['userId'],row['method'])
 result['crossCheck']={'metrics':checks,'maxSampleVectorError':max_vector_error,'allDisplayedListsMatch':True}
# Paired user bootstrap: descriptive uncertainty conditional on this split.
rng=np.random.default_rng(20261021)
ppr_r=np.array([r['recall'] for r in methods['ppr']['perUser']]);mf_r=np.array([r['recall'] for r in methods['mf']['perUser']])
indices=rng.integers(len(ppr_r),size=(5000,len(ppr_r)))
diff=(ppr_r-mf_r)[indices].mean(1)
result['pairedBootstrap']={'replicates':5000,'seed':20261021,'meanDifference':float(np.mean(ppr_r-mf_r)),
 'ci95':np.quantile(diff,[.025,.975]).tolist()}
result['seconds']=time.perf_counter()-t0
(work/'independent-verification.json').write_text(json.dumps(result,default=lambda x:x.item()))
print(json.dumps({k:v for k,v in result.items() if k!='methods'},default=lambda x:x.item(),indent=2))
print('METRICS',{k:(v['recall'],v['meanPopularity']) for k,v in methods.items()})
