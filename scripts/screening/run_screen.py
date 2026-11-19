"""Run/resume six matched-budget screening conditions; no online training."""
from pathlib import Path
import os,sys,json,time,random,hashlib,gc,argparse,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/baseline'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ['OFFLINE_RL_DAVIS_PATH']=os.environ.get('OFFLINE_RL_DAVIS_PATH',r'E:\offline_rl_smoke_test\data\DAVIS\JPEGImages\480p')
os.environ['MUJOCO_GL']='glfw';os.environ['WANDB_MODE']='disabled'
import numpy as np,torch,psutil
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from src.utils import ReplayBuffer,load_offline_dataset_into_buffer,set_seed_everywhere
from agents import ScreeningAgent
import published_comparator as pc
from evaluate_study import episode

DEST=ROOT/'experiments/screening_20261106'
MODES=['bc','drq','iql','awac','svea','infogating']
SEEDS=[0,1,2]
UPDATES=4000

def rng_state():return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
def restore_rng(r):
    random.setstate(r[0]);np.random.set_state(r[1]);torch.set_rng_state(r[2]);torch.cuda.set_rng_state_all(r[3])
def build(mode,seed):
    set_seed_everywhere(seed)
    if mode=='infogating':
        a=pc.mod.InfoGatingAgent((9,84,84),(6,),'cuda',1e-4,256,1024,.01,2000,2,'linear(1.0, 0.1, 250000)',.3,True,lam=.1)
        a.mode=mode;return a
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs/upstream')):
        cfg=compose(config_name='config',overrides=['algo=drqv2','task=cheetah_run_medium_expert_easy','batch_size=16','gpu=0','no_track=True'])
    cfg.algo.agent._target_='agents.ScreeningAgent'
    cfg.algo.agent.obs_dim=[9,84,84];cfg.algo.agent.action_dim=[6];cfg.algo.agent.device='cuda:0'
    return instantiate(cfg.algo.agent).configure(mode)
def pack(a,step,seconds):
    return {'step':step,'training_seconds':seconds,'mode':a.mode,
        'models':{k:v.state_dict() for k,v in vars(a).items() if isinstance(v,torch.nn.Module)},
        'optimizers':{k:v.state_dict() for k,v in vars(a).items() if isinstance(v,torch.optim.Optimizer)},'rng':rng_state()}
def save(a,path,step,seconds):
    temp=path.with_suffix('.tmp');torch.save(pack(a,step,seconds),temp);os.replace(temp,path)
def load(a,path):
    c=torch.load(path,weights_only=False)
    assert c['mode']==a.mode
    for group in ['models','optimizers']:
        for k,v in c[group].items():getattr(a,k).load_state_dict(v)
    restore_rng(c['rng']);return c
def update(a,replay,it,step):
    return a.update(it,step*2) if a.mode=='infogating' else a.update(replay,step*2,16)
def finite(a,metrics):
    assert all(np.isfinite(float(v)) for v in metrics.values()),metrics
    assert all(torch.isfinite(p).all().item() for v in vars(a).values() if isinstance(v,torch.nn.Module) for p in v.parameters())
def write_json(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));os.replace(temp,path)

def main(validate=False):
    DEST.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    protocol={'date':'2026-11-06','purpose':'Exploratory small-budget screening, not paper reproduction or definitive ranking',
        'methods':MODES,'training_seeds':SEEDS,'updates_per_run':UPDATES,'batch_size':16,
        'dataset':str(ROOT/'data/cheetah_medium_expert_easy'),'transitions':100000,'episodes':200,
        'evaluation_seeds':[701,702,703],'evaluation_conditions':['familiar_background','heldout_background'],
        'evaluation':'Same static-within-episode easy protocol as earlier six-method pilot (dynamic=False default); frozen final policies, raw returns, no training from simulator',
        'budget':'Equal optimizer iterations, NOT equal runtime or model size. InfoGating ignores rewards and has different encoder/policy.',
        'infogating_warmup':'Author threshold step<5000 retained: first 2499 iterations warmup, subsequent 1501 full-gate iterations.',
        'checkpoint_every_updates':250,'warmstart':False,
        'primary_ranking':'Mean held-out return, first average 3 episodes within each training seed, then mean and sample SD over 3 training seeds.',
        'caveat':'These evaluation videos/seeds were examined in earlier studies; exploratory development benchmark, not untouched confirmatory test.'}
    pp=DEST/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol,'Protocol changed; use a new experiment folder.'
    else:write_json(pp,protocol)
    print('LOADING DATA',flush=True)
    replay=ReplayBuffer(102000,3,.99,3)
    load_offline_dataset_into_buffer([ROOT/'data/cheetah_medium_expert_easy'],replay,3,102000,verbose=False)
    assert not replay.full
    episodes=int((replay.step_type[:len(replay)]==2).sum());assert episodes==200,episodes
    it=pc.Iterator(replay)
    write_json(DEST/'data_check.json',{'replay_rows':len(replay),'complete_episodes':episodes,'valid_sampling_indices':int(replay.valid.sum()),
        'source_manifest_sha256':hashlib.sha256((ROOT/'reports/data_manifest.json').read_bytes()).hexdigest(),
        'files':[{'name':p.name,'bytes':p.stat().st_size} for p in sorted((ROOT/'data/cheetah_medium_expert_easy').glob('*.hdf5'))]})
    print('DATA READY',len(replay),episodes,flush=True)
    if validate:
        from agents import expectile_loss
        assert abs(expectile_loss(torch.tensor([-2.,2.])).item()-2.)<1e-6
        checks=[]
        for mode in MODES:
            a=build(mode,90);a.train(True)
            initial=next(a.encoder.parameters()).detach().clone()
            update(a,replay,it,3000);finite(a,{})
            assert not torch.equal(initial,next(a.encoder.parameters()))
            cp=DEST/'validation.pt';save(a,cp,3000,0.)
            metrics=update(a,replay,it,3001);finite(a,metrics)
            expected={k:{n:t.detach().clone() for n,t in v.state_dict().items()} for k,v in vars(a).items() if isinstance(v,torch.nn.Module)}
            load(a,cp);again=update(a,replay,it,3001)
            for k,vals in expected.items():
                for n,t in vals.items():torch.testing.assert_close(getattr(a,k).state_dict()[n],t,rtol=0,atol=0)
            a.train(False)
            x=next(it)[0][0]
            with torch.no_grad():action=a.act(x,100000,True) if mode=='infogating' else a.act_with_logprob(x/255.,100000,True)[0]
            assert action.shape==(6,) and np.isfinite(action).all() and np.max(np.abs(action))<=1.00001
            checks.append({'mode':mode,'finite_update':True,'encoder_changed':True,'exact_resume_next_update':True,'valid_action':True})
            print('VALIDATED',mode,flush=True);del a,expected;gc.collect();torch.cuda.empty_cache()
        cp.unlink()
        write_json(DEST/'validation.json',{'passed':True,'checks':checks})
        return
    assert json.loads((DEST/'validation.json').read_text())['passed']
    for seed in SEEDS:
      for mode in MODES:
        run=DEST/'runs'/f'{mode}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
        if (run/'result.json').exists():continue
        print('START',mode,seed,flush=True)
        a=build(mode,seed);a.train(True);step=0;elapsed=0.;cp=run/'latest.pt'
        if cp.exists():
            c=load(a,cp);step=c['step'];elapsed=c['training_seconds']
        torch.cuda.reset_peak_memory_stats();peak=psutil.Process().memory_info().rss
        with (run/'metrics.jsonl').open('a') as log:
            while step<UPDATES:
                step+=1;t=time.perf_counter();metrics=update(a,replay,it,step);torch.cuda.synchronize();elapsed+=time.perf_counter()-t
                if step%100==0:
                    finite(a,metrics);peak=max(peak,psutil.Process().memory_info().rss)
                    log.write(json.dumps({'step':step,'training_seconds':elapsed,**metrics})+'\n');log.flush()
                if step%250==0:
                    save(a,cp,step,elapsed);print('TRAIN',mode,seed,step,round(elapsed,1),flush=True)
        save(a,cp,step,elapsed);a.train(False)
        ev=[];ef=run/'evaluation.jsonl'
        if ef.exists():ev=[json.loads(x) for x in ef.read_text().splitlines() if x.strip()]
        for condition in protocol['evaluation_conditions']:
          for es in protocol['evaluation_seeds']:
            if any(r['condition']==condition and r['eval_seed']==es for r in ev):continue
            set_seed_everywhere(10000+es)
            result=episode(a,mode,condition,es)
            assert result['steps']==500 and np.isfinite(result['return']),result
            with ef.open('a') as f:f.write(json.dumps(result)+'\n')
            ev.append(result);print('EVAL',mode,seed,condition,es,round(result['return'],3),flush=True)
        write_json(run/'result.json',{'mode':mode,'seed':seed,'updates':step,'training_seconds':elapsed,
            'peak_rss_bytes':peak,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'evaluation':ev})
        del a;gc.collect();torch.cuda.empty_cache()
    print('SCREENING COMPLETE',flush=True)

if __name__=='__main__':
    try:main('--validate' in sys.argv)
    except Exception:
        DEST.mkdir(parents=True,exist_ok=True)
        with (DEST/'failures.log').open('a') as f:f.write(traceback.format_exc()+'\n')
        raise
