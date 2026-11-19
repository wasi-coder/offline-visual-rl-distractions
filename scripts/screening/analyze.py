"""Aggregate only completed screen runs and export small auditable evidence."""
from pathlib import Path
import json,csv,statistics,sys,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'experiments/screening_20261106';O=ROOT/'results/screening_20261106'
LABEL={'bc':'BC','drq':'DrQv2+BC','iql':'Visual IQL*','awac':'Visual AWAC*','svea':'SVEA-style+BC*','infogating':'Adapted InfoGating*'}
def main():
    protocol=json.loads((D/'protocol.json').read_text())
    records=[];rows=[]
    for seed in protocol['training_seeds']:
      for mode in protocol['methods']:
        p=D/'runs'/f'{mode}_seed{seed}'/'result.json'
        if not p.exists():raise SystemExit('Incomplete screen: missing '+str(p))
        r=json.loads(p.read_text());assert r['updates']==protocol['updates_per_run']
        assert r['mode']==mode and r['seed']==seed
        ev=r['evaluation'];assert len(ev)==6
        assert len({(e['condition'],e['eval_seed']) for e in ev})==6
        assert {(e['condition'],e['eval_seed']) for e in ev}=={(c,s) for c in protocol['evaluation_conditions'] for s in protocol['evaluation_seeds']}
        assert all(e['steps']==500 for e in ev)
        records.append(r)
        for e in ev:rows.append({'method':mode,'training_seed':seed,**e})
    summary=[]
    for mode in protocol['methods']:
        runs=[r for r in records if r['mode']==mode]
        v={'method':mode,'label':LABEL[mode],'training_seeds':len(runs),
           'mean_training_seconds':statistics.mean(r['training_seconds'] for r in runs),
           'mean_peak_cuda_mib':statistics.mean(r['peak_cuda_bytes']/2**20 for r in runs)}
        for condition in protocol['evaluation_conditions']:
            means=[statistics.mean(e['return'] for e in r['evaluation'] if e['condition']==condition) for r in runs]
            v[condition]={'mean':statistics.mean(means),'sample_sd':statistics.stdev(means),'seed_means':means}
        summary.append(v)
    summary.sort(key=lambda r:r['heldout_background']['mean'],reverse=True)
    O.mkdir(parents=True,exist_ok=True)
    result={'complete':True,'runs':len(records),'evaluations':len(rows),
        'training_seconds_total':sum(r['training_seconds'] for r in records),'summary':summary}
    (O/'aggregate.json').write_text(json.dumps(result,indent=2))
    with (O/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    for name in ['protocol.json','validation.json','data_check.json']:
        shutil.copy2(D/name,O/name)
    for name in ['environment.json','data_hashes.json']:
        if (D/name).exists():shutil.copy2(D/name,O/name)
    shutil.copy2(ROOT/'reports/data_manifest.json',O/'source_data_manifest.json')
    (O/'run_results.json').write_text(json.dumps(records,indent=2))
    codefiles=list((ROOT/'scripts/screening').glob('*.py'))+[ROOT/'src/agents/drqv2_default.py',ROOT/'src/agents/base_agent.py',ROOT/'src/utils/replay_buffer.py',ROOT/'scripts/baseline/published_comparator.py',ROOT/'scripts/baseline/evaluate_study.py',ROOT/'references/agent-centric-representations-main/infogating.py']
    (O/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in codefiles},indent=2))
    top=summary[0]
    lines=['# Additional algorithms: small training comparison','',
      'Date: 6 November 2026. Status: all 18 training runs and 108 evaluation episodes completed.','',
      f"The highest observed held-out average was **{top['label']}**, at **{top['heldout_background']['mean']:.2f}**. This is a preliminary ranking under one short training protocol, not a reliable claim that this algorithm is best in general.",'',
      '## What was tested','',
      'Three new candidates were added: IQL (learn values using recorded actions), AWAC (copy better actions more strongly), and SVEA-style augmentation (make original and augmented images predict a shared value target). Fresh BC, DrQv2+BC and InfoGating runs serve as anchors. Asterisks indicate adaptations, not original-paper reproductions.','',
      'All runs used the existing 100,000-transition / 200-episode distracted cheetah subset, 4,000 updates, batch 16 and training seeds 0--2. No clean observations were used for training, and no evaluation transitions were added to the dataset.','',
      '## Measured results','',
      'Returns are raw episode rewards: higher is better. Each value is mean +/- sample SD across three independently trained policies, after averaging three episodes per policy in each condition.','',
      '| Method | Familiar return | Held-out background return | Training seconds/run |',
      '|---|---:|---:|---:|']
    for r in summary:
        a,b=r['familiar_background'],r['heldout_background']
        lines.append(f"| {r['label']} | {a['mean']:.2f} +/- {a['sample_sd']:.2f} | {b['mean']:.2f} +/- {b['sample_sd']:.2f} | {r['mean_training_seconds']:.1f} |")
    lines += ['', '### Dependence on the evaluation case', '',
        'Each entry below averages the three trained policies on one held-out evaluation seed. These are descriptive breakdowns, not additional independent training repetitions.', '',
        '| Method | Eval seed 701 | Eval seed 702 | Eval seed 703 |', '|---|---:|---:|---:|']
    for r in summary:
        vals=[statistics.mean(e['return'] for e in rows if e['method']==r['method'] and e['condition']=='heldout_background' and e['eval_seed']==s) for s in protocol['evaluation_seeds']]
        lines.append('| '+r['label']+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['',f"Total measured training-update time: {result['training_seconds_total']/60:.1f} minutes. Loading, initialization, checkpoint writes and simulator evaluation are excluded.",'',
      '![Returns with individual training-seed means](comparison.png)','',
      '## What this does and does not tell us','',
      '- The screen identifies candidates for a longer, predeclared follow-up. A low score here can reflect slow learning, unsuitable default hyperparameters or implementation adaptation; it does not disprove a published method.',
      '- The earlier six-method pilot used equal training time, whereas this screen uses equal update counts. Its old scores must not be mixed into this ranking.',
      '- These are static-within-episode distractions. The reused evaluator leaves dynamic=False; fixed_distraction=False only changes how distractions are sampled, not whether they move.',
      '- The evaluation videos and seeds have been used in earlier experiments. This is a development benchmark; any follow-up claim needs new held-out conditions.',
      '- Only three training seeds and one task were tested. The nine episodes per condition per method are not nine independent training seeds. No significance or convergence claim is made.',
      '- InfoGating retains its original warm-up and has only 1,501 full gate updates here. Its architecture and learning rate differ from the value-based agents. The comparison cannot isolate the causal benefit of gating.',
      '- IQL and AWAC are adapted to the shared visual encoder, scheduled-normal actor and three-step target. SVEA is adapted to DrQv2+BC with RGB gains. None is a faithful published benchmark reproduction.','',
      '## Verification','',
      'All six conditions passed finite-update, bounded-action, changed-encoder and exact next-update checkpoint/resume tests. Runs save every 250 updates. The data check confirms 200 complete episodes and stores the source-manifest checksum. The analysis refuses missing runs, wrong update counts, duplicated evaluation cases or wrong episode lengths. Full logs and checkpoints remain outside Git in the experiment folder.','',
      '## What to do next','',
      'Compare the leading candidate with BC, DrQv2+BC and InfoGating using a substantially longer budget, separate tuning conditions, additional training seeds and new background sources. Check reward-learning diagnostics before adding domain-adversarial alignment. Do not change the submitted paper on the basis of this exploratory screen alone.','',
      '## Papers and implementation notes','',
      '1. I. Kostrikov, A. Nair and S. Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022. https://arxiv.org/abs/2110.06169',
      '2. A. Nair, A. Gupta, M. Dalal and S. Levine, *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*, 2020/2021. https://arxiv.org/abs/2006.09359',
      '3. N. Hansen, H. Su and X. Wang, *Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation*, NeurIPS 2021. https://arxiv.org/abs/2107.00644',
      '4. M. Tomar et al., *Ignorance is Bliss: Robust Control via Information Gating*, NeurIPS 2023. https://arxiv.org/abs/2303.06121','',
      'Full adaptation details and reproduction commands: [screening README](../../scripts/screening/README.md). Existing upstream attributions remain in [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES.md).']
    (O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    sys.path.insert(0,str(ROOT/'report_dependencies'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
    for ax,c,title in zip(axes,protocol['evaluation_conditions'],['Familiar backgrounds','Held-out backgrounds']):
        x=np.arange(len(summary));means=[r[c]['mean'] for r in summary];sd=[r[c]['sample_sd'] for r in summary]
        ax.bar(x,means,color='#427c95',alpha=.6)
        ax.errorbar(x,means,yerr=sd,fmt='none',color='black',capsize=3)
        for j,r in enumerate(summary):ax.scatter(j+np.array([-.12,0,.12]),r[c]['seed_means'],color='#9d392d',s=22,zorder=4)
        ax.set_xticks(x,[r['label'] for r in summary],rotation=35,ha='right');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Raw episode return (higher is better)')
    fig.suptitle('Short screening: 4,000 updates, 100k transitions, 3 training seeds\nBars: mean; whiskers: seed SD; dots: training-seed means. *Adapted methods.',fontsize=11)
    fig.savefig(O/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
