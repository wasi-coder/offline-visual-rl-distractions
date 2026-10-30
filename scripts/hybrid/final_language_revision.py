from pathlib import Path
R=Path(__file__).resolve().parents[2];p=R/'experiments/hybrid_20261030/ieee_template/content.md';s=p.read_text(encoding='utf-8')
changes={
'encoders trained on distracted recordings can bind the policy to background appearance that changes at evaluation':'encoders trained on distracted recordings can learn background details that change during evaluation',
'Domain-adversarial alignment aims to suppress domain-identifiable features':'Domain-adversarial alignment aims to reduce features that reveal the visual domain',
'the two mechanisms appear complementary':'the two methods appear to support each other',
'The two mechanisms appear complementary':'The two methods appear to support each other',
'a controlled empirical finding and a reproducible experimental procedure':'a controlled experimental finding and a reproducible procedure',
'an unfavorable interaction contrast':'a negative interaction between gating and alignment',
'This finding narrows the hypothesis to what was tested and motivates more targeted investigations of the alignment trade-off.':'This finding applies to the tested setup and points to further checks of how alignment and gating interact.',
'Its observations combine useful agent information with appearance variation.':'The images contain both useful information about the agent and changing visual details.',
'discriminative':'useful for prediction',
'The statistical unit for reported variability is an independently trained agent.':'We measure variation across independently trained agents.',
'After a user-imposed deadline amendment before training began, target 40,000 updates for each of 3 seeds across 4 conditions.':'Before training began, the time budget was revised and the target was set to 40,000 updates for each of three seeds across four conditions.',
'Average test episodes within training seed before calculating across-seed means/SD.':'We average test episodes within each training seed before calculating the mean and standard deviation across seeds.',
'First train it by cross-entropy on detached encoder features. Freeze its parameters during the representation update, and reverse the feature gradient.':'We first train it using cross-entropy on detached encoder features. We then hold its parameters fixed during the representation update and reverse the feature gradient.',
}
for a,b in changes.items():s=s.replace(a,b)
p.write_text(s,encoding='utf-8')
out=R/'docs/manuscript';out.mkdir(exist_ok=True)
(out/'manuscript.md').write_text(s,encoding='utf-8')
print('Revised prose and saved current manuscript source for the repository.')
