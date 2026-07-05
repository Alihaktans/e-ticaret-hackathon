from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

ROOT = Path('.')
SAMPLE = ROOT/'data/raw/sample_submission.csv'
PAIRS = ROOT/'data/raw/submission_pairs.csv'
V33 = ROOT/'data/processed/v33_ft_pair_scores.parquet'
V34 = ROOT/'data/processed/v34/v34_cross_encoder_pair_scores.parquet'
V38 = ROOT/'data/processed/v38_lexical_pair_scores.parquet'
V39 = ROOT/'data/processed/v39_item_history_pair_scores.parquet'

ANCHORS = [
    ROOT/'submissions/FINAL_MAIN_v33_PERFECTED_qprob_top2000.csv',
    ROOT/'submissions/FINAL_CANDIDATE_v33_PERFECTED_qprob_top2000.csv',
]
DIRECT_BASES = {
    'raw_v35_b5000': [
        ROOT/'submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p1_b5000.csv',
        ROOT/'submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p06_b5000.csv',
        ROOT/'submissions/FINAL_CANDIDATE_v35_v33q2000_qswap_v34_v33_balanced_ce_only_strict_g0p03_b5000.csv',
    ],
    'v36p2_balanced': [ROOT/'submissions/FINAL_MAIN_v36p2_balanced.csv', ROOT/'submissions/FINAL_CANDIDATE_v36p2_balanced.csv'],
    'v36p2_strict': [ROOT/'submissions/FINAL_MAIN_v36p2_strict.csv', ROOT/'submissions/FINAL_CANDIDATE_v36p2_strict.csv'],
}
SAVED_CSVS = [ROOT/'reports/manual_review/v38_lexical_saved_candidates.csv', ROOT/'reports/manual_review/v39_item_history_saved_candidates.csv']
SAVED_VARIANTS = {
    'v38_precision_loose_2800': 'v38_raw_v35_b5000_v38_precision_score_loose_cap2800',
    'v38_veto_bad_only': 'v38_raw_v35_b5000_veto_bad_only',
    'v39_raw35_history_lex_veto_bad': 'v39_raw_v35_b5000_v39_history_lex_score_history_veto_contradiction_and_bad',
    'v39_raw35_history_gate_veto': 'v39_raw_v35_b5000_v39_history_gate_score_history_veto_contradiction',
}
LABEL_FILES = {
    'random_clean_v2': ROOT/'reports/manual_review/random_review_blind_v2_assistant_clean_high_only.csv',
    'manual_v1': ROOT/'reports/manual_review/manual_review_set_v1_assistant_labeled.csv',
    'review_v13_vs_v5': ROOT/'reports/manual_review/review_v13_vs_v5_changes_assistant_labeled.csv',
    'review_v15_vs_v13': ROOT/'reports/manual_review/review_v15_vs_v13_changes_assistant_labeled.csv',
    'v20_active': ROOT/'reports/manual_review/review_v20_active_learning_targets_assistant_labeled.csv',
    'v21_active': ROOT/'reports/manual_review/review_v21_active_learning_targets_assistant_labeled.csv',
    'v26_sparse': ROOT/'reports/manual_review/review_v26_sparse_additions_targets_assistant_labeled.csv',
}
REPORT = ROOT/'reports/manual_review'
SUB = ROOT/'submissions'
OUT_SUMMARY = REPORT/'v40_boundary_stability_summary.csv'
OUT_EVAL = REPORT/'v40_boundary_stability_eval.csv'
OUT_SAVED = REPORT/'v40_boundary_stability_saved_candidates.csv'
OUT_REVIEW = REPORT/'review_v40_boundary_stability_sample.csv'


def first_existing(paths):
    for p in paths:
        if p.exists(): return p
    return None

def md5(x): return hashlib.md5(str(x).encode()).hexdigest()[:8]

def load_pred(path, sample):
    d = pd.read_csv(path); d['id'] = d['id'].astype(str)
    if not d['id'].reset_index(drop=True).equals(sample['id'].reset_index(drop=True)):
        raise RuntimeError(f'id order mismatch: {path}')
    return d['prediction'].astype(np.int8).to_numpy()

def find_saved_variant(variant):
    for csv_path in SAVED_CSVS:
        if not csv_path.exists(): continue
        d = pd.read_csv(csv_path)
        if 'variant' not in d.columns or 'file' not in d.columns: continue
        hit = d[d['variant'].astype(str).eq(variant)]
        if len(hit):
            p = Path(str(hit.iloc[0]['file']))
            if p.exists(): return p
    return None

def align_parquet(path, sample, cols):
    d = pd.read_parquet(path, columns=['id'] + cols); d['id'] = d['id'].astype(str)
    if d['id'].reset_index(drop=True).equals(sample['id'].reset_index(drop=True)):
        return d[['id'] + cols].copy()
    return sample[['id']].merge(d[['id'] + cols], on='id', how='left', validate='one_to_one')

def metric_row(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    return dict(macro_f1=f1_score(y,p,average='macro',labels=[0,1],zero_division=0), precision=precision_score(y,p,zero_division=0), recall=recall_score(y,p,zero_division=0), pred_pos_ratio=float(p.mean()), true_pos_ratio=float(y.mean()), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def evaluate(variants, sample):
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample['id'])
    rows=[]
    for label_set, path in LABEL_FILES.items():
        if not path.exists():
            print('missing label:', label_set, path); continue
        lab = pd.read_csv(path)
        if 'id' not in lab.columns or 'assistant_label' not in lab.columns: continue
        lab['id'] = lab['id'].astype(str)
        lab = lab[lab['assistant_label'].isin([0,1,'0','1'])].copy()
        if len(lab) < 30: continue
        lab['assistant_label'] = lab['assistant_label'].astype(int)
        subsets = {'all': lab}
        if 'needs_recheck' in lab.columns:
            nr = pd.to_numeric(lab['needs_recheck'], errors='coerce').fillna(0).astype(int)
            subsets['clean'] = lab[nr == 0].copy()
        if 'assistant_confidence' in lab.columns and 'needs_recheck' in lab.columns:
            conf = lab['assistant_confidence'].astype(str).str.lower()
            nr = pd.to_numeric(lab['needs_recheck'], errors='coerce').fillna(0).astype(int)
            subsets['high_clean'] = lab[(nr==0) & conf.eq('high')].copy()
            subsets['high_medium_clean'] = lab[(nr==0) & conf.isin(['high','medium'])].copy()
        if label_set in ['v20_active','v21_active','v26_sparse'] and 'review_bucket' in lab.columns:
            for b,g in lab.groupby('review_bucket'):
                if len(g)>=20 and g['assistant_label'].nunique()>=2:
                    subsets[f'bucket_{b}'] = g.copy()
        for subset_name, part in subsets.items():
            if len(part)<30 or part['assistant_label'].nunique()<2: continue
            idx = part['id'].map(id_to_idx); ok = idx.notna()
            if ok.sum()<30: continue
            idx = idx[ok].astype(int).to_numpy(); y = part.loc[ok,'assistant_label'].astype(int).to_numpy()
            for name,pred in variants.items():
                r = dict(label_set=label_set, subset=subset_name, eval_key=f'{label_set}_{subset_name}', variant=name, n=int(len(y)))
                r.update(metric_row(y, pred[idx])); rows.append(r)
    return pd.DataFrame(rows)

def weighted(eval_df):
    weights = {
        'random_clean_v2_all': .26, 'random_clean_v2_clean': .26, 'manual_v1_clean': .25,
        'manual_v1_high_clean': .12, 'manual_v1_high_medium_clean': .10,
        'review_v15_vs_v13_clean': .13, 'review_v15_vs_v13_high_medium_clean': .11,
        'review_v13_vs_v5_clean': .05, 'v20_active_clean': .04, 'v21_active_clean': .09,
        'v21_active_high_medium_clean': .07, 'v26_sparse_clean': .07, 'v26_sparse_high_medium_clean': .05,
    }
    main_keys = {'random_clean_v2_all','manual_v1_clean','manual_v1_high_clean','manual_v1_high_medium_clean','review_v15_vs_v13_clean','review_v15_vs_v13_high_medium_clean'}
    rows=[]
    for name,g in eval_df.groupby('variant'):
        val=wsum=0.0; used=[]; main=[]
        for _,r in g.iterrows():
            k=r['eval_key']; w=weights.get(k,0.0)
            if w:
                val += w*float(r['macro_f1']); wsum += w; used.append(float(r['macro_f1']))
            if k in main_keys: main.append(float(r['macro_f1']))
        if wsum:
            rows.append(dict(variant=name, weighted_macro=float(val/wsum), used_min_macro=float(np.min(used)) if used else np.nan, main_min_macro=float(np.min(main)) if main else np.nan, main_mean_macro=float(np.mean(main)) if main else np.nan, eval_count=int(len(g)), mean_precision=float(g['precision'].mean()), mean_recall=float(g['recall'].mean())))
    return pd.DataFrame(rows)

def load_scores(sample, pairs, anchor):
    print('loading V33/V34/V38/V39 score tables...')
    v33 = align_parquet(V33, sample, ['v33_ft_pct_rank'])
    v34 = align_parquet(V34, sample, ['v34_ce_pct_rank'])
    v38 = align_parquet(V38, sample, ['v38_lex_score','v38_lex_pct_rank','v38_exact_phrase_title','v38_word_overlap_title','v38_must_token_coverage_full','v38_query_word_only_category_ratio'])
    if not V39.exists(): raise FileNotFoundError('V39 scores missing. Run scripts\\134_make_v39_item_history_filter.py first.')
    v39 = align_parquet(V39, sample, ['v39_hist_score','v39_hist_has_item','v39_hist_weighted_cov','v39_hist_must_cov','v39_hist_pct_rank'])
    df = pairs.copy()
    for c in ['v33_ft_pct_rank']: df[c] = pd.to_numeric(v33[c], errors='coerce').fillna(0).astype(np.float32)
    for c in ['v34_ce_pct_rank']: df[c] = pd.to_numeric(v34[c], errors='coerce').fillna(0).astype(np.float32)
    for c in v38.columns:
        if c!='id': df[c] = pd.to_numeric(v38[c], errors='coerce').fillna(0).astype(np.float32)
    for c in v39.columns:
        if c!='id': df[c] = pd.to_numeric(v39[c], errors='coerce').fillna(0).astype(np.float32)
    df['term_count'] = df.groupby('term_id')['id'].transform('count').astype(np.int32)
    quota = pd.DataFrame({'term_id':df['term_id'], 'anchor':anchor}).groupby('term_id')['anchor'].sum()
    df['anchor_quota'] = df['term_id'].map(quota).fillna(0).astype(np.int32)
    df['v33_rs'] = (1 - df['v33_ft_pct_rank']).clip(0,1)
    df['v34_rs'] = (1 - df['v34_ce_pct_rank']).clip(0,1)
    df['v38_rs'] = (1 - df['v38_lex_pct_rank']).clip(0,1)
    df['v39_rs'] = (1 - df['v39_hist_pct_rank']).clip(0,1)
    denom = np.maximum(1, df['term_count'].to_numpy(np.float32)-1)
    df['v33_rank'] = 1 + df['v33_ft_pct_rank'].to_numpy(np.float32)*denom
    df['v34_rank'] = 1 + df['v34_ce_pct_rank'].to_numpy(np.float32)*denom
    df['v38_rank'] = 1 + df['v38_lex_pct_rank'].to_numpy(np.float32)*denom
    df['v39_rank'] = 1 + df['v39_hist_pct_rank'].to_numpy(np.float32)*denom
    a = anchor.astype(np.float32)
    df['v40_consensus_score'] = (.31*df['v34_rs'] + .27*df['v33_rs'] + .24*df['v38_rs'] + .12*df['v39_rs'] + .06*a).astype(np.float32)
    df['v40_text_score'] = (.42*df['v38_rs'] + .23*df['v34_rs'] + .18*df['v33_rs'] + .11*df['v39_rs'] + .06*a).astype(np.float32)
    df['v40_history_score'] = (.32*df['v39_rs'] + .26*df['v38_rs'] + .22*df['v34_rs'] + .14*df['v33_rs'] + .06*a).astype(np.float32)
    return df

def pair_swaps(df, anchor, cand, score_col):
    add_mask = (anchor==0) & (cand==1); drop_mask = (anchor==1) & (cand==0)
    cols = ['id','term_id','item_id','term_count','anchor_quota',score_col,'v33_rs','v34_rs','v38_rs','v39_rs','v33_rank','v34_rank','v38_rank','v39_rank','v38_lex_score','v38_word_overlap_title','v38_must_token_coverage_full','v38_query_word_only_category_ratio','v39_hist_score','v39_hist_has_item','v39_hist_weighted_cov','v39_hist_must_cov']
    add = df.loc[add_mask, cols].copy().sort_values(['term_id',score_col], ascending=[True,False])
    drop = df.loc[drop_mask, cols].copy().sort_values(['term_id',score_col], ascending=[True,True])
    add['pair_rank'] = add.groupby('term_id').cumcount(); drop['pair_rank'] = drop.groupby('term_id').cumcount()
    sw = add.merge(drop, on=['term_id','pair_rank'], suffixes=('_add','_drop'), how='inner')
    sw['pair_score_gain'] = sw[f'{score_col}_add'] - sw[f'{score_col}_drop']
    quota = sw['anchor_quota_add'].astype(float); count = np.maximum(1, sw['term_count_add'].astype(float))
    wins=[]; strongs=[]; bounds=[]; contras=[]; gaps=[]; addtops=[]; droptops=[]
    for s in ['v33','v34','v38','v39']:
        rs_add = sw[f'{s}_rs_add'].astype(float); rs_drop = sw[f'{s}_rs_drop'].astype(float)
        rank_add = sw[f'{s}_rank_add'].astype(float); rank_drop = sw[f'{s}_rank_drop'].astype(float)
        sw[f'{s}_win'] = (rs_add > rs_drop).astype(np.int8)
        sw[f'{s}_strong'] = ((rs_add-rs_drop) >= .06).astype(np.int8)
        sw[f'{s}_boundary'] = ((rank_add <= quota) & (rank_drop > quota)).astype(np.int8)
        sw[f'{s}_contra'] = ((rs_add-rs_drop) <= -.08).astype(np.int8)
        sw[f'{s}_gap'] = ((rank_drop-rank_add)/count).astype(np.float32)
        sw[f'{s}_addtop'] = (rs_add >= .80).astype(np.int8)
        sw[f'{s}_droptop'] = (rs_drop >= .80).astype(np.int8)
        wins.append(f'{s}_win'); strongs.append(f'{s}_strong'); bounds.append(f'{s}_boundary'); contras.append(f'{s}_contra'); gaps.append(f'{s}_gap'); addtops.append(f'{s}_addtop'); droptops.append(f'{s}_droptop')
    sw['v40_win_count'] = sw[wins].sum(axis=1).astype(np.int8)
    sw['v40_strong_count'] = sw[strongs].sum(axis=1).astype(np.int8)
    sw['v40_boundary_count'] = sw[bounds].sum(axis=1).astype(np.int8)
    sw['v40_contradiction_count'] = sw[contras].sum(axis=1).astype(np.int8)
    sw['v40_mean_rank_gap'] = sw[gaps].mean(axis=1).astype(np.float32)
    sw['v40_add_top_count'] = sw[addtops].sum(axis=1).astype(np.int8)
    sw['v40_drop_top_count'] = sw[droptops].sum(axis=1).astype(np.int8)
    sw['v40_history_conflict'] = ((sw['v39_hist_has_item_add'].astype(int)==1) & (sw['v39_hist_has_item_drop'].astype(int)==1) & (sw['v39_hist_score_drop'] >= sw['v39_hist_score_add']+.12) & (sw['v39_hist_weighted_cov_drop'] >= sw['v39_hist_weighted_cov_add']+.18)).astype(np.int8)
    sw['v40_lex_conflict'] = ((sw['v38_lex_score_drop'] >= sw['v38_lex_score_add']+.18) & (sw['v38_word_overlap_title_drop'] >= sw['v38_word_overlap_title_add']) & (sw['v38_must_token_coverage_full_drop'] >= sw['v38_must_token_coverage_full_add'])).astype(np.int8)
    sw['v40_category_only_risk'] = ((sw['v38_query_word_only_category_ratio_add'] >= .50) & (sw['v38_word_overlap_title_add'] <= .10) & (sw['v39_hist_score_add'] < .15)).astype(np.int8)
    sw['v40_hard_conflict'] = ((sw['v40_history_conflict']==1) | (sw['v40_lex_conflict']==1) | (sw['v40_category_only_risk']==1)).astype(np.int8)
    sw['v40_stability_score'] = (.15*sw['v40_win_count'] + .12*sw['v40_strong_count'] + .18*sw['v40_boundary_count'] + .65*sw['v40_mean_rank_gap'].clip(-.20,.45) + .08*sw['v40_add_top_count'] - .10*sw['v40_drop_top_count'] - .16*sw['v40_contradiction_count'] - .32*sw['v40_hard_conflict']).astype(np.float32)
    return sw.sort_values(['v40_stability_score','pair_score_gain'], ascending=False).reset_index(drop=True)

def mask_for(sw, mode):
    if mode=='ultra':
        return (sw['v40_stability_score']>=.82) & (sw['v40_win_count']>=4) & (sw['v40_boundary_count']>=2) & (sw['v40_contradiction_count']==0) & (sw['v40_hard_conflict']==0)
    if mode=='strict':
        return (sw['v40_stability_score']>=.58) & (sw['v40_win_count']>=3) & (sw['v40_boundary_count']>=1) & (sw['v40_contradiction_count']<=1) & (sw['v40_hard_conflict']==0)
    if mode=='balanced':
        return (sw['v40_stability_score']>=.42) & (sw['v40_win_count']>=3) & (sw['v40_boundary_count']>=1) & (sw['v40_contradiction_count']<=1) & ~((sw['v40_hard_conflict']==1) & (sw['v40_boundary_count']<3))
    if mode=='impact':
        return (sw['v40_stability_score']>=.30) & (sw['v40_win_count']>=2) & (sw['v40_boundary_count']>=1) & (sw['v40_contradiction_count']<=2) & (sw['v40_category_only_risk']==0)
    raise ValueError(mode)

def apply_swaps(sample, anchor, sw, cap):
    take = sw.sort_values(['v40_stability_score','pair_score_gain'], ascending=False).head(cap).copy()
    id_to_idx = pd.Series(np.arange(len(sample)), index=sample['id'])
    pred = anchor.copy()
    add_idx = take['id_add'].astype(str).map(id_to_idx); drop_idx = take['id_drop'].astype(str).map(id_to_idx)
    if add_idx.isna().any() or drop_idx.isna().any(): raise RuntimeError('id map failed')
    pred[add_idx.astype(int).to_numpy()] = 1; pred[drop_idx.astype(int).to_numpy()] = 0
    return pred, take

def save_review(chosen):
    parts=[]
    for name, sw in chosen.items():
        if len(sw)==0: continue
        x=sw.copy(); x['variant']=name; parts += [x.head(100), x.tail(80)]
    if not parts: return
    r = pd.concat(parts, ignore_index=True).drop_duplicates(['variant','id_add','id_drop'], keep='first')
    keep = ['variant','term_id','pair_rank','id_add','item_id_add','id_drop','item_id_drop','v40_stability_score','pair_score_gain','v40_win_count','v40_strong_count','v40_boundary_count','v40_contradiction_count','v40_mean_rank_gap','v40_history_conflict','v40_lex_conflict','v40_category_only_risk','v40_hard_conflict','v33_rs_add','v33_rs_drop','v34_rs_add','v34_rs_drop','v38_rs_add','v38_rs_drop','v39_rs_add','v39_rs_drop','v38_lex_score_add','v38_lex_score_drop','v39_hist_score_add','v39_hist_score_drop']
    r[[c for c in keep if c in r.columns]].to_csv(OUT_REVIEW,index=False)

def main():
    REPORT.mkdir(parents=True, exist_ok=True); SUB.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE, usecols=['id']); sample['id'] = sample['id'].astype(str)
    pairs = pd.read_csv(PAIRS, usecols=['id','term_id','item_id']); pairs['id']=pairs['id'].astype(str); pairs['term_id']=pairs['term_id'].astype(str); pairs['item_id']=pairs['item_id'].astype(str)
    if not pairs['id'].reset_index(drop=True).equals(sample['id'].reset_index(drop=True)): raise RuntimeError('pairs/sample id mismatch')
    anchor_path = first_existing(ANCHORS)
    if anchor_path is None: raise FileNotFoundError('anchor missing')
    anchor = load_pred(anchor_path, sample)
    print('anchor:', anchor_path, 'ones:', int(anchor.sum()))
    baselines = {'anchor_v33_qprob2000': anchor.copy()}
    for name, paths in DIRECT_BASES.items():
        p = first_existing(paths)
        if p is not None:
            baselines[name] = load_pred(p, sample); print('base:', name, p, 'diff', int((baselines[name]!=anchor).sum()))
    for short, variant in SAVED_VARIANTS.items():
        p = find_saved_variant(variant)
        if p is not None:
            baselines[short] = load_pred(p, sample); print('saved base:', short, p, 'diff', int((baselines[short]!=anchor).sum()))
    df = load_scores(sample, pairs, anchor)
    variants = dict(baselines); meta=[]; chosen={}
    bases = [b for b in ['raw_v35_b5000','v39_raw35_history_lex_veto_bad','v39_raw35_history_gate_veto','v38_precision_loose_2800','v38_veto_bad_only','v36p2_balanced','v36p2_strict'] if b in baselines]
    score_cols = ['v40_consensus_score','v40_text_score','v40_history_score']
    caps = {'ultra':[500,900,1300], 'strict':[900,1500,2200], 'balanced':[1500,2500,3800], 'impact':[2200,3500,5200]}
    for base in bases:
        for sc in score_cols:
            sw = pair_swaps(df, anchor, baselines[base], sc)
            print('pool:', base, sc, len(sw))
            for mode, cap_list in caps.items():
                acc = sw[mask_for(sw, mode)].copy()
                if len(acc)==0: continue
                for cap in cap_list:
                    pred, take = apply_swaps(sample, anchor, acc, min(cap, len(acc)))
                    name = f'v40_{base}_{sc}_{mode}_cap{cap}'
                    variants[name] = pred; chosen[name] = take
                    meta.append(dict(variant=name, source='boundary_stability_layer', base=base, score_col=sc, mode=mode, cap=cap, accepted_pool=int(len(acc)), used_swaps=int(len(take)), stability_mean=float(take['v40_stability_score'].mean()), stability_min=float(take['v40_stability_score'].min()), win_count_mean=float(take['v40_win_count'].mean()), boundary_count_mean=float(take['v40_boundary_count'].mean()), hard_conflict_rate=float(take['v40_hard_conflict'].mean())))
                    print('candidate:', name, 'used', len(take), 'diff', int((pred!=anchor).sum()))
    print('evaluating:', len(variants))
    eval_df = evaluate(variants, sample); eval_df.to_csv(OUT_EVAL, index=False)
    summary = pd.DataFrame(meta)
    for name,pred in baselines.items():
        summary = pd.concat([summary, pd.DataFrame([dict(variant=name, source='baseline', base='', score_col='', mode='', cap='', accepted_pool='', used_swaps='', stability_mean=np.nan, stability_min=np.nan, win_count_mean=np.nan, boundary_count_mean=np.nan, hard_conflict_rate=np.nan)])], ignore_index=True)
    aux=[]
    for name,pred in variants.items():
        aux.append(dict(variant=name, ones=int(pred.sum()), pos_ratio=float(pred.mean()), diff_vs_anchor=int((pred!=anchor).sum()), diff_vs_raw_v35=int((pred!=baselines['raw_v35_b5000']).sum()) if 'raw_v35_b5000' in baselines else -1, diff_vs_v39_lex=int((pred!=baselines['v39_raw35_history_lex_veto_bad']).sum()) if 'v39_raw35_history_lex_veto_bad' in baselines else -1, diff_vs_v38_precision=int((pred!=baselines['v38_precision_loose_2800']).sum()) if 'v38_precision_loose_2800' in baselines else -1, diff_vs_v36p2_balanced=int((pred!=baselines['v36p2_balanced']).sum()) if 'v36p2_balanced' in baselines else -1))
    summary = summary.merge(pd.DataFrame(aux), on='variant', how='right')
    w = weighted(eval_df)
    if len(w): summary = summary.merge(w, on='variant', how='left')
    diff = summary['diff_vs_anchor'].fillna(0).astype(float)
    move = np.minimum(.020, np.log1p(diff)/np.log1p(25000)*.020)
    stab = np.minimum(.004, summary['stability_mean'].fillna(0).astype(float)*.004)
    penalty = np.maximum(0, diff-55000)/900000.0
    summary['v40_decision_score'] = summary['weighted_macro'].fillna(0) + move + stab - penalty
    summary = summary.sort_values(['v40_decision_score','weighted_macro'], ascending=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    save_review(chosen)
    save=[]
    top = summary[(summary['source']=='boundary_stability_layer') & (summary['diff_vs_anchor']>=1500) & (summary['diff_vs_anchor']<=60000)].head(35)
    for nm in top['variant'].tolist():
        if nm not in save: save.append(nm)
    for nm in ['raw_v35_b5000','v39_raw35_history_lex_veto_bad','v39_raw35_history_gate_veto','v38_precision_loose_2800','v38_veto_bad_only','v36p2_balanced','v36p2_strict','anchor_v33_qprob2000']:
        if nm in variants and nm not in save: save.append(nm)
    saved=[]
    for i,nm in enumerate(save[:45],1):
        out = SUB/f'FINAL_CANDIDATE_v40_rankstable_{i:03d}_{md5(nm)}.csv'
        pd.DataFrame({'id':sample['id'], 'prediction':variants[nm].astype(np.int8)}).to_csv(out,index=False)
        row = summary[summary['variant'].eq(nm)].iloc[0].to_dict(); row['file']=str(out); saved.append(row)
        print('saved:', out, '<-', nm)
    pd.DataFrame(saved).to_csv(OUT_SAVED,index=False)
    cols = ['variant','v40_decision_score','weighted_macro','source','base','score_col','mode','cap','used_swaps','stability_mean','stability_min','win_count_mean','boundary_count_mean','hard_conflict_rate','diff_vs_anchor','diff_vs_raw_v35','diff_vs_v39_lex','main_min_macro','main_mean_macro','mean_precision','mean_recall']
    print('\nTOP V40')
    print(summary[[c for c in cols if c in summary.columns]].head(80).to_string(index=False))
    print('\nSAVED')
    sd = pd.DataFrame(saved)
    if len(sd): print(sd[['variant','file','v40_decision_score','weighted_macro','diff_vs_anchor']].to_string(index=False))
    print('\noutputs:')
    print(OUT_SUMMARY); print(OUT_EVAL); print(OUT_SAVED); print(OUT_REVIEW)

if __name__ == '__main__':
    main()
