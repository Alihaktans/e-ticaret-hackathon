from pathlib import Path
import os, re, json, time, unicodedata
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path('.')
MODEL = 'Qwen/Qwen2.5-3B-Instruct'
SAMPLE = ROOT/'data/raw/sample_submission.csv'
REVIEW = ROOT/'reports/manual_review/review_v60_public080_bge_swaps.csv'
V60 = ROOT/'submissions/final_candidates_v60/v60_public080_ultra_cap250.csv'
OUT_DIR = ROOT/'reports/manual_review'
SUB_DIR = ROOT/'submissions/final_candidates_v64'
OUT_JSONL = OUT_DIR/'v64_qwen25_v60_judgements.jsonl'
OUT_REVIEW = OUT_DIR/'review_v64_qwen25_v60_swaps.csv'
OUT_SUMMARY = OUT_DIR/'v64_qwen25_candidate_summary.csv'
OUT_SAVED = OUT_DIR/'v64_qwen25_saved_candidates.csv'

TR = str.maketrans({'ı':'i','İ':'i','ş':'s','Ş':'s','ğ':'g','Ğ':'g','ü':'u','Ü':'u','ö':'o','Ö':'o','ç':'c','Ç':'c'})
BRANDS = {
    'huawei':['huawei'], 'ipad':['ipad','apple'], 'apple':['apple','ipad','iphone'],
    'ford':['ford'], 'gillette':['gillette'], 'padisah':['padisah','padişah'],
    'philips':['philips'], 'puma':['puma'], 'guess':['guess'], 'adidas':['adidas'],
    'skechers':['skechers'], 'nike':['nike'], 'columbia':['columbia'], 'tommy':['tommy','hilfiger'],
    'gap':['gap'], 'crocs':['crocs'], 'casper':['casper'], 'kron':['kron'],
    'lcw':['lcw','lc waikiki'], 'kahve dunyasi':['kahve dunyasi','kahve dünyası'],
}
ACCESSORY = {'kilif','kılıf','kordon','kayis','kayış','stand','tutucu','aksesuar','yedek','toz torbasi','toz torbası','bicak','bıçak','bicaklari','bıçakları','susu','süsü','kulpu','ayagi','ayağı','minder'}
MAIN_DEVICE = {'ps5','playstation','tablet','telefon','bisiklet','airfryer','subwoofer','hard disk','harddisk'}

def norm(x):
    if pd.isna(x): return ''
    x = str(x).translate(TR).lower()
    x = unicodedata.normalize('NFKD', x)
    x = re.sub(r'[^a-z0-9]+', ' ', x)
    return re.sub(r'\s+', ' ', x).strip()

def find_col(df, names, contains=None):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low: return low[n.lower()]
    if contains:
        for pat in contains:
            for c in df.columns:
                if pat.lower() in c.lower(): return c
    return None

def val(row, col):
    if col and col in row.index and not pd.isna(row[col]): return str(row[col])
    return ''

def parse_json(txt):
    m = re.search(r'\{.*\}', txt.strip(), flags=re.S)
    if not m:
        return {'winner':'UNCERTAIN','a_relevant':False,'b_relevant':False,'confidence':0.0,'reason':'no_json:'+txt[:160]}
    raw = m.group(0).replace('“','"').replace('”','"').replace("'", '"')
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {'winner':'UNCERTAIN','a_relevant':False,'b_relevant':False,'confidence':0.0,'reason':'json_error:'+str(e)+':'+raw[:160]}
    if 'winner' not in obj and 'decision' in obj:
        a = bool(obj.get('a_relevant', False)); b = bool(obj.get('b_relevant', False))
        if a and not b: obj['winner'] = 'A'
        elif b and not a: obj['winner'] = 'B'
        else: obj['winner'] = 'UNCERTAIN'
    w = str(obj.get('winner','UNCERTAIN')).upper().strip()
    if w not in {'A','B','TIE','UNCERTAIN'}:
        w = 'A' if ('A' in w and 'B' not in w) else ('B' if ('B' in w and 'A' not in w) else 'UNCERTAIN')
    try: conf = float(obj.get('confidence',0.0))
    except Exception: conf = 0.0
    if conf > 1: conf /= 100.0
    conf = max(0.0, min(1.0, conf))
    return {'winner':w, 'a_relevant':bool(obj.get('a_relevant',False)), 'b_relevant':bool(obj.get('b_relevant',False)), 'confidence':conf, 'reason':str(obj.get('reason',''))[:500]}

def hard_guard(query, add_title, add_brand, add_cat):
    q = norm(query); a = norm(' '.join([add_title, add_brand, add_cat])); reasons=[]
    for b, aliases in BRANDS.items():
        if f' {b} ' in f' {q} ' and not any(norm(x) in a for x in aliases):
            reasons.append('brand_missing:'+b)
    if 'ford focus' in q and not ('ford' in a and 'focus' in a): reasons.append('ford_focus_mismatch')
    if 'huawei matepad' in q and not ('huawei' in a and 'matepad' in a): reasons.append('huawei_matepad_mismatch')
    if 'ipad air' in q and not ('ipad' in a and 'air' in a): reasons.append('ipad_air_mismatch')
    elif 'ipad' in q and 'ipad' not in a: reasons.append('ipad_missing')
    for token, ctxs in [('11 5',['matepad']),('115',['matepad']),('24',['jant']),('20',['subwoofer']),('5w 30',['motor yag']),('5w30',['motor yag']),('2 tb',['hard disk']),('2tb',['harddisk']),('224',['dolap kulpu'])]:
        if token in q and not token in a and any(c in q for c in ctxs): reasons.append('important_number_missing:'+token)
    if ('ps5' in q or 'ps 5' in q or 'playstation' in q) and not any(w in q for w in ['aksesuar','stand','tutucu']):
        if any(w in a for w in ['stand','tutucu','duvar stand','aksesuar']): reasons.append('main_ps5_vs_accessory')
    if 'erkek' in q and 'kadin' in a and 'erkek' not in a: reasons.append('male_query_female_add')
    if 'kadin' in q and 'erkek' in a and 'kadin' not in a: reasons.append('female_query_male_add')
    if 'bebek' in q and 'bebek' not in a: reasons.append('baby_missing')
    if 'cocuk' in q and not any(w in a for w in ['cocuk','bebek','gs','junior']): reasons.append('child_missing')
    q_acc = any(norm(w) in q for w in ACCESSORY); a_acc = any(norm(w) in a for w in ACCESSORY)
    if a_acc and not q_acc and any(norm(w) in q for w in MAIN_DEVICE): reasons.append('main_item_vs_accessory')
    return reasons

def prompt(query, at, ab, ac, dt, db, dc):
    return f'''You are a strict Turkish e-commerce search relevance judge.

Search query:
{query}

Product A = proposed ADD candidate:
title: {at}
brand: {ab}
category: {ac}

Product B = current positive item that would be DROPPED:
title: {dt}
brand: {db}
category: {dc}

Choose which product is more relevant to the query.
Rules:
- Brand, model, number, size, gender, age group, vehicle/device model are hard constraints.
- If query asks for a main product, accessory/stand/case/holder/part is not enough.
- If both are relevant, choose A only if A is clearly better.
- If unsure, choose UNCERTAIN.
Return exactly JSON only:
{{"winner":"A|B|TIE|UNCERTAIN","a_relevant":true/false,"b_relevant":true/false,"confidence":0.0-1.0,"reason":"short Turkish reason"}}'''

def gen(tok, model, p):
    msgs=[{'role':'system','content':'You are a precise JSON-only e-commerce relevance judge.'},{'role':'user','content':p}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = 'System: JSON only.\nUser:\n'+p+'\nAssistant:\n'
    device = next(model.parameters()).device
    inputs = tok(text, return_tensors='pt').to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=170, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

def load_done(path):
    d={}
    if not path.exists(): return d
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            obj=json.loads(line); d[str(obj['id_add'])+'||'+str(obj['id_drop'])]=obj
        except Exception: pass
    return d

def main():
    os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1'); os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    OUT_DIR.mkdir(parents=True, exist_ok=True); SUB_DIR.mkdir(parents=True, exist_ok=True)
    sample=pd.read_csv(SAMPLE, usecols=['id']); sample['id']=sample['id'].astype(str)
    v60=pd.read_csv(V60); v60['id']=v60['id'].astype(str)
    if not v60['id'].reset_index(drop=True).equals(sample['id'].reset_index(drop=True)): raise RuntimeError('V60 id order mismatch')
    review=pd.read_csv(REVIEW); review['v60_order']=np.arange(len(review), dtype=np.int32)
    cols={
        'query':find_col(review,['query_x','query','dst_query_real'],['query']),
        'id_add':find_col(review,['id_add','add_id']), 'id_drop':find_col(review,['id_drop','drop_id']),
        'title_add':find_col(review,['title_add','add_title']), 'title_drop':find_col(review,['title_drop','drop_title']),
        'cat_add':find_col(review,['category_add','add_category','cat_add']), 'cat_drop':find_col(review,['category_drop','drop_category','cat_drop']),
        'brand_add':find_col(review,['brand_add','add_brand']), 'brand_drop':find_col(review,['brand_drop','drop_brand']),
    }
    for k in ['query','id_add','id_drop','title_add','title_drop']:
        if cols[k] is None: raise RuntimeError(f'missing {k}; columns={list(review.columns)}')
    done=load_done(OUT_JSONL); print('review rows',len(review),'already judged',len(done))
    tok=AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype='auto', device_map='auto', low_cpu_mem_usage=True)
    model.eval(); print('model device', next(model.parameters()).device)
    start=time.time()
    with OUT_JSONL.open('a', encoding='utf-8') as f:
        for i,row in review.iterrows():
            key=str(row[cols['id_add']])+'||'+str(row[cols['id_drop']])
            if key in done: continue
            q=val(row,cols['query']); at=val(row,cols['title_add']); ab=val(row,cols['brand_add']); ac=val(row,cols['cat_add'])
            dt=val(row,cols['title_drop']); db=val(row,cols['brand_drop']); dc=val(row,cols['cat_drop'])
            raw=gen(tok,model,prompt(q,at,ab,ac,dt,db,dc)); parsed=parse_json(raw); guards=hard_guard(q,at,ab,ac)
            obj={'row_index':int(i),'query':q,'id_add':str(row[cols['id_add']]),'id_drop':str(row[cols['id_drop']]),'title_add':at,'title_drop':dt,'qwen_raw':raw[:1200],**parsed,'hard_guard_reasons':'|'.join(guards),'elapsed_min':(time.time()-start)/60}
            f.write(json.dumps(obj, ensure_ascii=False)+'\n'); f.flush()
            print(i+1, q[:42], 'winner',obj['winner'],'a',obj['a_relevant'],'b',obj['b_relevant'],'conf',round(obj['confidence'],2),'guard',obj['hard_guard_reasons'][:60])
    jud=[]
    for line in OUT_JSONL.read_text(encoding='utf-8').splitlines():
        if line.strip(): jud.append(json.loads(line))
    jud=pd.DataFrame(jud).drop_duplicates(['id_add','id_drop'], keep='last')
    review['id_add_str']=review[cols['id_add']].astype(str); review['id_drop_str']=review[cols['id_drop']].astype(str)
    jud['id_add_str']=jud['id_add'].astype(str); jud['id_drop_str']=jud['id_drop'].astype(str)
    out=review.merge(jud.drop(columns=['id_add','id_drop'], errors='ignore'), on=['id_add_str','id_drop_str'], how='left')
    conf=pd.to_numeric(out['confidence'], errors='coerce').fillna(0)
    out['qwen_accept_ultra']=out['winner'].eq('A') & out['a_relevant'].eq(True) & out['b_relevant'].eq(False) & (conf>=0.65) & out['hard_guard_reasons'].fillna('').eq('')
    out['qwen_accept_safe']=out['winner'].eq('A') & out['a_relevant'].eq(True) & (conf>=0.60) & out['hard_guard_reasons'].fillna('').eq('')
    out['qwen_accept_agree']=out['winner'].eq('A') & out['a_relevant'].eq(True) & (conf>=0.50)
    out['v64_rank_score']=conf + .30*out['qwen_accept_ultra'].astype(float) + .15*out['qwen_accept_safe'].astype(float) - .00001*out['v60_order'].astype(float)
    out=out.sort_values(['v64_rank_score','v60_order'], ascending=[False,True])
    out.to_csv(OUT_REVIEW,index=False)
    id_to_idx=pd.Series(np.arange(len(sample)), index=sample['id']); v60_pred=v60['prediction'].astype(np.int8).to_numpy(); base=v60_pred.copy()
    for _,r in review.iterrows():
        aid=str(r[cols['id_add']]); did=str(r[cols['id_drop']])
        if aid in id_to_idx.index: base[int(id_to_idx.loc[aid])]=0
        if did in id_to_idx.index: base[int(id_to_idx.loc[did])]=1
    rows=[]
    for mode,flag in [('ultra','qwen_accept_ultra'),('safe','qwen_accept_safe'),('agree','qwen_accept_agree')]:
        cand=out[out[flag].eq(True)].sort_values(['v64_rank_score','v60_order'], ascending=[False,True])
        print(mode,'pool',len(cand))
        for cap in [25,50,75,100,150,200,250,300]:
            used=min(cap,len(cand))
            if used<=0: continue
            pred=base.copy(); take=cand.head(used)
            for _,r in take.iterrows():
                pred[int(id_to_idx.loc[str(r['id_add_str'])])]=1; pred[int(id_to_idx.loc[str(r['id_drop_str'])])]=0
            name=f'v64_qwen25_{mode}_cap{cap}'; file=SUB_DIR/f'{name}.csv'
            pd.DataFrame({'id':sample['id'],'prediction':pred.astype(np.int8)}).to_csv(file,index=False)
            rows.append({'variant':name,'file':str(file),'mode':mode,'cap':cap,'used_swaps':int(used),'pool':int(len(cand)),'qwen_conf_mean':float(pd.to_numeric(take['confidence'],errors='coerce').fillna(0).mean()),'b_relevant_rate':float(take['b_relevant'].fillna(False).astype(bool).mean()),'hard_guard_reject_rate':float((take['hard_guard_reasons'].fillna('')!='').mean()),'ones':int(pred.sum()),'pos_ratio':float(pred.mean()),'diff_vs_reconstructed_base':int((pred!=base).sum()),'diff_vs_v60_cap250':int((pred!=v60_pred).sum())})
    summ=pd.DataFrame(rows)
    summ.to_csv(OUT_SUMMARY,index=False); summ.to_csv(OUT_SAVED,index=False)
    print('\nSUMMARY')
    if len(summ): print(summ.to_string(index=False))
    print('\noutputs:'); print(OUT_JSONL); print(OUT_REVIEW); print(OUT_SUMMARY); print(OUT_SAVED); print(SUB_DIR)
if __name__=='__main__': main()
