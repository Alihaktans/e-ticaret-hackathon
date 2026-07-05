"""Reverse A/B presentation to measure positional bias in the V71 Qwen audit."""
from pathlib import Path
import json, re, time
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path('.'); IN=ROOT/'reports/manual_review/v71_qwen25_consensus_audit.csv'
OUT=ROOT/'reports/manual_review/v71_qwen25_consensus_reverse_audit.jsonl'
SUMMARY=ROOT/'reports/experiments/v71_qwen25_position_bias_summary.json'; MODEL='Qwen/Qwen2.5-3B-Instruct'

def parse(text):
 m=re.search(r'\{.*\}',text,flags=re.S)
 if not m:return {'parse_error':True}
 try:return json.loads(m.group())
 except Exception:return {'parse_error':True}

def main():
 d=pd.read_csv(IN);tok=AutoTokenizer.from_pretrained(MODEL);model=AutoModelForCausalLM.from_pretrained(MODEL,dtype='auto',device_map='auto',low_cpu_mem_usage=True);model.eval()
 system=('Sen katı bir Türkçe e-ticaret arama alaka denetçisisin. A ve B ürününü birbirinden bağımsız değerlendir. '
         'Marka, model, sayı, beden, cinsiyet, yaş ve ürün türüne dikkat et. Yalnız tek satır geçerli JSON döndür.')
 parsed=[];start=time.time()
 with OUT.open('w',encoding='utf8') as f:
  for i,r in d.iterrows():
   # Deliberately reversed: original drop is A, proposed add is B.
   prompt=f'''Sorgu: {r.query}
A başlık: {r.title_drop}
A kategori: {r.category_drop}
A marka: {r.brand_drop}
B başlık: {r.title_add}
B kategori: {r.category_add}
B marka: {r.brand_add}

Her ürünü bağımsız puanla: 0=alakasız, 1=zayıf/kısmi, 2=alakalı, 3=tam eşleşme. Şema: {{"a_score":0,"b_score":0,"preferred":"A|B|TIE","a_hard_conflict":false,"b_hard_conflict":false,"reason":"kısa gerekçe"}}'''
   chat=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True);enc=tok(chat,return_tensors='pt',truncation=True,max_length=768).to(model.device)
   with torch.no_grad():out=model.generate(**enc,max_new_tokens=110,do_sample=False,pad_token_id=tok.eos_token_id)
   answer=tok.decode(out[0,enc.input_ids.shape[1]:],skip_special_tokens=True).strip();obj=parse(answer);parsed.append(obj);f.write(json.dumps({'proposal_rank':int(r.proposal_rank),'band':r.band,'answer':answer,'parsed':obj},ensure_ascii=False)+'\n');f.flush();print(i+1,obj,flush=True)
 rows=[]
 for r,obj in zip(d.itertuples(),parsed):
  rows.append({'band':r.band,'valid':not obj.get('parse_error',False),'reverse_add_score_gt_drop':obj.get('b_score',-1)>obj.get('a_score',-1),'reverse_prefers_add':obj.get('preferred')=='B'})
 x=pd.DataFrame(rows);bands=[]
 for band,g in x.groupby('band',sort=False):
  v=g[g.valid];bands.append({'band':band,'n':len(g),'valid':len(v),'reverse_add_score_gt_drop_rate':float(v.reverse_add_score_gt_drop.mean()),'reverse_prefers_add_rate':float(v.reverse_prefers_add.mean())})
 result={'model':MODEL,'elapsed_minutes':(time.time()-start)/60,'bands':bands,'warning':'Large original/reverse disagreement indicates positional bias.'};SUMMARY.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
