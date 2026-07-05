from pathlib import Path
import json,re,time
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
MODEL='Qwen/Qwen2.5-3B-Instruct';REVIEW=Path('reports/manual_review/review_v61_pairwise_berturk_swaps.csv');OUT=Path('reports/manual_review/v63_qwen25_probe.jsonl')
QUERIES=['ekmek kızartma makinesi','ipad air tablet kılıfı','ford focus koltuk kılıfı','24 jant bisiklet','n & d kedi maması','mini tost makinası','philips airfryer','iphone 13 pro','5w30 motor yağı','banyo dolabı seti','oyuncu mousepad','kron dağ bisikleti','gillette tıraş bıçağı','çift kişilik yatak örtüsü','7. sınıf matematik','ofis kitaplık','mini isıtıcı','büyük şemsiye']
def main():
 d=pd.read_csv(REVIEW);qcol='query_x' if 'query_x' in d else 'query';d=d.drop_duplicates(['id_add','id_drop']);rows=[]
 for q in QUERIES:
  x=d[d[qcol].astype(str).str.casefold().eq(q.casefold())]
  if len(x):rows.append(x.iloc[0])
 print('probe rows',len(rows));tok=AutoTokenizer.from_pretrained(MODEL);model=AutoModelForCausalLM.from_pretrained(MODEL,dtype='auto',device_map='auto',low_cpu_mem_usage=True);model.eval();print('device map', getattr(model, 'hf_device_map', 'single_device')); print('model device', next(model.parameters()).device)
 system='Sen Türkçe e-ticaret arama alaka uzmanısın. Sorgunun tam niyetini, marka, model, sayı, cinsiyet, yaş, ürün türü ve aksesuar/ana ürün ayrımını katı uygula. Sadece tek satır JSON döndür.'
 OUT.parent.mkdir(parents=True,exist_ok=True);f=OUT.open('w',encoding='utf8');t=time.time()
 for i,r in enumerate(rows):
  prompt=f'''Sorgu: {r[qcol]}\nA ürünü: {r.title_add}\nA kategorisi: {r.category_add}\nB ürünü: {r.title_drop}\nB kategorisi: {r.category_drop}\n\nA'nın B yerine pozitif seçilmesi doğru bir düzeltme mi? Eğer ikisi de alakalıysa düzeltme kanıtlanmamıştır. JSON şeması: {{"decision":"ACCEPT|REJECT","a_relevant":true,"b_relevant":false,"confidence":0,"reason":"kısa gerekçe"}}'''
  text=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors='pt',truncation=True,max_length=768).to(model.device)
  with torch.no_grad():out=model.generate(**enc,max_new_tokens=100,do_sample=False,pad_token_id=tok.eos_token_id)
  ans=tok.decode(out[0,enc.input_ids.shape[1]:],skip_special_tokens=True).strip();rec={'query':r[qcol],'title_add':r.title_add,'title_drop':r.title_drop,'answer':ans};f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();print(i+1,r[qcol],ans,flush=True)
 f.close();print('saved',OUT,'elapsed_min',(time.time()-t)/60)
if __name__=='__main__':main()
