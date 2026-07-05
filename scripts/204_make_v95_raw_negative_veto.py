"""Combine v89 with validated raw lexical-negative and numeric/size vetoes."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path('.');BASE=ROOT/'submissions/final_candidates_v89/FINAL_CANDIDATE_v89_v71_dual_veto_below_m2.csv';SCORE=ROOT/'data/processed/v83_independent_test_scores.parquet';FEAT=ROOT/'data/processed/v21_test_features.parquet';OUT=ROOT/'submissions/final_candidates_v95';REPORT=ROOT/'reports/experiments/v95_raw_negative_veto.json'
def main():
 b=pd.read_csv(BASE);s=pd.read_parquet(SCORE);f=pd.read_parquet(FEAT,columns=['id','query_has_digit','digit_mismatch','query_has_size','size_mismatch','gender_mismatch']);b.id=b.id.astype(str);s.id=s.id.astype(str);f.id=f.id.astype(str)
 if not b.id.equals(s.id) or not b.id.equals(f.id):raise RuntimeError('alignment')
 base=b.prediction.to_numpy(np.int8);constraint=(f.query_has_digit.eq(1)&f.digit_mismatch.eq(1))|(f.query_has_size.eq(1)&f.size_mismatch.eq(1));OUT.mkdir(parents=True,exist_ok=True);rows=[]
 variants={'raw003':s.v82_score.le(.03),'raw004115':s.v82_score.le(.04115),'raw005':s.v82_score.le(.05),'constraint_raw003':constraint|s.v82_score.le(.03),'constraint_raw004115':constraint|s.v82_score.le(.04115),'constraint_raw005':constraint|s.v82_score.le(.05),'constraint_raw004115_gender':constraint|s.v82_score.le(.04115)|f.gender_mismatch.eq(1)}
 for name,veto in variants.items():
  pred=base.copy();pred[veto.to_numpy()&(base==1)]=0;p=OUT/f'FINAL_CANDIDATE_v95_{name}.csv';pd.DataFrame({'id':b.id,'prediction':pred}).to_csv(p,index=False);rows.append({'variant':name,'vetoed':int((pred!=base).sum()),'positives':int(pred.sum()),'ratio':float(pred.mean()),'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 rep={'holdout_raw_negative_precision':{'.03':'not directly gridded; stricter than 97.69%', '.04115':.9768683274,'.05':'between 94.52% and 97.23%'},'rows':rows};REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(rep,indent=2),encoding='utf8');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
