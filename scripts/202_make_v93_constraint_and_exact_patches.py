"""Break v71 quotas only for validated hard conflicts and exact title+category evidence."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT=Path('.')
FEATURES=ROOT/'data/processed/v21_test_features.parquet'
BASE=ROOT/'submissions/final_candidates_v89/FINAL_CANDIDATE_v89_v71_dual_veto_below_m2.csv'
OUT=ROOT/'submissions/final_candidates_v93'
REPORT=ROOT/'reports/experiments/v93_constraint_exact_report.json'


def save(ids,pred,name,base,rows):
    path=OUT/f'FINAL_CANDIDATE_v93_{name}.csv'
    pd.DataFrame({'id':ids,'prediction':pred.astype(np.int8)}).to_csv(path,index=False)
    rows.append({'variant':name,'file':str(path),'positives':int(pred.sum()),'ratio':float(pred.mean()),
                 'changed_vs_v89':int((pred!=base).sum()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


def main():
    cols=['id','term_id','query_has_digit','digit_mismatch','gender_mismatch','query_brand_mismatch',
          'exact_query_in_title','category_cov','title_cov','title_jacc','q_token_count']
    f=pd.read_parquet(FEATURES,columns=cols);b=pd.read_csv(BASE)
    f.id=f.id.astype(str);b.id=b.id.astype(str)
    if not f.id.equals(b.id):raise RuntimeError('alignment')
    base=b.prediction.to_numpy(np.int8);OUT.mkdir(parents=True,exist_ok=True);rows=[]
    digit=f.query_has_digit.eq(1)&f.digit_mismatch.eq(1)
    gender=f.gender_mismatch.eq(1)
    exact=(f.exact_query_in_title.eq(1)&f.category_cov.ge(.999)&f.digit_mismatch.eq(0)&
           f.gender_mismatch.eq(0)&f.query_brand_mismatch.eq(0))
    add_pool=np.flatnonzero(exact.to_numpy()&(base==0))
    # Most literal first; category coverage is already exact.
    order=np.lexsort((-f.title_cov.to_numpy()[add_pool],-f.title_jacc.to_numpy()[add_pool]))
    add_pool=add_pool[order]
    remove_masks={'digit':digit,'digit_gender':digit|gender}
    for rem_name,rem in remove_masks.items():
        pred=base.copy();pred[rem.to_numpy()&(base==1)]=0
        save(f.id,pred,f'remove_{rem_name}',base,rows)
        for cap in [1000,2500,5000,10000,20000]:
            p=pred.copy();take=add_pool[:cap];p[take]=1
            save(f.id,p,f'remove_{rem_name}_exactadd{cap}',base,rows)
    report={'validation':{'digit_negative_precision':.9811560065,'digit_gender_union_negative_precision':.9714890844,
                          'exact_category_known_positive_rate':.755956},
            'test_pools':{'digit_remove':int((digit.to_numpy()&(base==1)).sum()),
                          'gender_remove':int((gender.to_numpy()&(base==1)).sum()),
                          'exact_add':int(len(add_pool))},'candidates':rows}
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
