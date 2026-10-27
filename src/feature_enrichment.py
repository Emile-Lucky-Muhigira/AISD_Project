"""Optional baseline-wave education features; no outcome-based construction.

Original modelling tables are never overwritten. Keys must identify one child.
Questionnaire sources: Wave 1 Part A Q22, Q32--38, Q55/56/58;
Wave 2 Part A corresponding questions (No=5 rather than No=2).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

LITERACY = ['reads_english', 'writes_english', 'written_calculations']
NEW_COLUMNS = LITERACY + ['school_expense_payer', 'teaching_arrangement',
                          'teacher_absent_days_routed']
PAYERS = {1:'father', 2:'mother', 3:'both_parents', 4:'other_household_member',
          5:'other_relative', 6:'non_relative', 7:'self', 8:'other', 9:'ngo', 95:'other'}


def questionnaire_features(education, literacy, wave):
    """Inputs aligned to the baseline children, with identical RangeIndexes.

    Routed absence is main-teacher days for one teacher, otherwise the maximum
    observed English/Math teacher days. It is NOT total days without instruction;
    one available subject is sufficient. Unknown arrangement stays unknown.
    Retain the established 0--22 school-day validity rule, applied per answer.
    """
    if wave not in (1, 2):
        raise ValueError('Baseline wave must be 1 or 2.')
    if not education.index.equals(literacy.index):
        raise ValueError('Education and literacy rows must be aligned.')
    names = (['s1fiii_55','s1fiii_56','s1fiii_58'] if wave == 1 else
             ['readenglish','writeenglish','writtencalculations'])
    out = pd.DataFrame(index=education.index)
    for name, source in zip(LITERACY, names):
        out[name] = pd.to_numeric(literacy[source], errors='coerce').map({1:1., 2 if wave == 1 else 5:0.})
    payer = 's1fi_22' if wave == 1 else 'whopaid'
    allowed = range(1,10) if wave == 1 else [1,2,3,4,5,6,7,95]
    p = pd.to_numeric(education[payer], errors='coerce')
    out['school_expense_payer'] = p.where(p.isin(allowed)).map(PAYERS).fillna('unknown')
    cols = (['s1fi_32','s1fi_34','s1fi_36','s1fi_38'] if wave == 1 else
            ['oneteacher','teacherabsent','englangteacherabsent','mathteacherabsent'])
    arrangement = pd.to_numeric(education[cols[0]], errors='coerce')
    one, multiple = arrangement.eq(1), arrangement.eq(2 if wave == 1 else 5)
    out['teaching_arrangement'] = np.select([one,multiple], ['one_teacher','subject_teachers'], default='unknown')
    days = education[cols[1:]].apply(pd.to_numeric, errors='coerce')
    days = days.where(days.ge(0) & days.le(22))
    out['teacher_absent_days_routed'] = np.nan
    out.loc[one,'teacher_absent_days_routed'] = days.loc[one,cols[1]]
    out.loc[multiple,'teacher_absent_days_routed'] = days.loc[multiple,cols[2:]].max(axis=1)
    return out


def _keys(frame):
    keys = frame[['FPrimary','hhmid']].apply(pd.to_numeric, errors='raise')
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError('Known unique child keys required.')
    return keys.reset_index(drop=True)


def enrich_frame(base, raw_dir, wave):
    """Read only education/literacy from the child's baseline wave."""
    keys = _keys(base)
    if set(NEW_COLUMNS) & set(base):
        raise ValueError('Input is already enriched.')
    if wave not in (1,2):
        raise ValueError('Baseline wave must be 1 or 2.')
    files = (['s1fi.dta','s1fiii.dta'] if wave == 1 else
             ['01fi_generaleducation.dta','01fiii_literacy.dta'])
    joined=[]
    for filename in files:
        path=Path(raw_dir)/f'Wave {wave}'/filename
        try:
            data,_=pyreadstat.read_dta(path)
        except UnicodeDecodeError:
            data,_=pyreadstat.read_dta(path,encoding='latin1')
        data=data.rename(columns={'s1fi_hhmid':'hhmid'})
        for c in keys:
            data[c]=pd.to_numeric(data[c],errors='raise')
        # Unidentified raw records cannot be linked; never choose among duplicates.
        data=data.dropna(subset=list(keys))
        joined.append(keys.merge(data,on=list(keys),how='left',validate='one_to_one',sort=False))
    features=questionnaire_features(*joined,wave)
    result=pd.concat([base.reset_index(drop=True).drop(columns='dropout'),features,
                      base['dropout'].reset_index(drop=True)],axis=1)
    pd.testing.assert_frame_equal(result[list(base.columns)],base.reset_index(drop=True))
    return result


def export_enriched(root):
    root=Path(root); proc=root/'data/processed'
    raw=root/'data/raw/Ghana_Panel_Survey/Data'
    outputs={}
    for prefix,wave in [('train_w1_w2',1),('test_w2_w3',2)]:
        base=pd.read_csv(proc/f'{prefix}_raw.csv')
        enriched=enrich_frame(base,raw,wave)
        enriched.to_csv(proc/f'{prefix}_enriched.csv',index=False)
        outputs[prefix]=enriched
    return outputs
