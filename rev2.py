src=open("revision.py").read(); exec(src.split("res={}")[0].replace("B=1000","B=500"))
tr=base[base.cycle.isin(OLD)]; mods=fit_tiers(tr)
out=[]
def oe(m,dd,t="T3"): y=base.y.values[m]; p=mods[t].predict_proba(dd.loc[m,TIERS[t]])[:,1]; return y.mean()/p.mean()
GROUPS={"Electrolytes (Na, K, Cl, HCO3)":["sodium_si","potassium_si","chloride_si","bicarbonate_si"],"Sodium":["sodium_si"],"Potassium":["potassium_si"],
 "Chloride":["chloride_si"],"Bicarbonate":["bicarbonate_si"],"Albumin":["albumin_si"],"Total protein":["total_protein_si"],"Calcium":["calcium_si"],
 "Phosphate":["phosphorus_si"],"Uric acid":["uric_acid_si"],"Glucose":["glucose_si"],"Lipids (cholesterol, triglycerides)":["cholesterol_si","triglycerides_si"],
 "Liver enzymes (ALT, AST, ALP)":["alt_si","ast_si","alp_si"]}
for c in NEW:
    m=(base.cycle==c).values; base_oe=oe(m,base)
    for g,cols in GROUPS.items():
        q=qm(base,base.cycle.isin(OLD),base.cycle==c,cols=cols)
        ccols=[x for x in cols if x in FEAT_ADJ]
        cd=cdc(base,cols=ccols) if ccols else None
        out.append(dict(cycle=c,group=g,OE_unadj=round(base_oe,3),OE_QM_this_group=round(oe(m,q),3),dOE_QM=round(oe(m,q)-base_oe,3),
                        OE_CDC_this_group=round(oe(m,cd),3) if cd is not None else None,dOE_CDC=round(oe(m,cd)-base_oe,3) if cd is not None else None))
pd.DataFrame(out).to_csv("res/rev2_attribution.csv",index=False); print(pd.DataFrame(out).to_string())
