import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import re

def get_model_color(name):
    colors = {'claude':'#4A90E2','gpt':'#45B764','gemini':'#E6A040','mistral':'#9B6B9E','deepseek':'#E57373'}
    return next((v for k,v in colors.items() if k in name.lower()), '#757575')

def get_score_column(df, model):
    pref = {
        'claude-3-5-sonnet-20241022': 'anthropic/',
        'gemini-1.5-pro-002': 'google/',
        'gpt-4o-2024-08-06': 'openai/'
    }
    if model in pref:
        c = f"{pref[model]}{model}_score"
        if c in df.columns: return c
    b = f"{model}_score"
    if b in df.columns: return b
    for col in df.columns:
        if 'score' in col.lower() and model in col:
            return col
    return None

def parse_mean_se(val):
    """Parse a string like '0.123 (0.111, 0.135)' → (0.123, standard_error). Otherwise return (nan,nan)."""
    if val=="NaN": return np.nan,np.nan
    match = re.match(r'([\-\d\.]+)\s*\(([\-\d\.]+),\s*([\-\d\.]+)\)', str(val))
    if match:
        mv,lv,uv = float(match.group(1)), float(match.group(2)), float(match.group(3))
        return mv, (uv - mv)/1.96
    try:
        # Possibly a single float
        return float(val), np.nan
    except:
        return np.nan, np.nan

def process_files(directory):
    d = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(directory,'results_*.csv')):
        base = os.path.basename(f).replace('results_','').replace('.csv','')
        if '_run' in base:
            *m, r = base.split('_run')
            model = '_run'.join(m)
            d[model][r].append(f)
        else:
            d[base]['1'].append(f)
    d = {m:dict(runs) for m,runs in d.items()}

    judges = ['claude-3-5-sonnet-20241022','gemini-1.5-pro-002','gpt-4o-2024-08-06']
    res, vis = {}, []

    for base_model, runs in d.items():
        raw_scores, raw_avg, adj_avg = [], [], []
        for run, files in runs.items():
            df = pd.read_csv(files[0])
            jcols = [get_score_column(df,j) for j in judges if get_score_column(df,j)]
            rsc = [df[c].mean() for c in jcols]
            raw_scores.append(rsc)
            unadj = np.mean(rsc)
            raw_avg.append(unadj)
            if base_model in judges:
                other = [x for x in judges if x!=base_model]
                rec,giv=[],[]
                for oj in other:
                    c = get_score_column(df,oj)
                    if c: rec.append(df[c].mean())
                for oj in other:
                    if oj in d and run in d[oj]:
                        df2 = pd.read_csv(d[oj][run][0])
                        bc = get_score_column(df2, base_model)
                        if bc: giv.append(df2[bc].mean())
                adj = np.nan
                if rec and giv:
                    h = np.mean(giv)
                    adj = (sum(rec)+h)/(len(other)+1)
                adj_avg.append(adj)
            else:
                adj_avg.append(np.nan)

        K = len(runs)
        unmean = np.mean(raw_scores)
        if K>1:
            se = np.std([np.mean(x) for x in raw_scores])/np.sqrt(K)
        else:
            varz = [df[c].var() for c in jcols] if jcols else [0]
            se = np.sqrt(np.mean(varz)/(len(df)*len(jcols))) if jcols else 0
        ci = 1.96*se

        df0 = pd.read_csv(files[0])
        perjudge = {}
        for j in judges:
            c = get_score_column(df0,j)
            if c:
                if K>1:
                    vals = []
                    for fset in runs.values():
                        dfr = pd.read_csv(fset[0])
                        cc = get_score_column(dfr,j)
                        if cc: vals.append(dfr[cc].mean())
                    if vals:
                        mm = np.mean(vals)
                        st = np.std(vals)/np.sqrt(len(vals))
                    else: mm,st=np.nan,np.nan
                else:
                    mm = df0[c].mean()
                    st = np.sqrt(df0[c].var()/len(df0))
                if not np.isnan(mm):
                    ci_j=1.96*st
                    perjudge[j]=f"{mm:.3f} ({mm - ci_j:.3f}, {mm + ci_j:.3f})"

        res[base_model] = {
            "completions": K,
            "n": K*len(df0),
            **perjudge,
            "Unadjusted Average Score": f"{unmean:.3f} ({unmean - ci:.3f}, {unmean + ci:.3f})"
        }

        if base_model in judges:
            val = [x for x in adj_avg if not np.isnan(x)]
            if val:
                am = np.mean(val)
                se_a = np.std(val)/np.sqrt(K) if K>1 else se
                ci_a=1.96*se_a
                res[base_model]["Average Score"] = f"{am:.3f} ({am - ci_a:.3f}, {am + ci_a:.3f})"
                n_j = len(judges)
                diffs=[]
                for i in range(K):
                    if not np.isnan(adj_avg[i]):
                        diffs.append((raw_avg[i] - adj_avg[i])*n_j)
                if diffs:
                    sp_m=np.mean(diffs)
                    sp_s=np.std(diffs)/np.sqrt(K) if K>1 else se*n_j
                    sp_ci=1.96*sp_s
                    res[base_model]["Self-Preference"] = f"{sp_m:.3f} ({sp_m - sp_ci:.3f}, {sp_m + sp_ci:.3f})"
                else:
                    res[base_model]["Self-Preference"]="NaN"
            else:
                res[base_model]["Average Score"]=res[base_model]["Unadjusted Average Score"]
                res[base_model]["Self-Preference"]="NaN"
        else:
            res[base_model]["Average Score"]=res[base_model]["Unadjusted Average Score"]
            res[base_model]["Self-Preference"]="NaN"

        # For plotting: parse the "Average Score" or fallback
        sc_str = res[base_model]["Average Score"] if base_model in judges else res[base_model]["Unadjusted Average Score"]
        mVal,seVal = parse_mean_se(sc_str)
        if not np.isnan(mVal) and not np.isnan(seVal):
            half = 1.96*seVal
            vis.append({"model":base_model,"score":mVal,"ci_lower":mVal-half,"ci_upper":mVal+half,"color":get_model_color(base_model)})
        else:
            # fallback
            vis.append({"model":base_model,"score":unmean,"ci_lower":unmean-ci,"ci_upper":unmean+ci,"color":get_model_color(base_model)})

    df_res=pd.DataFrame(res).T

    def wavg_ci(col,only_j=False):
        pts=[]
        for i,row in df_res.iterrows():
            if only_j and i not in judges: continue
            mv,se_ = parse_mean_se(row[col]) if col in row else (np.nan,np.nan)
            if not np.isnan(mv):
                pts.append((row["n"],mv,se_))
        if not pts: return "NaN"
        N = sum(x[0] for x in pts)
        wm = sum(x[0]*x[1] for x in pts)/N
        var_ = 0
        for nn,mv,sev in pts:
            if not np.isnan(sev):
                var_+=(nn**2)*(sev**2)
        wvar=var_/(N**2)
        wse=np.sqrt(wvar)
        ci95=1.96*wse
        return f"{wm:.3f} ({wm - ci95:.3f}, {wm + ci95:.3f})"

    # Weighted average row
    w_row={}
    for c in df_res.columns:
        if c=="completions": w_row[c]=""
        elif c=="n": w_row[c]=df_res["n"].sum()
        elif c=="Self-Preference": w_row[c]=wavg_ci(c,True)
        else: w_row[c]=wavg_ci(c,False)
    df_res.loc["Weighted Average"]=w_row

    return df_res, vis

def plot_results(vis_data, save_path):
    plt.figure(figsize=(12,6))
    x=np.arange(len(vis_data))
    sc=[d["score"] for d in vis_data]
    col=[d["color"] for d in vis_data]
    yerr=np.array([[d["score"]-d["ci_lower"] for d in vis_data],
                   [d["ci_upper"]-d["score"] for d in vis_data]])
    bars=plt.bar(x, sc, color=col)
    plt.errorbar(x, sc, yerr=yerr, fmt='none', ecolor='gray', capsize=5)
    for b in bars:
        h=b.get_height()
        plt.text(b.get_x()+b.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom')
    plt.xlabel("Model")
    plt.ylabel("Average Score")
    plt.xticks(x,[d["model"] for d in vis_data],rotation=45,ha='right')
    plt.tight_layout()
    plt.savefig(save_path,bbox_inches='tight',dpi=300)
    print(f"\nFigure saved to: {os.path.abspath(save_path)}")
    plt.close()

def main(dir_,save_path="results_plot.png"):
    pd.set_option("display.max_columns",None)
    pd.set_option("display.width",1000)
    pd.set_option("display.max_colwidth",None)
    pd.set_option("display.max_rows",None)
    df,vis=process_files(dir_)
    print("\nLLM-as-a-Judge scores with pooled runs. 95% c.i. in brackets.\n")
    print(df)
    plot_results(vis,save_path)

if __name__=="__main__":
    main("/content/drive/MyDrive/eval_outputs/")
