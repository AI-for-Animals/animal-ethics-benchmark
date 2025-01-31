import os, sys, re, csv, shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from inspect_ai.log import list_eval_logs, read_eval_log

DFLT_SCORE=0

def mkd(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def last_log(ld="./logs"):
    logs=list_eval_logs(ld,recursive=False)
    if not logs:return None
    return sorted(logs,key=lambda x:os.path.basename(x.name))[-1].name

def cat_letter(txt):
    if not txt.strip():return None
    lines=txt.split('\n')
    if not lines:return None
    m=re.match(r'^\[([A-E])\]',lines[0].strip())
    return m.group(1)if m else None

def parse_eval(logf=None,logd="./logs"):
    if not logf:
        logf=last_log(logd)
        if not logf:
            print(f"No .eval logs in {logd}")
            return{}
    logf=unquote(logf)
    if logf.startswith("file://"):logf=logf[7:]
    if not os.path.exists(logf):
        alt=os.path.join(logd,os.path.basename(logf))
        if os.path.exists(alt):logf=alt
        else:
            print(f"Log file not found:{logf}")
            return{}
    log=read_eval_log(logf,resolve_attachments=True)
    if not log or not log.samples:
        print(f"Empty or invalid log:{logf}")
        return{}
    sc=[s.name for s in(log.results.scores or[])if s.name not in("avg","error_count")]
    catmap={"A":1,"B":2,"C":3,"D":4,"E":5}
    sc_scores={m:[] for m in sc}
    sc_cats={m:[] for m in sc}
    rows=[]
    n=len(log.samples)

    # Instead of auto sid=1..n, we fetch sample.metadata["sample_id"]
    # If missing, fallback to enumerating
    for sample in log.samples:
        sid=sample.metadata.get("sample_id", None)
        if sid is None:
            # fallback if there's no sample_id
            sid=999999  # or some placeholder
        ans=""
        for msg in(sample.messages or[]):
            if getattr(msg,"source",None)=="generate"and msg.role=="assistant":
                ans=msg.content
                break
        # collect model texts
        pm_txt={m:""for m in sc}
        for ev in sample.events or[]:
            if ev.event=="model"and ev.model in sc:
                try:
                    out=ev.output.choices[0].message.content
                    txt=out[0].text if isinstance(out,list)else out
                    pm_txt[ev.model]=txt.strip()
                except: pass

        final_scores=sample.scores.get("final_digit_model_graded_qa",{})
        if hasattr(final_scores,"value"):
            newvals={}
            for k,v in final_scores.value.items():
                vv=str(v).strip("[]")
                newvals[k]=int(vv)if vv in{"-1","0","1"}else DFLT_SCORE
            final_scores=newvals

        # store
        for m in sc:
            scval=final_scores.get(m,DFLT_SCORE)
            sc_scores[m].append(scval)
            c=cat_letter(pm_txt[m])
            sc_cats[m].append(catmap.get(c,np.nan))

        # Build row -> sample_id, input, final_answer, model1_assessment,... model1_category,... model1_score,...
        row=[sid,
             (sample.input or"").replace("\n"," "),
             ans.replace("\n"," ")]
        for m in sc:
            row.append(pm_txt[m])  # model's raw text
        for m in sc:
            last_c=sc_cats[m][-1]
            if not np.isnan(last_c):
                # invert catmap
                found_letter=""
                for letter,val in catmap.items():
                    if val==int(last_c):
                        found_letter=letter
                        break
                row.append(found_letter)
            else:
                row.append("")
        for m in sc:
            row.append(sc_scores[m][-1])
        rows.append(row)

    # Convert dict->list-of-lists
    alpha_sc=[];alpha_cat=[]
    for m in sc:
        alpha_sc.append(sc_scores[m])
        alpha_cat.append(sc_cats[m])
    return {"models":sc,"scores":alpha_sc,"cats":alpha_cat,"n":n,"rows":rows}

def parse_csv(cf):
    if not os.path.exists(cf):
        print(f"CSV file not found: {cf}")
        return{}
    df=pd.read_csv(cf)
    if df.empty:
        print(f"No data in CSV:{cf}")
        return{}
    sc=[c[:-6]for c in df.columns if c.endswith("_score")]
    if not sc:
        print(f"No *_score columns found in {cf}")
        return{}
    catmap={"A":1,"B":2,"C":3,"D":4,"E":5}
    n=len(df)
    alpha_sc=[];alpha_cat=[]
    for m in sc:
        s=df[m+"_score"].fillna(DFLT_SCORE).astype(int).values
        catcol=m+"_category"
        if catcol in df.columns:
            catvals=df[catcol].fillna("").values
            c_list=[]
            for x in catvals:
                c_list.append(catmap[x] if x in catmap else np.nan)
        else:
            c_list=[np.nan]*n
        alpha_sc.append(s)
        alpha_cat.append(c_list)
    return {"models":sc,"scores":alpha_sc,"cats":alpha_cat,"n":n,"rows":None}

def write_csv(rows,models,outd,logf=""):
    mkd(outd)
    m=re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\+\d{2}-\d{2})",logf)
    ts=m.group(1) if m else datetime.now().strftime("%Y%m%d_%H%M%S")
    cols=["sample_id","input","final_answer"]
    for mm in models:cols.append(mm+"_assessment")
    for mm in models:cols.append(mm+"_category")
    for mm in models:cols.append(mm+"_score")
    csvpath=os.path.join(outd,f"results_{ts}.csv")
    with open(csvpath,"w",newline="",encoding="utf-8")as f:
        w=csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"Results CSV saved to:{csvpath}")

def analyze(ms,scs,cts,outd):
    mkd(outd)
    if not ms:
        print("No models found.")
        return
    arr_sc=np.array(scs,dtype=float)
    arr_ct=np.array(cts,dtype=float)
    n=arr_sc.shape[1] if arr_sc.ndim==2 else 0
    print(f"\nNumber of samples:{n}")

    # Flatten for aggregate
    fsc=arr_sc.flatten()
    fsc=fsc[~np.isnan(fsc)]
    meanv=np.mean(fsc) if len(fsc) else float('nan')

    # Flatten categories
    fct=arr_ct.flatten()
    fct=fct[~np.isnan(fct)]
    inv={1:'A',2:'B',3:'C',4:'D',5:'E'}
    let=[inv[int(x)] for x in fct]
    # Tally them, then sort them by alphabetical letter
    tally={}
    for c in let:
        tally[c]=tally.get(c,0)+1
    # sort
    sorted_tally=dict(sorted(tally.items(),key=lambda kv:kv[0]))

    print(f"All {len(ms)} scorers:\n  Average score:{meanv:.3f}\n  Categories:{sorted_tally}")

    for i,m in enumerate(ms):
        scv=arr_sc[i]
        scv=scv[~np.isnan(scv)]
        mm=np.mean(scv) if len(scv) else float('nan')

        catv=arr_ct[i]
        catv=catv[~np.isnan(catv)]
        sub={}
        for val in catv:
            l=inv[int(val)]
            sub[l]=sub.get(l,0)+1
        sub=dict(sorted(sub.items(),key=lambda kv:kv[0]))

        print(f"\n{m}:\n  Average score:{mm:.3f}\n  Categories:{sub}")

    # Also write to analysis_YYYYMMDD_hhmmss.txt
    aname="analysis_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".txt"
    outp=os.path.join(outd,aname)
    with open(outp,"w",encoding="utf-8")as f:
        f.write(f"Number of samples:{n}\n")
        f.write(f"All {len(ms)} scorers:\n  Average score:{meanv:.3f}\n  Categories:{sorted_tally}\n\n")
        for i,m in enumerate(ms):
            scv=arr_sc[i]
            scv=scv[~np.isnan(scv)]
            mm=np.mean(scv) if len(scv) else float('nan')
            catv=arr_ct[i]
            catv=catv[~np.isnan(catv)]
            sub={}
            for val in catv:
                l=inv[int(val)]
                sub[l]=sub.get(l,0)+1
            sub=dict(sorted(sub.items(),key=lambda kv:kv[0]))
            f.write(f"{m}:\n  Average score:{mm:.3f}\n  Categories:{sub}\n\n")
    print(f"\nAnalysis summary saved to:{outp}\n")

def main():
    args=sys.argv[1:]
    lf=None;cf=None;ld="./logs";od="./outputs"
    i=0
    while i<len(args):
        if args[i]=="--log-file"and i+1<len(args):lf=args[i+1];i+=2
        elif args[i]=="--log-dir"and i+1<len(args):ld=args[i+1];i+=2
        elif args[i]=="--csv-file"and i+1<len(args):cf=args[i+1];i+=2
        elif args[i]=="--output-dir"and i+1<len(args):od=args[i+1];i+=2
        else:
            print(f"Error with arg:{args[i]}")
            sys.exit(1)
    if cf:
        d=parse_csv(cf)
        if d.get("n",0)>0:
            analyze(d["models"],d["scores"],d["cats"],od)
    else:
        d=parse_eval(logf=lf,logd=ld)
        if d.get("n",0)>0:
            write_csv(d["rows"],d["models"],od,os.path.basename(lf) if lf else"")
            analyze(d["models"],d["scores"],d["cats"],od)

if __name__=="__main__":
    main()
