#!/usr/bin/env python3
"""Publication figures generated only from recorded experimental outputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scientific-contracts-matplotlib"))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def group(row):
    variant = row["variant"].lower()
    if row["expected_state"] == "fault":
        return "faulty_variant"
    if row["expected_state"] == "not_evaluated":
        return "ineligible" if "eligib" in row["id"].lower() + variant else "missing_evidence"
    if "repair" in variant:
        return "repair"
    if "benign" in variant:
        return "benign_rewrite"
    return "conformant_original"

METHODS = ("schema", "source_pattern", "behavioral_contract")

ROOT = Path(__file__).resolve().parents[1] / "study/camera_ready"
BLUE="#0072B2"; ORANGE="#C45A12"; INK="#17212B"; GRAY="#707982"
PALE_BLUE="#E5F1F8"; PALE_ORANGE="#FBECDD"; PALE_GRAY="#F0F2F4"
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":10,"axes.labelsize":9,
 "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5,"pdf.fonttype":42,"ps.fonttype":42,
 "text.color":INK,"axes.labelcolor":INK,"axes.edgecolor":INK,"figure.facecolor":"white","savefig.facecolor":"white"})
METADATA={}

def read_json(relative):return json.loads((ROOT/relative).read_text())
def read_csv(relative):
    with (ROOT/relative).open() as f:return list(csv.DictReader(f))
def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#DCE0E3",linewidth=.5);ax.set_axisbelow(True)
def save(fig,out,name,sources,question):
    fig.canvas.draw()
    canvas=fig.bbox
    outside=[]
    for obj in fig.findobj(mpl.text.Text):
        if obj.get_visible() and obj.get_text():
            box=obj.get_window_extent(fig.canvas.get_renderer())
            if box.x0 < -1 or box.x1 > canvas.x1+1 or box.y0 < -1 or box.y1 > canvas.y1+1:
                outside.append(obj.get_text())
    if outside:raise ValueError(f"Text outside fixed-size canvas in {name}: {outside}")
    fig.savefig(out/f"{name}.pdf",metadata={"Title":question})
    fig.savefig(out/f"{name}.svg")
    fig.savefig(out/f"{name}.png",dpi=180)
    small=[t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_text() and t.get_visible() and t.get_fontsize()<8]
    if small:raise ValueError(f"undersized figure text in {name}: {small}")
    METADATA[name]={"question":question,"native_width_inches":5.5,"minimum_authored_font_pt":8,
      "sources":{str(Path("study/camera_ready") / p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    plt.close(fig)

def matrix(out):
    data=read_json("results/behavioral_summary_v1.json");rows=[]
    labels={"agentprm":"AgentPRM","toolace":"ToolACE","agentabstain":"AgentAbstain"}
    for target in labels:
        rr=[r for r in data["rows"] if r["target"]==target]
        for label,cat in [("Faulty variants detected","faulty_variant"),("Genuine repairs accepted","repair"),("Harmless rewrites flagged","benign_rewrite"),("Missing / ineligible:\nnot evaluated","absent")]:
            selected=[r for r in rr if (group(r) in {"missing_evidence","ineligible"} if cat=="absent" else group(r)==cat)]
            rows.append((target,label,cat,selected))
    fig,ax=plt.subplots(figsize=(5.5,5.35));fig.subplots_adjust(left=.49,right=.99,top=.83,bottom=.07)
    for i,(target,label,cat,rr) in enumerate(rows):
        y=i+(i//4)*.8
        for j,m in enumerate(METHODS):
            ds=[r["methods"][m]["decision"] for r in rr]
            available=sum(x in {"pass","fail"} for x in ds)
            unavailable=(m=="native_tests" or (m=="source_pattern" and target=="agentabstain"))
            if not rr:text="n/a";fill="white"
            elif unavailable:text="Unavailable";fill=PALE_GRAY
            else:
                desired="fail" if cat in {"faulty_variant","benign_rewrite"} else "not_evaluated" if cat=="absent" else "pass"
                n=ds.count(desired);text=f"{n}/{len(rr)}"
                good=(n==0 if cat=="benign_rewrite" else n==len(rr))
                fill=PALE_BLUE if good else PALE_ORANGE
                if "error" in ds:text+=f"; {ds.count('error')} err"
            ax.add_patch(Rectangle((j-.5,y-.45),1,.9,facecolor=fill,edgecolor="white",linewidth=1))
            ax.text(j,y,text,ha="center",va="center",fontsize=8 if unavailable else 9)
        ax.text(-.61,y,label,ha="right",va="center",fontsize=8.5,transform=ax.transData)
        if i%4==0:ax.text(-.61,y-.54,labels[target],ha="right",va="bottom",fontsize=9.5,weight="bold")
    ax.set_xlim(-.5,2.5);ax.set_ylim(13.3,-1);ax.set_xticks(range(3),["Field / type\nvalidation","Original\nadapters","Behavioral\ncontracts"]);ax.xaxis.tick_top();ax.tick_params(length=0,pad=12);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    fig.text(.01,.98,"Faults, repairs, and harmless changes",fontsize=10,weight="bold",va="top")
    fig.text(.01,.94,"Each fraction counts evaluated cases.",fontsize=8.5)
    fig.text(.01,.015,"Blue: expected decision. Orange: miss or false alarm. Gray: coverage unavailable.",fontsize=8)
    save(fig,out,"behavior_matrix",["results/behavioral_summary_v1.json"],"Do checks follow implementation behavior?")

def agentprm(out):
    states=read_csv("results/agentprm/state_targets_v1.csv")
    rows=[r for r in states if int(r['length'])==8 and int(r['terminal_score'])==1 and float(r['gamma'])==.95]
    sensitivity=read_json("results/agentprm/case_analysis_v1.json")["primary_target_sensitivity"]
    cells=[r for r in sensitivity if r["terminal_score"]==1]
    fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.75),gridspec_kw={"width_ratios":[1.03,1]});fig.subplots_adjust(left=.11,right=.98,bottom=.24,top=.8,wspace=.48)
    x=[int(r['position']) for r in rows];original=[float(r['original_signed_return']) for r in rows];repaired=[float(r['repaired_signed_return']) for r in rows]
    a.plot(x,original,color=ORANGE,marker="o",markersize=3,label="Original")
    a.plot(x,repaired,color=BLUE,linestyle="--",marker="s",markersize=3,label="Repaired")
    a.set(xlabel="Trajectory position",ylabel="Signed target q",xticks=[0,2,4,7],ylim=(.65,1.025));style(a)
    a.text(4.2,.96,"Repaired",color=BLUE,fontsize=8.5)
    a.text(4.2,.75,"Original",color=ORANGE,fontsize=8.5)
    b.plot([r['length'] for r in cells],[r['max_abs_exported_distortion'] for r in cells],color=BLUE,marker="o",markersize=3)
    b.set_xscale("log",base=2);b.set_xticks([1,4,16,64,128],["1","4","16","64","128"])
    b.set(xlabel="Trajectory length",ylabel="Largest change in\nexported target",ylim=(0,.55));style(b)
    fig.text(.01,.97,"AgentPRM: temporal discount assigned to each state",fontsize=9.5,weight="bold",va="top")
    fig.text(.11,.84,"A  Eight-state success; γ = 0.95",fontsize=8.5);fig.text(.62,.82,"B  Largest target change\n    by trajectory length",fontsize=8.5,va="bottom")
    fig.text(.11,.04,f"Mean in both versions: {np.mean(original):.5f}.",fontsize=8.5)
    save(fig,out,"agentprm_trajectory",["results/agentprm/state_targets_v1.csv","results/agentprm/case_analysis_v1.json"],"How can unchanged aggregates hide state-level supervision changes?")

def toolace(out):
    rows=read_csv("results/toolace/controlled_boundary_sweep_v1.csv")
    fixture=next(r for r in rows if r['id']=="TACE-ORIGINAL" and int(r['prompt_tokens'])==10 and int(r['response_tokens_including_eos'])==4 and int(r['maximum'])==8)
    assert int(fixture['coded_supervised_predictions'])==0 and int(fixture['oracle_supervised_predictions'])==4
    original=[f"P{i}" for i in range(10)]+["R0","R1","R2","EOS"]
    fig,ax=plt.subplots(figsize=(5.5,3.35));fig.subplots_adjust(left=.01,right=.99,bottom=.12,top=.83)
    start=6;labels=["Original sequence","Retained inputs","Next-token labels","Original loss mask","Repaired loss mask"]
    for row,label in enumerate(labels):
        y=4-row;ax.text(-.5,y+.38,label,ha="right",va="center",fontsize=8.5)
        indices=range(14) if row==0 else range(6,14)
        for k in indices:
            unused=row>=2 and k==13
            if row in [0,1]:txt=original[k];fill=PALE_GRAY if k<10 else PALE_ORANGE
            elif row==2:txt=original[k+1] if k<13 else "—";fill=PALE_ORANGE if 9<=k<=12 else PALE_GRAY
            else:
                active=row==4 and 9<=k<=12
                txt="—" if unused else "1" if active else "0";fill=PALE_BLUE if active else PALE_GRAY
            if row==0 and k<6:fill="#E7E8EA"
            ax.add_patch(Rectangle((k,y),.95,.76,facecolor=fill,edgecolor="#AAB1B7",linewidth=.5))
            ax.text(k+.475,y+.38,txt,ha="center",va="center",fontsize=8)
            if row==0 and k<6:ax.plot([k+.12,k+.82],[y+.1,y+.66],color=GRAY,linewidth=.7)
    ax.text(2.8,4.95,"Six prompt tokens removed",ha="center",fontsize=8.5,color=GRAY)
    ax.text(3.5,-.43,"Mask position i supervises retained token i + 1.",ha="center",fontsize=8.5)
    ax.set_xlim(-7,14.1);ax.set_ylim(-.7,5.3);ax.axis("off")
    fig.text(.01,.98,"ToolACE: crop coordinates and next-token prediction",fontsize=9.5,weight="bold",va="top")
    save(fig,out,"toolace_prediction_mask",["results/toolace/controlled_boundary_sweep_v1.csv"],"Why does a left crop remove valid response supervision?")

SCENARIOS={"ambiguous_action_specification":"Ambiguous action","conflicting_constraints":"Conflicting constraints","conflicting_evidence":"Conflicting evidence","critical_tool_failure":"Critical tool failure","emergent_risk_discovery":"Emergent risk","high_stakes_action":"High-stakes action","insufficient_tool_capability":"Insufficient tool capability","missing_critical_parameter":"Missing parameter"}
def abstain(out):
    data=read_json("results/agentabstain/case_analysis_v1.json");ss=data['subtype_sensitivity']['scenarios']
    fig,(a,b)=plt.subplots(1,2,figsize=(5.5,4.05),gridspec_kw={"width_ratios":[1.15,1]});fig.subplots_adjust(left=.34,right=.97,top=.78,bottom=.31,wspace=.34)
    y=np.arange(len(ss));info=[s['counts']['informational'] for s in ss];oper=[s['counts']['operational'] for s in ss]
    a.barh(y,info,color=BLUE,height=.6,label="Informational");a.barh(y,oper,left=info,color=ORANGE,height=.6,label="Operational")
    for i,(x,z) in enumerate(zip(info,oper)):
        a.text(x/2,i,str(x),color="white",ha="center",va="center",fontsize=8.5)
        a.text(x+z/2,i,str(z),color="white",ha="center",va="center",fontsize=8.5)
    a.set_yticks(y,[SCENARIOS[s['category']] for s in ss]);a.invert_yaxis();a.set(xlabel="Released complete pairs",xlim=(0,42),xticks=[0,20,40]);style(a);a.grid(False)
    bounds=[s['max_gap_pp'] for s in ss]
    b.hlines(y,0,bounds,color="#BBC2C8",linewidth=1);b.scatter(bounds,y,color=BLUE,s=22,zorder=3)
    for i,x in enumerate(bounds):b.text(x+.18,i,f"{x:.2f}",ha="left",va="center",fontsize=8.5)
    b.set(ylim=(7.6,-.6),xlim=(0,7.4),xlabel="Maximum gap\n(percentage points)",xticks=[0,3,6],yticks=[]);style(b)
    fig.text(.01,.98,"AgentAbstain: subtype counts and weighting differences",fontsize=9.5,weight="bold",va="top")
    fig.text(.34,.84,"A  Pair counts",fontsize=9);fig.text(.72,.84,"B  Maximum difference",fontsize=9)
    fig.legend(*a.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.60,.015),ncol=2,frameon=False)
    fig.text(.01,.135,f"Maximum overall difference: {data['subtype_sensitivity']['macro_max_gap_pp']:.4f} percentage points.",fontsize=8.5)
    save(fig,out,"agentabstain_composition_bounds",["results/agentabstain/case_analysis_v1.json","results/agentabstain/subtype_sensitivity_v1.csv"],"How much convention sensitivity does released composition allow?")

def supplements(out):
    pref=read_csv("results/agentprm/preference_sensitivity_v1.csv")
    (out/"preference_grid_source_columns.json").write_text(json.dumps(list(pref[0]),indent=2)+"\n")
    census=read_csv("results/toolace/official_loader_census_v1.csv")
    lengths=[int(r['sequence_tokens']) for r in census if r['empty_response']=='False']
    fig,ax=plt.subplots(figsize=(5.5,2.85));fig.subplots_adjust(left=.13,right=.98,top=.82,bottom=.3)
    ax.hist(lengths,bins=np.arange(0,4201,150),color=BLUE,edgecolor="white",linewidth=.3)
    ax.axvline(8192,color=ORANGE,linestyle="--",linewidth=1);ax.text(8092,ax.get_ylim()[1]*.85,"Official limit\n8,192",ha="right",fontsize=9,color=ORANGE)
    ax.set(xlim=(0,8500),xticks=[0,2000,4000,6000,8192],xlabel="Prompt + response + EOS tokens",ylabel="Converted records");style(ax)
    fig.text(.01,.98,"ToolACE: sequence lengths after official conversion",fontsize=9.5,weight="bold",va="top")
    fig.text(.13,.035,f"{len(lengths):,} nonempty records; 570 EOS-only records not shown.\nMaximum sequence length: 3,945 tokens.",fontsize=8.5)
    save(fig,out,"toolace_released_lengths",["results/toolace/official_loader_census_v1.csv"],"Is the crop defect activated at the official configuration?")
    data=read_json("results/agentabstain/case_analysis_v1.json")['subtype_sensitivity']['missing_pair_sensitivity']
    fig,ax=plt.subplots(figsize=(5.5,2.95));fig.subplots_adjust(left=.14,right=.98,top=.82,bottom=.33)
    xx=[r['requested_pairs_removed_per_scenario'] for r in data];yy=[100*r['macro_max_gap'] for r in data]
    ax.plot(xx,yy,color=BLUE,marker='o',markersize=4);ax.set(xlabel="Pairs removed per scenario\n(from one subtype)",ylabel="Maximum macro gap (pp)",xticks=xx,ylim=(0,max(yy)*1.15));style(ax)
    fig.text(.01,.98,"AgentAbstain: sensitivity to missing task pairs",fontsize=9.5,weight="bold",va="top")
    fig.text(.14,.035,"Hypothetical removals from one subtype per scenario.",fontsize=8.5)
    save(fig,out,"agentabstain_missing_pairs",["results/agentabstain/case_analysis_v1.json"],"How does incomplete evaluated support alter the bound?")

def main():
    global ROOT
    p=argparse.ArgumentParser()
    p.add_argument('--results-root',type=Path,default=ROOT/'results')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'figures')
    a=p.parse_args()
    ROOT=a.results_root.resolve().parent
    a.output_dir.mkdir(parents=True,exist_ok=True)
    matrix(a.output_dir);agentprm(a.output_dir);toolace(a.output_dir);abstain(a.output_dir);supplements(a.output_dir)
    (a.output_dir/'figure_provenance.json').write_text(json.dumps(METADATA,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'figures':list(METADATA),'output_dir':str(a.output_dir)},indent=2))

if __name__=='__main__':main()
