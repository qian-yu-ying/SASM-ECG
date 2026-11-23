"""Reproduce Figure 5 from audited frozen-prediction summary CSVs only."""
from micro_common import *
os.environ['MPLCONFIGDIR']=str(RUN/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    guard();assert (RUN/'MODULE3_COMPLETE.json').exists();pin(10)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':8.5,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    colors={'LP':'#777777','Direct-LP':'#0072B2','TaskRes-ECG/BCE':'#D55E00','MeanAnchor':'#009E73','SASM-Ens':'#CC79A7'}
    short={'LP':'LP','Direct-LP':'Direct-LP','TaskRes-ECG/BCE':'TaskRes-ECG/BCE','MeanAnchor':'MeanAnchor','SASM-Ens':'SASM-Ens'}
    fig,ax=plt.subplots(2,2,figsize=(11.5,9.5),layout='constrained');a,b,c,d=ax.ravel()
    rb=pd.read_csv(OUT/'module3_reliability_bins_ecgf_merl.csv');scope=pd.read_csv(V151/'results/v1.5.1_adapters/scope_summary.csv')
    a.plot([0,1],[0,1],ls='--',color='black',lw=1,label='Perfect calibration')
    for m in ['LP','Direct-LP','TaskRes-ECG/BCE','MeanAnchor']:
        g=rb[(rb.backbone=='ECGFounder')&(rb.method==m)&(rb['count']>0)].sort_values('bin')
        ece=scope[(scope.scope=='ECGFounder')&(scope.method==m)&(scope.calibration=='Raw')].ece.iloc[0]*100
        a.plot(g.mean_predicted_probability,g.empirical_positive_rate,marker='o',ms=3,lw=1.5,color=colors[m],label=f'{short[m]} (ECE {ece:.2f}%)')
    a.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='(a) ECGFounder reliability (Raw)');a.grid(alpha=.15)
    a.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=1,frameon=False)
    df=pd.read_csv(OUT/'module3_pathology_dca_summary.csv');subset='Pathology_Excl_Normal_And_SR'
    for m,cal,style,label in [('MeanAnchor','Raw','-','MeanAnchor (Raw)'),('MeanAnchor','DevTemperature','--','MeanAnchor (DevT)'),('TaskRes-ECG/BCE','Raw','-','TaskRes-ECG/BCE (Raw)'),('Direct-LP','Raw','-','Direct-LP (Raw)'),('LP','Raw','-','LP (Raw)')]:
        g=df[(df.subset==subset)&(df.method==m)&(df.calibration==cal)].groupby('pt').NB.mean();b.plot(g.index,g.values,ls=style,marker='o',ms=3,lw=1.5,color=colors[m],label=label)
    g=df[(df.subset==subset)&(df.method=='MeanAnchor')&(df.calibration=='Raw')].groupby('pt').treat_all_NB.mean()
    b.plot(g.index,g.values,color='black',ls=':',label='Treat all');b.axhline(0,color='black',ls='--',lw=1,label='Treat none')
    b.set(xlabel='Risk threshold',ylabel='Net benefit',title='(b) Exploratory pathology DCA',xticks=[.02,.05,.10,.15,.20]);b.grid(alpha=.15)
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False)
    methods=['LP','Direct-LP','TaskRes-ECG/BCE','MeanAnchor','SASM-Ens'];labels=['LP','Direct-LP','TaskRes','MeanAnchor','SASM-Ens']
    rules=pd.read_csv(V151/'results/v1.5.1_adapters/translational_locked_rules_summary.csv');rules=rules[rules.calibration=='Raw'];x=np.arange(5);w=.24
    for j,(rule,metric,label,color) in enumerate([('A_ChapmanGlobal','specificity_mean','Rule A specificity','#0072B2'),('A_ChapmanGlobal','sensitivity_mean','Rule A sensitivity','#E69F00'),('B_SupportNegative','specificity_mean','Rule B specificity','#009E73')]):
        values=rules[rules.rule==rule].groupby('method')[metric].mean().reindex(methods).to_numpy()*100;c.bar(x+(j-1)*w,values,w,label=label,color=color)
        for idx in [2,3]:c.text(x[idx]+(j-1)*w,values[idx]+1,f'{values[idx]:.1f}',ha='center',va='bottom',rotation=90,fontsize=8)
    c.set(xticks=x,xticklabels=labels,ylim=(0,112),yticks=[0,20,40,60,80,100],ylabel='Macro-average (%)',title='(c) Query-label-free locked thresholds');c.tick_params(axis='x',rotation=15)
    c.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False)
    sub=pd.read_csv(V151/'results/v1.5.1_adapters/subgroup_48slices_summary.csv');groups=['Male','Female','Age_lt60','Age_ge60'];xx=np.arange(4);w=.19
    for j,m in enumerate(methods[:4]):
        vals=sub[sub.method==m].groupby('subgroup').ap_mean.mean().reindex(groups).to_numpy()*100;d.bar(xx+(j-1.5)*w,vals,w,label=short[m],color=colors[m])
    d.set(xticks=xx,xticklabels=['Male','Female','Age <60','Age >=60'],ylim=(0,80),ylabel='Macro-AP (%)',title='(d) Demographic subgroups (12-cond. mean)')
    d.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False)
    for p in [c,d]:p.grid(axis='y',alpha=.15);p.set_axisbelow(True)
    fig.savefig(OUT/'figure5_bspc_v152.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(OUT/'figure5_bspc_v152.png',dpi=300,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    caption=('Figure 5. Frozen-model calibration and exploratory deployment diagnostics. (a) Reliability coordinates pool record-label pairs across 6 ECGFounder conditions (3 cohorts x 2 shots), with 20 fixed support seeds per condition: 120 episode evaluations per method. Legend ECE is the equally condition-averaged classwise macro-ECE, not pooled ECE. (b) Exploratory DCA excludes Normal/NORM and SR endpoints identified in the metadata. Within each of 12 conditions, endpoint-wise NB is macro-averaged and then averaged over 20 support seeds; the 12 conditions are equally weighted. Curves do not establish prospective clinical utility. (c) Values are classwise macro metrics averaged over the 20 seeds within each condition and then equally across 12 conditions (240 episode evaluations per method). Rule A uses a Chapman-only global threshold; Rule B uses support-negative quantiles. External specificity need not attain 95%. (d) Descriptive 12-condition mean subgroup AP, with 20 support seeds per condition; this is not a parity or fairness test. Repeated episodes share fixed query records and are not independent patient cohorts. TaskRes abbreviates TaskRes-ECG/BCE in panel (c); all adapter labels denote the specified ECG/BCE extensions. No query labels were used to select the locked model parameters or Rule A/B thresholds.')
    (OUT/'figure5_caption.txt').write_text(caption+'\n',encoding='utf-8')
    (OUT/'figure5_include.tex').write_text('\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figure5_bspc_v152.pdf}\n\\caption{'+caption.replace('%','\\%')+'}\n\\label{fig:clinical-v152}\n\\end{figure}\n')
    import fitz
    doc=fitz.open(OUT/'figure5_bspc_v152.pdf');assert len(doc)==1;page=doc[0];words=page.get_text('words');bounds=page.rect;outside=[];overlap=[]
    for i,u in enumerate(words):
        rect=fitz.Rect(u[:4])
        if not bounds.contains(rect):outside.append(u[4])
        for j in range(i):
            v=words[j];q=fitz.Rect(v[:4]);inter=rect&q
            if inter.get_area()>.25*min(rect.get_area(),q.get_area()):overlap.append([u[4],v[4]])
    report=dict(passed=not outside and not overlap,vector_pdf=True,pages=1,words=len(words),outside_page=outside,overlapping_word_pairs=overlap,caption_sha256=sha(OUT/'figure5_caption.txt'))
    dump(OUT/'figure5_layout_check.json',report)
    if not report['passed']:raise RuntimeError('Figure layout gate failed; repair native plot before packaging')
    notice('FIGURE5_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:dump(RUN/'status/FIGURE_ERROR.json',dict(error=traceback.format_exc()));raise
