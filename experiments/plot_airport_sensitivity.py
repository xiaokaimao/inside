"""Shared Appendix four-panel sensitivity plotter; Airport remains the default."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LogLocator, FuncFormatter
import numpy as np
from experiments.run_airport_sensitivity import validate
from experiments.result_paths import by_format


def build_figure(report):
    validate(report)
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':12,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(21.8,6.2))
    fig.subplots_adjust(left=.057,right=.995,bottom=.20,top=.97,wspace=.30)
    lam=sorted((r for r in report['summary'] if r['candidate_pool']==64),key=lambda r:r['lambda0'])
    pools=sorted((r for r in report['summary'] if r['lambda0']==1/16),key=lambda r:r['candidate_pool'])
    specs=[(lam,'lambda0',[('rmse','RMSE','#D97700')]),
           (lam,'lambda0',[('D1',r'$D_1$','#0072B2'),('D2',r'$D_2$','#C45100')]),
           (pools,'candidate_pool',[('rmse','RMSE','#D97700')]),
           (pools,'candidate_pool',[('design_seconds','Design time','#0072B2')])]
    metadata=[]
    for i,(ax,(rows,xkey,metrics)) in enumerate(zip(axes,specs)):
        x=np.array([r[xkey] for r in rows]);bounds=[]
        for metric,label,color in metrics:
            means=np.array([r['statistics'][metric]['mean'] for r in rows])
            std=np.array([r['statistics'][metric]['std'] for r in rows])
            lower,upper=means-std,means+std
            if np.any(lower<=0):raise ValueError('Nonpositive SD bounds need an explicit log-axis policy')
            ax.fill_between(x,lower,upper,color=color,alpha=.17,linewidth=0)
            ax.plot(x,means,color=color,marker='o',linewidth=2.5,markersize=6,label=label)
            bounds.extend(lower.tolist()+upper.tolist())
            chosen=np.flatnonzero(x==(1/16 if xkey=='lambda0' else 64))[0]
            ax.scatter(x[chosen],means[chosen],s=105,facecolors='none',edgecolors='#222222',linewidths=1.4,zorder=5)
            metadata.append({'panel':i,'metric':metric,'x':x.tolist(),'mean':means.tolist(),
                             'std':std.tolist(),'lower':lower.tolist(),'upper':upper.tolist()})
        ax.set_xscale('log',base=2);ax.set_yscale('log')
        ax.set_xticks(x, [f'1/{int(round(1/v))}' if v<1 else '1' for v in x] if i<2 else [str(int(v)) for v in x])
        ax.xaxis.set_minor_locator(NullLocator())
        lo,hi=np.log10(min(bounds)),np.log10(max(bounds));span=max(hi-lo,.18)
        ax.set_ylim(10**(lo-.12*span),10**(hi+.40*span))
        ax.set_xlim(x[0]/1.23,x[-1]*1.23)
        if i in (1,3):
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,2,5)))
            ax.yaxis.set_major_formatter(FuncFormatter(
                (lambda v, _: f'{v:.2g}') if i==3 else (lambda v, _: f'{v:.0e}')))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis='x',labelsize=11)
        ax.set_box_aspect(1)
        ax.set_xlabel(r'$\lambda$' if i<2 else r'Candidate-pool size $K$',fontsize=14)
        ax.set_ylabel(['RMSE','Moment discrepancy','RMSE','Design time (s)'][i],fontsize=14)
        ax.set_axisbelow(True)
        ax.grid(True, which='major', color='#D8DDE4', linewidth=.8, alpha=.85)
        ax.grid(True, which='minor', axis='y', color='#EEF1F4', linewidth=.5, alpha=.7)
        for spine in ax.spines.values():spine.set_color('#59616C')
    shared_limits=(min(axes[0].get_ylim()[0],axes[2].get_ylim()[0]),
                   max(axes[0].get_ylim()[1],axes[2].get_ylim()[1]))
    axes[0].set_ylim(*shared_limits);axes[2].set_ylim(*shared_limits)
    handles,labels=axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc='upper right', ncol=2,
                   frameon=False, fontsize=12, borderaxespad=.7)
    # Keep a clear header strip for the in-panel legend.
    fig.canvas.draw()
    axis = axes[1]
    for tick in (*axis.xaxis.get_major_ticks(), *axis.xaxis.get_minor_ticks()):
        tick.gridline.set_ydata([0, .83])
    for tick in (*axis.yaxis.get_major_ticks(), *axis.yaxis.get_minor_ticks()):
        location = axis.transAxes.inverted().transform(axis.transData.transform((1, tick.get_loc())))[1]
        if location > .83:
            tick.gridline.set_visible(False)
    return fig,metadata


def export(source:Path,output:Path,audit_alignment=False):
    report=json.loads(source.read_text());fig,values=build_figure(report)
    with by_format(output.with_suffix('.csv')).open('w', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['lambda0','effective_lambda','K','repeats','metric','mean','sample_std'])
        for row in report['summary']:
            for metric, stat in row['statistics'].items():
                writer.writerow([row['lambda0'],row['effective_lambda'],row['candidate_pool'],
                                 row['repeats'],metric,stat['mean'],stat['std']])
    if audit_alignment:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=by_format(output.with_suffix('.alignment.json')),
            overlay_svg=None,tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    for suffix in ('.pdf',):
        path=output.with_suffix(suffix);fig.savefig(path,dpi=600,facecolor='white')
    plt.close(fig)
    meta={'panel_tags':False, 'legend_location':'inside_moment_panel_upper_right',
          'grid':'major grid with light minor-y grid; legend header kept clear',
          'configuration':report['configuration'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'plotted_values':values,'default_marker':'black outer ring at lambda0=1/16 or K=64',
          'units':'one independent estimator seed per replicate, fixed game and budget',
          'uncertainty':'sample standard deviation (ddof=1), not SE or confidence interval',
          'outputs':{s:hashlib.sha256(output.with_suffix(s).read_bytes()).hexdigest() for s in ('.pdf',)}}
    by_format(output.with_suffix('.metadata.json')).write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=Path('results/json/airport_inside_sensitivity.json'))
    parser.add_argument('--output',type=Path,default=Path('results/pdf/airport_inside_sensitivity.pdf'))
    parser.add_argument('--audit-alignment',action='store_true')
    args=parser.parse_args();export(args.report,args.output,args.audit_alignment)
