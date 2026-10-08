"""Pairwise + third-order scalability: matched interior budget / n^2 across sizes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
import numpy as np
from experiments.run_interaction_baselines import METHODS
from experiments.plot_inside_four_panel import (PLOT_METHOD_STYLES, DISPLAY_LABELS,
    _legend_handles, _clip_series_to_viewport, _calls_label)
from experiments.result_paths import by_format


def summaries(report):
    cfg=report['configuration']; expected={(degree,b,r,m) for degree in cfg.get('degrees',[2,3])
        for b in cfg['budgets'] for r in range(cfg['repeats']) for m in cfg['methods']}
    actual=[(c['degree'],c['multiplier'],c['repeat'],c['method']) for c in report['cells']]
    if report['status']!='complete' or set(actual)!=expected or len(actual)!=len(expected):
        raise ValueError('plot requires a complete, nonduplicated comparison grid')
    result=[]
    for degree in cfg.get('degrees',[2,3]):
        truth=np.asarray(next(g['truth'] for g in report['games'] if g['degree']==degree))
        for method in cfg['methods']:
            for b in cfg['budgets']:
                cells=[c for c in report['cells'] if (c['degree'],c['method'],c['multiplier'])==(degree,method,b)]
                errors=np.array([np.sqrt(np.mean((np.asarray(c['estimate'])-truth)**2)) for c in cells])
                np.testing.assert_allclose(errors,[c['rmse'] for c in cells],rtol=1e-12,atol=1e-16)
                if any(c['actual_calls']>c['target_calls'] for c in cells):raise ValueError('budget exceeded')
                result.append({'degree':degree,'method':method,'multiplier':b,
                    'actual_calls':float(np.mean([c['actual_calls'] for c in cells])),
                    'mean':float(errors.mean()),'std':float(errors.std(ddof=1)) if len(errors)>1 else 0.0})
    return result


DISPLAY_METHODS = ('inside_greedy', 'inside_orbit', 'ofa', 'cc', 's_diff')


def plot(sources, output, audit_alignment=False):
    reports=[json.loads(source.read_text()) for source in sources]
    if len({r['configuration']['players'] for r in reports}) != len(reports):
        raise ValueError('scalability sources must have distinct player counts')
    rows=[dict(row,players=report['configuration']['players'])
          for report in reports for row in summaries(report)
          if row['degree']==3 and row['method'] in DISPLAY_METHODS
          and row['multiplier'] in [report['configuration']['players']*k for k in (1,2,4)]]
    methods=list(DISPLAY_METHODS)
    for report in reports:
        cfg=report['configuration']
        if not all(n in cfg['budgets'] for n in [cfg['players']*k for k in (1,2,4)]):
            raise ValueError('missing required normalized budget')
        if any(m not in cfg['methods'] for m in methods):
            raise ValueError('missing comparison method')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':13,'svg.fonttype':'none','pdf.fonttype':42,'axes.labelcolor':'#303640',
        'xtick.color':'#4B5563','ytick.color':'#4B5563'})
    fig,axes=plt.subplots(1,len(reports),figsize=(17,6.3))
    fig.subplots_adjust(left=.065,right=.99,bottom=.14,top=.80,wspace=.30)
    display=lambda m:'ofa_iid_ratio' if m=='ofa' else m
    for ax,report in zip(axes,reports):
        cfg=report['configuration'];n=cfg['players']
        budgets=[n*k for k in (1,2,4)]
        limits=(min(budgets)*n*.80+2*n+2,max(budgets)*n*1.22+2*n+2)
        panel=[r for r in rows if r['players']==n]
        positive=[r['mean']-r['std'] for r in rows if r['mean']-r['std']>0]
        positive += [r['mean'] for r in rows if r['mean']>0]
        floor=min(positive)*.5
        if any(r['mean']<=0 for r in panel):
            raise ValueError('exact zero RMSE needs an explicit plot policy; never silently floor measured errors')
        upper=max(r['mean']+r['std'] for r in rows)*1.5
        for method in methods:
            series=sorted([r for r in panel if r['method']==method],key=lambda r:r['actual_calls'])
            x=np.array([r['actual_calls'] for r in series]); y=np.array([r['mean'] for r in series]);sd=np.array([r['std'] for r in series])
            style=PLOT_METHOD_STYLES[display(method)];focal=method.startswith('inside_')
            bx,lower,_=_clip_series_to_viewport(x,np.maximum(y-sd,floor),limits)
            _,higher,_=_clip_series_to_viewport(x,y+sd,limits)
            ax.fill_between(bx,lower,higher,color=style['color'],alpha=.16,linewidth=0,zorder=2)
            px,py,markers=_clip_series_to_viewport(x,y,limits)
            ax.plot(px,py,color=style['color'],marker=style['marker'],markevery=markers,
                linestyle=style['linestyle'],linewidth=style['linewidth'],markersize=7.5,
                markerfacecolor=style['color'] if focal else 'white',markeredgewidth=1.25,
                zorder=4 if focal else 3)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(*limits);ax.set_ylim(floor,upper)
        ax.xaxis.set_major_locator(FixedLocator([b*n+2*n+2 for b in budgets]))
        ax.xaxis.set_major_formatter(FuncFormatter(_calls_label));ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True,which='major',color='#D8DDE4',linewidth=1,alpha=.92)
        ax.grid(True,which='minor',axis='y',color='#EEF1F4',linewidth=.7,alpha=.82)
        ax.set_axisbelow(True);ax.set_box_aspect(1)
        for spine in ax.spines.values():spine.set_color('#555E69');spine.set_linewidth(1.25)
        ax.text(.035,1.025,f'n = {n}',
                transform=ax.transAxes,va='bottom',fontsize=13.5,fontweight='semibold')
    fig.legend(handles=_legend_handles(tuple(display(m) for m in methods)),
               loc='upper center',bbox_to_anchor=(.535,.985),ncol=len(methods),frameon=False,fontsize=10.8)
    fig.supylabel('RMSE',x=.012,fontsize=15)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    y=min(ax.xaxis.get_tightbbox(renderer).y0 for ax in axes)/fig.bbox.height-8/(fig.get_figheight()*72)
    fig.supxlabel('Utility calls',x=.535,y=y,va='top',fontsize=15)
    if audit_alignment:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=by_format(output.with_suffix('.alignment.json')),
            overlay_svg=None,tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,facecolor='white');plt.close(fig)
    metadata={'sources':[{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()} for source in sources],
        'outputs':{'pdf':str(output)},'statistics':'mean +/- sample SD, ddof=1, per-seed RMSE',
        'rows':rows,'x_axis':'mean actual utility calls; clip only outside common target-budget viewport',
        'display_methods':methods,
        'omitted_methods':[m for m in cfg['methods'] if m not in methods],
        'omission_reason':'user-requested presentation subset; all source measurements retained',
        'y_axis':'log; no measured means floored; nonpositive lower SD bounds clipped to viewport'}
    by_format(output.with_suffix('.metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    csv_path=by_format(output.with_suffix('.csv'))
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',type=Path,default=[
        Path('results/json/interaction_selected_baselines_n500_extended.json'),
        Path('results/json/interaction_scalability_n1000.json'),
        Path('results/json/interaction_scalability_n2000.json')])
    p.add_argument('--output',type=Path,default=Path('results/pdf/interaction_scalability_rmse.pdf'))
    p.add_argument('--audit-alignment',action='store_true')
    args=p.parse_args();plot(args.sources,args.output,args.audit_alignment)
