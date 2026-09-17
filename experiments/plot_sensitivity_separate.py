"""Export four standalone sensitivity plots per dataset, with in-axis legends."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from experiments.plot_airport_sensitivity import build_figure
from experiments.result_paths import by_format

NAMES = ('lambda_rmse', 'lambda_moments', 'k_rmse', 'k_design_time')


def export_separate(dataset):
    source=Path(f'results/json/{dataset}_inside_sensitivity.json')
    report=json.loads(source.read_text())
    outputs=[]
    for index,name in enumerate(NAMES):
        figure,values=build_figure(report)
        axis=figure.axes[index]
        for other in list(figure.axes):
            if other is not axis:figure.delaxes(other)
        figure.set_size_inches(7,6.5)
        axis.set_position([.17,.15,.79,.80])
        for text in list(axis.texts):text.remove()
        handles,labels=axis.get_legend_handles_labels()
        if index!=1:labels=['INSIDE-Coalition']
        legend=axis.legend(handles,labels,loc='upper right',ncol=2 if index==1 else 1,
                    frameon=True,facecolor='white',edgecolor='#59616C',framealpha=1.0,
                    fancybox=False,fontsize=12,borderaxespad=.8)
        legend.get_frame().set_linewidth(.8)
        # Draw uniform full-span grid lines. Omit only the portions occluded by
        # the opaque legend, so PDF geometry matches the visible rendering.
        axis.grid(False, which='both')
        figure.canvas.draw()
        box=legend.get_window_extent(figure.canvas.get_renderer()).transformed(axis.transAxes.inverted())
        segments=[]
        for direction, ticks in (('x', (*axis.xaxis.get_major_ticks(),*axis.xaxis.get_minor_ticks())),
                                 ('y', (*axis.yaxis.get_major_ticks(),*axis.yaxis.get_minor_ticks()))):
            positions=set()
            for tick in ticks:
                point=(tick.get_loc(),axis.get_ylim()[0]) if direction=='x' else (axis.get_xlim()[0],tick.get_loc())
                pos=axis.transAxes.inverted().transform(axis.transData.transform(point))[0 if direction=='x' else 1]
                if not 0 < pos < 1 or round(pos,12) in positions:continue
                positions.add(round(pos,12))
                covered=(box.x0 <= pos <= box.x1) if direction=='x' else (box.y0 <= pos <= box.y1)
                low,high=(box.y0,box.y1) if direction=='x' else (box.x0,box.x1)
                intervals=((0,max(0,low)),(min(1,high),1)) if covered else ((0,1),)
                for start,end in intervals:
                    if end<=start:continue
                    segments.append([(pos,start),(pos,end)] if direction=='x' else [(start,pos),(end,pos)])
        axis.add_collection(LineCollection(segments,transform=axis.transAxes,
                            colors='#D8DDE4',linewidths=.8,alpha=.85,zorder=.5))
        output=Path(f'results/pdf/{dataset}_sensitivity_{name}')
        for suffix in ('.pdf',):
            target=output.with_suffix(suffix)
            figure.savefig(target,dpi=600,facecolor='white')
            outputs.append(target)
        plt.close(figure)
        by_format(output.with_suffix('.metadata.json')).write_text(json.dumps({
            'dataset':dataset,'plot':name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'panel_tags':False,'legend_location':'inside_upper_right',
            'grid':'full-span uniform horizontal/vertical grid; only opaque legend footprint occludes lines',
            'grid_linewidth_pt':.8, 'grid_color':'#D8DDE4', 'legend_frame':True,
            'uncertainty':'mean +/- sample std across 3 seeds, ddof=1',
            'plotted_values':[v for v in values if v['panel']==index],
            'outputs':{s:hashlib.sha256(output.with_suffix(s).read_bytes()).hexdigest() for s in ('.pdf',)}
        },indent=2)+'\n')
    for path in outputs:print(path,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['airport','wine','cancer'],default=['wine','cancer'])
    args=parser.parse_args()
    for dataset in args.datasets:export_separate(dataset)
