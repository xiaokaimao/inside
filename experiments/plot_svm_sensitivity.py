"""Export the Wine and Breast Cancer parameter-sensitivity four-panel figures."""
import argparse
from pathlib import Path
from experiments.plot_airport_sensitivity import export

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['wine','cancer'],default=['wine','cancer'])
    parser.add_argument('--audit-alignment',action='store_true')
    args=parser.parse_args()
    for dataset in args.datasets:
        export(Path(f'results/json/{dataset}_inside_sensitivity.json'),
               Path(f'results/pdf/{dataset}_inside_sensitivity.pdf'),args.audit_alignment)
