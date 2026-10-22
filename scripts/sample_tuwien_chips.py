"""Sample 512 x 512 px training chips from the TU Wien FLOOD-HM-MASKED masks, stratified by flood cover.
  python scripts/sample_tuwien_chips.py --masks <folder with the TU Wien mask .tif files> --out ../outputs/FLOOD_TRAINING_DATA/00_raw_downloads
Writes tuwien_inventory.csv and chip_candidates.csv to --out."""
import argparse
import rasterio, numpy as np, glob, re, json, os, pandas as pd
from rasterio.warp import transform as wtransform
_ap = argparse.ArgumentParser()
_ap.add_argument("--masks", required=True, help="folder with the TU Wien FLOOD-HM-MASKED .tif files")
_ap.add_argument("--out", default="../outputs/FLOOD_TRAINING_DATA/00_raw_downloads")
_a = _ap.parse_args()
os.makedirs(_a.out, exist_ok=True)
rng=np.random.default_rng(42)
B=512  # 512 px blocks at 20 m = 10.24 km
rows=[]; inv=[]
for f in sorted(glob.glob(os.path.join(_a.masks, '*.tif'))):
    n=os.path.basename(f); m=re.search(r'MASKED_(\d{8}T\d{6})__VV_([AD]\d{3})_(E\d{3}N\d{3}T3)',n)
    ts,orb,tile=m.groups()
    with rasterio.open(f) as s:
        a=s.read(1); H,W=a.shape
        nb=H//B
        blk=a[:nb*B,:nb*B].reshape(nb,B,nb,B).transpose(0,2,1,3).reshape(nb,nb,B*B)
        valid=(blk!=255).mean(-1); flood=(blk==1).sum(-1)/np.maximum(1,(blk!=255).sum(-1))
        inv.append(dict(file=n,ts=ts,orbit=orb,tile=tile,valid_pct=round(100*(a!=255).mean(),1),flood_km2=round((a==1).sum()*0.0004,1)))
        cand=[(i,j,valid[i,j],flood[i,j]) for i in range(nb) for j in range(nb) if valid[i,j]>=0.7]
        bins={'none':[c for c in cand if c[3]<0.005],'low':[c for c in cand if 0.005<=c[3]<0.1],'mid':[c for c in cand if 0.1<=c[3]<0.4],'high':[c for c in cand if c[3]>=0.4]}
        want={'none':1,'low':2,'mid':2,'high':2}
        for k,lst in bins.items():
            for idx in rng.permutation(len(lst))[:want[k]]:
                i,j,v,fl=lst[idx]
                x,y=s.transform*((j+0.5)*B,(i+0.5)*B)
                lon,lat=wtransform(s.crs,'EPSG:4326',[x],[y]); 
                rows.append(dict(ts=ts,orbit=orb,tile=tile,mask_file=n,bi=i,bj=j,valid=round(float(v),3),flood_frac=round(float(fl),4),bin=k,lon=round(lon[0],5),lat=round(lat[0],5)))
I=pd.DataFrame(inv); C=pd.DataFrame(rows)
I.to_csv(os.path.join(_a.out,'tuwien_inventory.csv'),index=False); C.to_csv(os.path.join(_a.out,'chip_candidates.csv'),index=False)
print(len(I),'masks', I.ts.nunique(),'acquisitions', I.tile.nunique(),'tiles', I.ts.str[:8].nunique(),'dates')
print(len(C),'chips'); print(C.bin.value_counts().to_dict()); print(C.groupby('tile').size().to_dict())
print('lon',C.lon.min(),C.lon.max(),'lat',C.lat.min(),C.lat.max())
