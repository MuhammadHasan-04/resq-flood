import json, zipfile, numpy as np, pandas as pd, rasterio, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
O=Path('/home/claude/W/out'); S=Path('/home/claude/W/out_splits'); F=Path('/home/claude/W/report/fig')
plt.rcParams.update({"font.size":10,"axes.titlesize":11,"figure.dpi":110})
d=pd.read_csv(S/'manifest.csv'); t=d[d.dataset=='tuwien_pakistan_2022']; s=d[d.dataset!='tuwien_pakistan_2022']
def rd(cid, kind):
    r=d[d.chip_id==cid].iloc[0]
    with rasterio.open(O/(r.sar_file if kind=='sar' else r.label_file)) as f: a=f.read()
    if kind=='sar': a=np.where(a==-32768,np.nan,a/100.)
    return a
# 1 map
if not Path('/home/claude/W/gadm').exists():
    zipfile.ZipFile('/mnt/user-data/uploads/th/outputs/datasets/gadm_pakistan_boundaries/gadm41_PAK_shp.zip').extractall('/home/claude/W/gadm')
pk=gpd.read_file('/home/claude/W/gadm/gadm41_PAK_1.shp')
fig,ax=plt.subplots(figsize=(8,8)); pk.boundary.plot(ax=ax,color='#888',lw=.6)
col={'train':'#2b7bba','val':'#f39c12','test':'#c0392b','excluded_buffer':'#999','train_aux_s1f_weak':'#27ae60','test_external_s1f_hand':'#8e44ad'}
lab={'train':'TU Wien train','val':'TU Wien val','test':'TU Wien test (incl. Dadu)','excluded_buffer':'buffer (dropped)','train_aux_s1f_weak':'Sen1Floods11 weak (aux train)','test_external_s1f_hand':'Sen1Floods11 hand (external test)'}
for k,g in d.groupby('split'):
    u=g.drop_duplicates(['lon','lat']); ax.scatter(u.lon,u.lat,s=14,c=col[k],label=f"{lab[k]} ({len(g)} chips)",edgecolor='k',lw=.2)
ax.set_xlim(60.5,78);ax.set_ylim(23.5,37.5) if False else ax.set_ylim(21,37.2);ax.set_xlim(60.5,77.9)
ax.legend(loc='upper left',fontsize=8,frameon=True);ax.set_title('Training chips across Pakistan (each dot = 9 x 9 km chip, 512 x 512 px)')
ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');plt.tight_layout();plt.savefig(F/'f1_chip_map_splits.png',dpi=130);plt.close()
# 2 raw vs Lee
cid='TW0208'; a=rd(cid,'sar'); L=rd(cid,'lab')
fig,ax=plt.subplots(2,3,figsize=(13,8.6))
for i,(b,tl) in enumerate([(0,'VV raw (dB)'),(2,'VV Lee 7x7 (dB)')]):
    ax[i,0].imshow(a[b],cmap='gray',vmin=-22,vmax=-2);ax[i,0].set_title(f'{cid} {tl}')
    ax[i,1].imshow(a[b][150:278,150:278],cmap='gray',vmin=-22,vmax=-2);ax[i,1].set_title(tl+' - zoom 128 px')
    w=L[0]==1;l=L[0]==0
    ax[i,2].hist(a[b][l&np.isfinite(a[b])],bins=120,range=(-28,2),alpha=.6,density=True,label='dry (label 0)',color='#c98d26')
    ax[i,2].hist(a[b][w&np.isfinite(a[b])],bins=120,range=(-28,2),alpha=.6,density=True,label='water (label 1)',color='#1f6fb2')
    ax[i,2].set_title(tl+' histogram');ax[i,2].legend(fontsize=8)
for x in ax[:,:2].ravel(): x.axis('off')
plt.tight_layout();plt.savefig(F/'f2_raw_vs_lee.png',dpi=120);plt.close()
# 3 Lee effect on Otsu agreement
fig,ax=plt.subplots(1,2,figsize=(12,4.2))
order=['none','low','mid','high']
ax[0].boxplot([t[t.sample_bin==b].otsu_iou_raw for b in order],positions=np.arange(4)-.18,widths=.3,patch_artist=True,boxprops=dict(facecolor='#ddd'))
ax[0].boxplot([t[t.sample_bin==b].otsu_iou_lee for b in order],positions=np.arange(4)+.18,widths=.3,patch_artist=True,boxprops=dict(facecolor='#7fb3e0'))
ax[0].set_xticks(range(4),[f'{b}\n(n={sum(t.sample_bin==b)})' for b in order]);ax[0].set_ylabel('IoU of Otsu(VV) vs TU Wien label')
ax[0].set_title('Same simple threshold: raw (grey) vs Lee 7x7 (blue)');ax[0].set_xlabel('chip flood-fraction bin')
dd=(t.otsu_iou_lee-t.otsu_iou_raw); ax[1].hist(dd,bins=40,color='#2b7bba');ax[1].axvline(0,color='k')
ax[1].set_title(f'IoU gain from Lee per chip: median {dd.median():+.3f}, better in {(dd>0).mean()*100:.0f}% of chips');ax[1].set_xlabel('IoU(Lee) - IoU(raw)')
plt.tight_layout();plt.savefig(F/'f3_lee_effect_otsu.png',dpi=120);plt.close()
# 4 permanent water removal
cmw=ListedColormap(['#f2efe6','#1f6fb2']);
def lab_rgb(x,c1):
    r=np.zeros(x.shape+(3,));r[x==0]=(0.95,0.93,0.88);r[x==1]=c1;r[x==255]=(0.6,0.6,0.6);return r
for cid,fn in (('TW0215','f4_permanent_water_removal.png'),('TW0100','f4b_permanent_water_removal.png')):
    a=rd(cid,'sar');L=rd(cid,'lab');r=d[d.chip_id==cid].iloc[0]
    fig,ax=plt.subplots(1,5,figsize=(19,4.4))
    ax[0].imshow(a[2],cmap='gray',vmin=-22,vmax=-2);ax[0].set_title(f'{cid} {r.date} VV Lee (dB)')
    ax[1].imshow(lab_rgb(L[2],(0.05,0.2,0.6)));ax[1].set_title(f'JRC permanent water ({r.perm_frac*100:.1f}%)')
    ax[2].imshow(lab_rgb(L[0],(0.12,0.44,0.7)));ax[2].set_title(f'label_water = flood + permanent ({r.water_frac*100:.1f}%)')
    ax[3].imshow(lab_rgb(L[1],(0.85,0.2,0.15)));ax[3].set_title(f'label_flood: permanent -> ignore ({r.flood_frac*100:.1f}%)')
    ax[4].imshow(lab_rgb(L[3],(0.2,0.6,0.3)));ax[4].set_title(f'valid mask ({r.valid_frac*100:.0f}%), grey = ignore')
    for x in ax: x.axis('off')
    plt.tight_layout();plt.savefig(F/fn,dpi=110);plt.close()
# 5 balance
fig,ax=plt.subplots(1,3,figsize=(15,4))
c=d.groupby(['dataset','split']).size().unstack(0).fillna(0);c.plot.barh(ax=ax[0],stacked=True,color=['#27ae60','#2b7bba']);ax[0].set_title('Chips per split');ax[0].set_xlabel('chips')
ax[1].hist(t.flood_frac*100,bins=40,color='#c0392b');ax[1].set_title('TU Wien chips: flood % per chip');ax[1].set_xlabel('flooded % of valid px')
tc=t.groupby('date').size();ax[2].bar(range(len(tc)),tc.values,color='#2b7bba');ax[2].set_xticks(range(len(tc)),[x[4:6]+'-'+x[6:] for x in tc.index.astype(str)],rotation=60);ax[2].set_title('TU Wien chips per acquisition date (2022, mm-dd)')
plt.tight_layout();plt.savefig(F/'f5_balance.png',dpi=120);plt.close()
# 6 label QC
fig,ax=plt.subplots(figsize=(6.5,5.5))
for (k,g),c_ in zip(d.groupby('label_type'),['#8e44ad','#2b7bba','#27ae60']):
    ax.scatter(g.vv_land_mean_db,g.vv_water_mean_db,s=10,label=f'{k} (n={len(g)})',c=c_,alpha=.7)
x=np.linspace(-18,-5);ax.plot(x,x,'k--',lw=.8);ax.plot(x,x-2,'r:',lw=1,label='QC limit: water must be >= 2 dB darker')
ax.set_xlabel('mean VV (Lee) of dry pixels, dB');ax.set_ylabel('mean VV (Lee) of water pixels, dB');ax.legend(fontsize=8);ax.set_title('Does the label agree with the radar? (per chip)')
plt.tight_layout();plt.savefig(F/'f6_label_qc.png',dpi=120);plt.close()
# 7 gallery
ids=['TW0284','TW0138','TW0404','TW0049','S1F_hand_'+s[s.label_type=='hand'].sort_values('water_frac').chip_id.iloc[-1].split('_')[-1]]
fig,ax=plt.subplots(2,len(ids),figsize=(4*len(ids),8))
for j,cid in enumerate(ids):
    a=rd(cid,'sar');L=rd(cid,'lab');r=d[d.chip_id==cid].iloc[0]
    rgb=np.dstack([np.clip((a[2]+22)/20,0,1),np.clip((a[3]+28)/20,0,1),np.clip((a[2]-a[3]-2)/12,0,1)])
    ax[0,j].imshow(np.nan_to_num(rgb));ax[0,j].set_title(f'{cid} ({r.split})\nRGB = VV, VH, VV-VH (Lee)',fontsize=9)
    ov=np.nan_to_num(rgb).copy();ov[L[1]==1]=ov[L[1]==1]*.3+np.array([1,0.1,0.1])*.7;ov[L[2]==1]=ov[L[2]==1]*.3+np.array([0.1,0.3,1])*.7;ov[L[3]==0]=0.6
    ax[1,j].imshow(ov);ax[1,j].set_title(f'red flood {r.flood_frac*100:.0f}%, blue permanent, grey ignore',fontsize=9)
for x in ax.ravel(): x.axis('off')
plt.tight_layout();plt.savefig(F/'f7_gallery.png',dpi=110);plt.close()
print('ok')
