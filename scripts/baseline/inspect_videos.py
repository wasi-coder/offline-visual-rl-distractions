from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
import json
root=Path(__file__).resolve().parents[2]
paths=[root/'videos'/n for n in ['bc_seed0.mp4','drq_seed0.mp4','consistency_seed0.mp4','infogating_seed0.mp4']]
canvas=Image.new('RGB',(768,4*282),'white');draw=ImageDraw.Draw(canvas);records=[]
for i,p in enumerate(paths):
 reader=imageio.get_reader(p);meta=reader.get_meta_data()
 for j,idx in enumerate([0,249,499]):
  frame=reader.get_data(idx);canvas.paste(Image.fromarray(frame),(j*256,i*282+26))
  draw.text((j*256+5,i*282+5),p.stem+' frame '+str(idx),fill='black')
 records.append({'video':str(p),'fps':meta['fps'],'size':meta['size'],'duration':meta.get('duration'),'sampled_frames':[0,249,499]});reader.close()
canvas.save(root/'reports/video_inspection.png')
(root/'reports/video_inspection.json').write_text(json.dumps(records,indent=2))
print(records)
