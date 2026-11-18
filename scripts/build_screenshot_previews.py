"""Create transparent 'executed notebook output preview' images from actual run results.

These are designed previews of notebook outputs, not screenshots of Colab/Dicoding UI.
"""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bird_naive_bayes import evaluate as bird_eval
from src.recommendation_ab import evaluate as reco_eval
from src.conversion_ab import evaluate as conv_eval

ROOT=Path(__file__).resolve().parents[1]
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
MONO='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

def getfont(sz, mono=False):
    try:return ImageFont.truetype(MONO if mono else FONT, sz)
    except:return ImageFont.load_default()

def make_preview(name,title,subheading,lines,figure):
    im=Image.new('RGB',(1500,950),'#f8fafc');d=ImageDraw.Draw(im)
    d.rounded_rectangle((30,25,1470,922),radius=22,fill='#ffffff',outline='#cbd5e1',width=2)
    d.rounded_rectangle((30,25,1470,104),radius=22,fill='#12233b')
    d.rectangle((30,82,1470,104),fill='#12233b')
    for x,color in [(65,'#fb7185'),(93,'#fbbf24'),(121,'#34d399')]:d.ellipse((x-9,56-9,x+9,56+9),fill=color)
    d.text((160,44),name,fill='#e2e8f0',font=getfont(21,True))
    d.text((65,136),title,fill='#13213b',font=getfont(32))
    d.text((66,185),subheading,fill='#64748b',font=getfont(19))
    d.rounded_rectangle((66,235,1430,365),radius=13,fill='#0f172a')
    y=253
    for line in lines:
        d.text((90,y),line,fill='#a7f3d0',font=getfont(19,True));y+=30
    chart=Image.open(ROOT/'assets'/'figures'/figure).convert('RGB')
    chart.thumbnail((1350,510))
    # preserve original chart and centre it, genuine output of the runnable case study
    ox=(1500-chart.width)//2;oy=394
    im.paste(chart,(ox,oy))
    d.text((69,887),'Rendered from executed portfolio notebook outputs · synthetic data',fill='#64748b',font=getfont(17))
    path=ROOT/'assets'/'screenshots'/name;im.save(path,optimize=True);print(path)

b=bird_eval();r=reco_eval();z=conv_eval()
make_preview('01-notebook-output.png','Bird classification · executed output',
 'Synthetic feature distributions + log-likelihood Naive Bayes',
 [f'Samples: {len(b["data"])}  |  Train: {len(b["train"])}  |  Test: {len(b["test"])}',
  f'Hold-out accuracy: {b["accuracy"]:.1%}  |  Species: Finch, Sparrow, Swallow'],
 '01-bird-naive-bayes.png')
make_preview('02-notebook-output.png','Recommendation experiment · executed output',
 'Independent groups + Welch two-sided t-test',
 [f'Control mean: {r["control_mean"]:.3f}  |  Treatment mean: {r["treatment_mean"]:.3f}',
  f'Welch t: {r["t_statistic"]:.4f}  |  p-value: {r["p_value"]:.3e}'],
 '02-recommendation-ab.png')
make_preview('03-notebook-output.png','Software conversion · executed output',
 'Pooled-proportion two-sided Z-test',
 [f'Control: {z["control_rate"]:.2%}  |  Treatment: {z["treatment_rate"]:.2%}',
  f'Z statistic: {z["z_statistic"]:.4f}  |  p-value: {z["p_value"]:.6f}'],
 '03-conversion-ab.png')
