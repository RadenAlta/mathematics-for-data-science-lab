"""Generate genuine PNG figures from the runnable portfolio examples."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from .bird_naive_bayes import evaluate as bird_evaluate
from .recommendation_ab import evaluate as reco_evaluate
from .conversion_ab import evaluate as conv_evaluate

ROOT=Path(__file__).resolve().parents[1]
FIG_DIR=ROOT/'assets'/'figures'
FIG_DIR.mkdir(parents=True,exist_ok=True)


def save_bird_chart():
    d=bird_evaluate()
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    for species,g in d['data'].groupby('species'):
        axes[0].hist(g['wingspan_cm'],bins=24,alpha=.55,label=species,density=True)
    axes[0].set(title='Synthetic wing-span distributions',xlabel='Wingspan (cm)',ylabel='Density')
    axes[0].legend(frameon=False)
    matrix=d['confusion'].to_numpy()
    im=axes[1].imshow(matrix,cmap='Blues')
    labels=d['confusion'].index.tolist()
    axes[1].set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
                xlabel='Predicted species',ylabel='Actual species',title=f"Held-out confusion matrix · accuracy {d['accuracy']:.1%}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j,i,str(matrix[i,j]),ha='center',va='center',
                         color='white' if matrix[i,j]>matrix.max()/2 else 'black',fontsize=13)
    fig.colorbar(im,ax=axes[1],fraction=.046,pad=.04)
    fig.suptitle('01  |  Distribution-aware Naive Bayes classification',fontsize=17,fontweight='bold')
    out=FIG_DIR/'01-bird-naive-bayes.png';fig.savefig(out,dpi=145,bbox_inches='tight');plt.close(fig)
    return out,d


def save_reco_chart():
    r=reco_evaluate()
    a,b=r['groups']
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    names=['Control','Recommendation']
    means=[a.mean(),b.mean()]
    ci=[1.96*a.std(ddof=1)/len(a)**.5,1.96*b.std(ddof=1)/len(b)**.5]
    axes[0].bar(names,means,yerr=ci,capsize=8,alpha=.8)
    axes[0].set(title='Mean articles read per user',ylabel='Mean articles · 95% CI (approx.)',ylim=(0,max(means)*1.2))
    for i,v in enumerate(means): axes[0].text(i,v+.12,f'{v:.2f}',ha='center',fontweight='bold')
    bins=np.arange(0,max(a.max(),b.max())+2)-.5
    axes[1].hist(a,bins=bins,alpha=.55,label='Control',density=True)
    axes[1].hist(b,bins=bins,alpha=.55,label='Recommendation',density=True)
    axes[1].set(title=f"Distribution of article reads · Welch t={r['t_statistic']:.2f}",
                xlabel='Articles read',ylabel='Probability density')
    axes[1].legend(frameon=False)
    fig.suptitle('02  |  Recommendation feature A/B experiment',fontsize=17,fontweight='bold')
    out=FIG_DIR/'02-recommendation-ab.png';fig.savefig(out,dpi=145,bbox_inches='tight');plt.close(fig)
    return out,r


def save_conversion_chart():
    r=conv_evaluate()
    rates=[r['control_rate']*100,r['treatment_rate']*100]
    ci=[100*1.96*np.sqrt(r['control_rate']*(1-r['control_rate'])/r['control_users']),
        100*1.96*np.sqrt(r['treatment_rate']*(1-r['treatment_rate'])/r['treatment_users'])]
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    axes[0].bar(['Control','New feature'],rates,yerr=ci,capsize=8,alpha=.8)
    axes[0].set(title='Observed conversion rates',ylabel='Conversion rate (%)',ylim=(0,85))
    for i,v in enumerate(rates): axes[0].text(i,v+2,f'{v:.2f}%',ha='center',fontweight='bold')
    x=np.linspace(-4.5,4.5,500)
    axes[1].plot(x,norm.pdf(x),lw=2)
    xleft=x[x<=r['z_statistic']]
    axes[1].fill_between(xleft,0,norm.pdf(xleft),alpha=.35)
    xright=x[x>=abs(r['z_statistic'])]
    axes[1].fill_between(xright,0,norm.pdf(xright),alpha=.35)
    axes[1].axvline(r['z_statistic'],ls='--')
    axes[1].set(xlabel='Z score',ylabel='Standard normal density',
                title=f"Two-sided z-test: z={r['z_statistic']:.4f}, p={r['p_value']:.4f}")
    fig.suptitle('03  |  Software feature conversion A/B experiment',fontsize=17,fontweight='bold')
    out=FIG_DIR/'03-conversion-ab.png';fig.savefig(out,dpi=145,bbox_inches='tight');plt.close(fig)
    return out,r


def save_review_summary():
    fig,ax=plt.subplots(figsize=(13,4.8),layout='constrained')
    labels=['Naive Bayes','Recommendation A/B','Software feature A/B']
    counts=[5,5,5]
    ax.barh(labels,counts,alpha=.85)
    ax.set(xlim=(0,5.8),xlabel='Correct answers (out of 5)',title='Knowledge Check — independently reviewed course achievement')
    ax.invert_yaxis()
    for i,c in enumerate(counts): ax.text(c+.13,i,f'{c}/5',va='center',fontweight='bold',fontsize=13)
    fig.suptitle('Official course outcome: Advanced · 4.0 / 4.0',fontweight='bold',fontsize=19)
    out=FIG_DIR/'04-course-achievement.png';fig.savefig(out,dpi=145,bbox_inches='tight');plt.close(fig)
    return out


def main():
    a, bird=save_bird_chart();b,reco=save_reco_chart();c,conv=save_conversion_chart();d=save_review_summary()
    print('Bird accuracy:',bird['accuracy'])
    print('Recommendation mean:',round(reco['control_mean'],4),round(reco['treatment_mean'],4),
          't=',round(reco['t_statistic'],4),'p=',reco['p_value'])
    print('Conversion rates:',conv['control_rate'],conv['treatment_rate'],
          'z=',conv['z_statistic'],'p=',conv['p_value'])
    print('Figures:');print('\n'.join(map(str,[a,b,c,d])))

if __name__=='__main__':main()
