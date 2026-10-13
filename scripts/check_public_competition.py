"""Read public competition metadata and published rules through official API.

No host settings, hidden assets, submission mutation or leaderboard probing.
"""
import json
from datetime import datetime,timezone
from pathlib import Path
from download_data import authenticated_session,COMPETITION

def main():
    session=authenticated_session();base=f'https://www.kaggle.com/api/v1/competitions/{COMPETITION}'
    out={'checked_UTC':datetime.now(timezone.utc).isoformat(),'source':'official public Kaggle API','competition':COMPETITION}
    metadata=f'https://www.kaggle.com/api/v1/competitions/get/{COMPETITION}'
    for name,url,params in [('metadata',metadata,None),('rules',base+'/pages',{'pageName':'rules'})]:
        response=session.get(url,params=params,timeout=(30,60));out[name+'_HTTP_status']=response.status_code
        if response.status_code!=200:continue
        value=response.json()
        if name=='metadata':
            fields=['deadline','mergerDeadline','newEntrantDeadline','maxDailySubmissions','maxTeamSize','submissionsDisabled','evaluationMetric']
            out['metadata']={k:value.get(k) for k in fields}
        else:
            pages=value.get('pages',[]) if isinstance(value,dict) else value
            public=[]
            for page in pages:
                if page.get('name',page.get('pageName'))!='rules':continue
                public.append(page)
            Path('data').mkdir(exist_ok=True)
            Path('data/published_rules_response.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
            out['published_rule_pages']=len(public)
            out['rule_text_local']='data/published_rules_response.json'
    path=Path('reports/official/competition_metadata.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)

if __name__=='__main__':main()
