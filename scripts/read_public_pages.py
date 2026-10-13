"""Snapshot published participant-facing competition pages; no host APIs."""
import json
from pathlib import Path
from datetime import datetime, timezone
from download_data import authenticated_session, COMPETITION

if __name__=='__main__':
    session=authenticated_session()
    response=session.get(f'https://www.kaggle.com/api/v1/competitions/{COMPETITION}/pages',timeout=(30,60))
    response.raise_for_status()
    value=response.json()
    pages=value.get('pages',[]) if isinstance(value,dict) else value
    Path('data/published_competition_pages.json').write_text(json.dumps(pages,indent=2),encoding='utf-8')
    record={'checked_utc':datetime.now(timezone.utc).isoformat(),'source':'official public participant pages endpoint',
        'pages':[{'name':p.get('name',p.get('pageName')),'characters':len(p.get('content',''))} for p in pages]}
    Path('reports/research/published_pages_audit.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(record)
