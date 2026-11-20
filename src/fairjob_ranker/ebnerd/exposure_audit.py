"""Label-free, article-free prior-exposure features for an audit-only replay.

Availability is an explicit 60-second replay assumption, not measured production
ingestion latency. All requests at a timestamp see only strictly older available
events. These features describe the sampled logging policy, not organic demand.
"""
from collections import Counter,deque
import numpy as np
from .features import log_age

EXPOSURE_FEATURES=['past_exposures','past_exposures_1h','past_exposures_24h',
    'first_exposure_age','last_exposure_age','exposure_unseen','user_past_exposures']

class ExposureState:
    def __init__(self,delay_seconds=60):
        if delay_seconds<0:raise ValueError('Nonnegative availability delay required')
        self.delay=delay_seconds;self.pending=deque();self.hour=deque();self.day=deque()
        self.total=Counter();self.one_hour=Counter();self.one_day=Counter()
        self.first={};self.last={};self.users={};self.now=None

    def advance(self,now):
        now=int(now)
        if self.now is not None and now<self.now:raise ValueError('Chronological replay required')
        self.now=now
        while self.pending and self.pending[0][0]+self.delay<now:
            stamp,user,ids=self.pending.popleft()
            self.total.update(ids);self.users.setdefault(user,Counter()).update(ids)
            for item in ids:self.first.setdefault(item,stamp);self.last[item]=stamp
            if stamp>=now-3600:self.hour.append((stamp,ids));self.one_hour.update(ids)
            if stamp>=now-86400:self.day.append((stamp,ids));self.one_day.update(ids)
        for queue,counts,window in [(self.hour,self.one_hour,3600),(self.day,self.one_day,86400)]:
            while queue and queue[0][0]<now-window:
                _,ids=queue.popleft();counts.subtract(ids)
                for item in ids:
                    if counts[item]==0:del counts[item]

    def matrix(self,now,user,ids):
        self.advance(now);prior=self.users.get(int(user),{})
        return np.array([[np.log1p(self.total[i]),np.log1p(self.one_hour[i]),np.log1p(self.one_day[i]),
            log_age(now,self.first.get(i)),log_age(now,self.last.get(i)),int(i not in self.first),
            np.log1p(prior.get(i,0))] for i in map(int,ids)],dtype=np.float32)

    def observe(self,now,user,ids):
        if self.now!=int(now):raise ValueError('Score timestamp before observing its candidates')
        self.pending.append((int(now),int(user),tuple(map(int,ids))))
