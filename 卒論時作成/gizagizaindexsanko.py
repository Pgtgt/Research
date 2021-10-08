def unwrap(wrap):#-1.5~1.5で折り返される位相を展開かつ位相が０に近い部分を
    phi=np.empty(len(wrap))
    afterchangeindex=[]
    n=0
    phi[0]=wrap[0]
    for i in range(1,len(wrap)):
        if((wrap[i] - wrap[i-1])<-2):
        # if((np.diff(wrap[i-1]))<-2):
            n=n+1
            afterchangeindex.append(i)
        phi[i]=wrap[i]+(n*math.pi)
        
    return phi,afterchangeindex
    
    
    phi,afterchangeindex=unwrap(wrap)

Ph0index_sub=np.empty(len(afterchangeindex)-1)
# Ph0index_sub = [0] * (len(afterchangeindex)-1)
wrap_abs=np.abs(wrap)
len_afterchangeindex=len(afterchangeindex)
for i in range (0,len_afterchangeindex-1):
    start=afterchangeindex[i]
    end=afterchangeindex[i+1]
    Ph0index_sub[i]=np.argmin(wrap_abs[start:end])
len_Ph0index_sub=len(Ph0index_sub)
Ph0index = [int(Ph0index_sub[_])+afterchangeindex[_] for _ in range(len_Ph0index_sub)]
    
    
