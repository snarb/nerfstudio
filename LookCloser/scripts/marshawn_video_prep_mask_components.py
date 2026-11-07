"""Conservative native-mask island qualification; discarded pixels stay UNKNOWN."""
import cv2,numpy as np
from scipy.ndimage import distance_transform_edt

def qualify(mask,reference=None,margin=128):
    mask=np.asarray(mask,dtype=bool)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    if count<=2:return mask.copy(),np.zeros_like(mask),dict(components=count-1,discarded_pixels=0,margin_native=margin)
    sizes=stats[1:,cv2.CC_STAT_AREA];largest=1+int(sizes.argmax());primary=labels==largest;near=distance_transform_edt(~primary)<=margin
    # A sizeable disconnected body part is protected even if no previous-frame
    # label reaches it. Tiny distant islands have no reliable person identity.
    keep=np.zeros(count,bool);keep[largest]=True;ref=None
    if reference is not None:ref=cv2.resize(reference.astype('uint8'),(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    rows=[]
    for k in range(1,count):
        component=labels==k;size=int(sizes[k-1]);close=bool(near[component].any());prior=int((component&ref).sum()) if ref is not None else 0
        retain=bool(k==largest or close or size>=max(4096,int(sizes.max()*.02)) or prior>=max(16,int(size*.05)))
        keep[k]=retain;rows.append(dict(component=k,pixels=size,near_primary=close,prior_overlap=prior,retained=retain))
    accepted=keep[labels];discarded=mask&~accepted
    return accepted,discarded,dict(components=count-1,discarded_pixels=int(discarded.sum()),margin_native=margin,rows=rows,discarded_policy='UNKNOWN, never certifiedclearbackground')
