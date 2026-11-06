// CPU-only exact voxel DDA for hypothetical support-threshold ray diagnostics.
// Never queries field density or changes any occupancy artifact.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>
extern "C" int support_ray_maxima(const double* origins, const double* directions,
    int64_t nrays, const double* bounds, int resolution, const uint8_t* gate0,
    const uint8_t* gate1, const uint8_t* counts, int channels, int16_t* maxima,
    uint8_t* floor_hit) {
  const int64_t volume=int64_t(resolution)*resolution*resolution;
  int failures=0;
  #pragma omp parallel for num_threads(2) reduction(+:failures)
  for(int64_t ray=0;ray<nrays;++ray) {
    const double* o=origins+3*ray; const double* d=directions+3*ray;
    for(int group=0;group<2;++group)for(int c=0;c<channels;++c)
      maxima[(group*nrays+ray)*channels+c]=-1;
    double entry=0.,leave=std::numeric_limits<double>::infinity(); bool hit=true;
    for(int axis=0;axis<3;++axis) {
      if(d[axis]==0.) { if(o[axis]<bounds[axis]||o[axis]>=bounds[axis+3])hit=false; }
      else {
        double a=(bounds[axis]-o[axis])/d[axis],b=(bounds[axis+3]-o[axis])/d[axis];
        entry=std::max(entry,std::min(a,b));leave=std::min(leave,std::max(a,b));
      }
    }
    hit=hit&&leave>entry;floor_hit[ray]=hit;
    if(!hit)continue;
    int index[3],step[3];double next[3],pitch[3];
    for(int axis=0;axis<3;++axis) {
      pitch[axis]=(bounds[axis+3]-bounds[axis])/resolution;
      index[axis]=std::min(resolution-1,std::max(0,int(std::floor((o[axis]+entry*d[axis]-bounds[axis])/pitch[axis]))));
      step[axis]=d[axis]>0?1:(d[axis]<0?-1:0);
      next[axis]=step[axis] ? (bounds[axis]+(index[axis]+(step[axis]>0))*pitch[axis]-o[axis])/d[axis] : std::numeric_limits<double>::infinity();
    }
    double current=entry;int iterations=0;
    while(current<leave&&index[0]>=0&&index[0]<resolution&&index[1]>=0&&index[1]<resolution&&index[2]>=0&&index[2]<resolution) {
      if(++iterations>3*resolution+6){++failures;break;}
      double end=std::min(leave,std::min(next[0],std::min(next[1],next[2])));
      int64_t cell=(int64_t(index[0])*resolution+index[1])*resolution+index[2];
      if(end>current)for(int group=0;group<2;++group) {
        if(!(group?gate1[cell]:gate0[cell]))continue;
        for(int c=0;c<channels;++c)maxima[(group*nrays+ray)*channels+c]=std::max<int16_t>(maxima[(group*nrays+ray)*channels+c],counts[(group*channels+c)*volume+cell]);
      }
      if(end>=leave)break;
      // Update all axes at exact tied crossings; zero-length cells add no evidence.
      double crossing=std::min(next[0],std::min(next[1],next[2]));
      for(int axis=0;axis<3;++axis)if(next[axis]==crossing) {
        index[axis]+=step[axis];
        next[axis]=(bounds[axis]+(index[axis]+(step[axis]>0))*pitch[axis]-o[axis])/d[axis];
      }
      current=std::max(current,end);
    }
  }
  return failures;
}
