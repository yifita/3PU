#include "grid.h"

#include <algorithm>
#include <iostream>
using namespace std;
using namespace vcg;

class XSort {
  public:
  bool operator()(const CVertex *a, const CVertex *b) {
    return a->P()[0] < b->P()[0];
  }
};

class YSort {
  public:
  bool operator()(const CVertex *a, const CVertex *b) {
    return a->P()[1] < b->P()[1];
  }
};

class ZSort {
  public:
  bool operator()(const CVertex *a, const CVertex *b) {
    return a->P()[2] < b->P()[2];
  }
};
// divid sample into some grids
// and each grid has their points index in the index vector of sample.
void CGrid::init(std::vector<CVertex> &vert, Box3f &box, double _radius) {
     
//     cout << "enter grid::init"<<endl;
     
  radius = _radius;

  samples.resize(vert.size());
  for(int i = 0; i < samples.size(); i++) {
    samples[i] = &vert[i];
	//// new
	//samples[i]->m_index = i;
  }


  Point3f min = box.min;
  Point3f max = box.max; 

  xside = (int)ceil((max[0] - min[0])/radius);
  yside = (int)ceil((max[1] - min[1])/radius);
  zside = (int)ceil((max[2] - min[2])/radius);
  
//  cout << "radius = "<<radius<<endl;
//  cout << "xside, yside, zside "<<xside<<" "<<yside<<" "<<zside<<endl;
  xside = (xside > 0) ? xside : 1;
  yside = (yside > 0) ? yside : 1;
  zside = (zside > 0) ? zside : 1;

  assert(xside > 0 && yside > 0 && zside > 0);
  
  index.resize(xside*yside*zside+1, -1);  //x + xside*x + xside*yside*z

  sort(samples.begin(), samples.end(), ZSort()); //this would be very slow 

  int startz = 0;
  for(int z = 0; z < zside; z++) {
    int endz = startz;
    double maxz = min[2] + (z+1)*radius;
    while(endz < samples.size() && samples[endz]->P()[2] < maxz)
      ++endz; 
    
    sort(samples.begin()+startz, samples.begin()+endz, YSort());
    
    int starty = startz;
    for(int y = 0; y < yside; y++) {
      int endy = starty;        
      double maxy = min[1] + (y+1)*radius;
      while(endy < endz && samples[endy]->P()[1] < maxy)
        ++endy;
         
      sort(samples.begin()+starty, samples.begin()+endy, XSort());
      
      int startx = starty;
      for(int x = 0; x < xside; x++) {
        int endx = startx;
        index[x + xside*y + xside*yside*z] = endx;          
        double maxx = min[0] + (x+1)*radius;
        while(endx < endy && samples[endx]->P()[0] < maxx)
          ++endx;
        
        startx = endx;
      }
      starty = endy;
      
    }
    startz = endz;
  }
  index[xside*yside*zside] = startz;  // in order to compute the last grid's range

// for debug..
//   for (int i = 0; i < samples.size(); i++)
//   {
// 	  samples[i]->m_index = i;
//   }
}

void CGrid::iterate(void (*self)(iterator starta, iterator enda, double radius),
                 void (*other)(iterator starta, iterator enda, 
                              iterator startb, iterator endb, double radius)) {

  static int corner[8*3] = { 0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1,
                             0, 1, 1,  1, 0, 1,  1, 1, 0,  1, 1, 1 };
  
  static int diagonals[14*2] = { 0, 0, //remove this line to avoid self intesextion
                                 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7,
                                 2, 3, 1, 3, 1, 2,                       
                                 1, 4, 2, 5, 3, 6 };

  for(int z = 0; z < zside; z++) {
    for(int y = 0; y < yside; y++) {
      for(int x = 0; x < xside; x++) {
        int origin = cell(x, y, z);
        self(startV(origin), endV(origin), radius);  // 
        // compute between other girds
        for(int d = 2; d < 28; d += 2) { // skipping self
          int *cs = corner + 3*diagonals[d];
          int *ce = corner + 3*diagonals[d+1];
          if((x + cs[0] < xside) && (y + cs[1] < yside) && (z + cs[2] < zside) &&
             (x + ce[0] < xside) && (y + ce[1] < yside) && (z + ce[2] < zside)) {
			 
             origin = cell(x+cs[0], y+cs[1], z+cs[2]);
             int dest = cell(x+ce[0], y+ce[1], z+ce[2]);
             other(startV(origin), endV(origin), 
                   startV(dest),   endV(dest), radius);        
          }
        } // for( int d...)      
      }
    }
  }
}


void CGrid::sample(CGrid &points, 
                void (*sample)(iterator starta, iterator enda, 
                               iterator startb, iterator endb, double radius)) {

   static int corner[8*3] = { 0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1,
                              0, 1, 1,  1, 0, 1,  1, 1, 0,  1, 1, 1 };

   static int diagonals[14*2] = { 0, 0, //remove this line to avoid self intesextion
                                  0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7,
                                  2, 3, 1, 3, 1, 2,                       
                                  1, 4, 2, 5, 3, 6 };

  for(int z = 0; z < zside; z++) {
    for(int y = 0; y < yside; y++) {
      for(int x = 0; x < xside; x++) {     
        int origin = cell(x, y, z);  
         
        if(!isEmpty(origin) && !points.isEmpty(origin)) 
          sample(startV(origin), endV(origin), 
                 points.startV(origin),   points.endV(origin), radius);  
                        
        for(int d = 2; d < 28; d += 2) { //skipping self
          int *cs = corner + 3*diagonals[d];
          int *ce = corner + 3*diagonals[d+1];
          if((x+cs[0] < xside) && (y+cs[1] < yside) && (z+cs[2] < zside) &&
             (x+ce[0] < xside) && (y+ce[1] < yside) && (z+ce[2] < zside)) {

             origin   = cell(x+cs[0], y+cs[1], z+cs[2]);

             int dest = cell(x+ce[0], y+ce[1], z+ce[2]);

             if(!isEmpty(origin) && !points.isEmpty(dest))           // Locally 
               sample(startV(origin), endV(origin), 
                      points.startV(dest),   points.endV(dest), radius); 

             if(!isEmpty(dest) && !points.isEmpty(origin))  
               sample(startV(dest), endV(dest), 
                      points.startV(origin),   points.endV(origin), radius);        
          }
        }      
      }
    }
  }
}