#ifndef FIXED_GRID_H
#define FIXED_GRID_H

#include <vector>
#include "CMesh.h"
#include <fstream>
using namespace std;


class CGrid {
  public:
    std::vector<CVertex *> samples;  
    std::vector<int> index;    // the start index of each grid in the sample points which is order by Zsort
    int xside, yside, zside;
    double radius;   

    typedef std::vector<CVertex *>::iterator iterator;
    
    CGrid() {}
    void init(std::vector<CVertex> &vert, vcg::Box3f &box, double radius);

    // compute the repulsion terms, update vertex.p & vertex.wp
    void iterate(void (*self)(iterator starta, iterator enda, double radius),
                 void (*other)(iterator starta, iterator enda, 
                              iterator startb, iterator endb, double radius));

    // compute the data loyalty terms, update vertex.s & vertex.ws
    void sample(CGrid &points, 
                void (*sample)(iterator starta, iterator enda, 
                               iterator startb, iterator endb, double radius));
                     
    int cell(int x, int y, int z) { return x + xside*(y + yside*z); }
    bool isEmpty(int cell) { return index[cell+1] == index[cell]; }
    iterator startV(int origin) { return samples.begin() + index[origin]; }  
	iterator endV(int origin) { return samples.begin() + index[origin+1]; }

};


#endif
