#pragma once
//#include <vcg/simplex/vertex/base.h>
//#include <vcg/simplex/vertex/component_ocf.h>
//#include <vcg/simplex/edge/base.h>
//#include <vcg/simplex/face/base.h>
//#include <vcg/simplex/face/component_ocf.h>

#include <vcg/complex/complex.h>

#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export_ply.h>

#include <cstdlib> //for rand()
#include <ctime> //for time()

//#include <vcg/complex/used_types.h>
//#include <vcg/complex/trimesh/base.h>
//#include <vcg/complex/trimesh/allocate.h>

#include <vcg/simplex/face/topology.h>

//#include <vcg/complex/trimesh/update/bounding.h>
//#include <vcg/complex/trimesh/update/color.h>
//#include <vcg/complex/trimesh/update/flag.h>
//#include <vcg/complex/trimesh/update/normal.h>
//#include <vcg/complex/trimesh/update/position.h>
//#include <vcg/complex/trimesh/update/quality.h>
//#include <vcg/complex/trimesh/update/selection.h>
//#include <vcg/complex/trimesh/update/topology.h>

//#include <vcg\space\point3.h>



#include<vcg/simplex/face/distance.h>
#include<vcg/simplex/face/component_ep.h>
#include <vcg/complex/algorithms/create/platonic.h>
#include <vcg/complex/algorithms/update/normal.h>
#include <vcg/complex/algorithms/update/component_ep.h>
#include <vcg/complex/algorithms/update/flag.h>
#include <vcg/space/intersection3.h>
#include <vcg/space/index/aabb_binary_tree/aabb_binary_tree.h>


#include <cstdlib> //for rand()
#include <ctime> //for time()

#include <vector>
using std::vector;
using namespace vcg;

//commonly used VCG type
// Forward declarations needed for creating the used types
class CVertexO;
class CEdgeO;
class CFaceO;

// Declaration of the semantic of the used types
class CUsedTypesO: public vcg::UsedTypes < vcg::Use<CVertexO>::AsVertexType,
  vcg::Use<CEdgeO   >::AsEdgeType,
  vcg::Use<CFaceO  >::AsFaceType >{};


// The Main Vertex Class
// Most of the attributes are optional and must be enabled before use.
// Each vertex needs 40 byte, on 32bit arch. and 44 byte on 64bit arch.

class CVertexO  : public vcg::Vertex< CUsedTypesO,
  vcg::vertex::InfoOcf,           /*  4b */
  vcg::vertex::Coord3f,           /* 12b */
  vcg::vertex::BitFlags,          /*  4b */
  vcg::vertex::Normal3f,          /* 12b */
  vcg::vertex::Qualityf,          /*  4b */
  vcg::vertex::Color4b,           /*  4b */
  vcg::vertex::VFAdjOcf,          /*  0b */
  vcg::vertex::MarkOcf,           /*  0b */
  vcg::vertex::TexCoordfOcf,      /*  0b */
  vcg::vertex::CurvaturefOcf,     /*  0b */
  vcg::vertex::CurvatureDirfOcf,  /*  0b */
  vcg::vertex::RadiusfOcf         /*  0b */
>{};

// The Main Edge Class
// Currently it does not contains anything.
class CEdgeO : public vcg::Edge<CUsedTypesO, vcg::edge::EVAdj> {
public:
  inline CEdgeO(){};
  inline CEdgeO( CVertexO * v0, CVertexO * v1){ V(0)= v0 ; V(1)= v1;};
  static inline CEdgeO OrderedEdge(CVertexO* v0,CVertexO* v1){
    if(v0<v1) return CEdgeO(v0,v1);
    else return CEdgeO(v1,v0);
  }
};

// Each face needs 32 byte, on 32bit arch. and 48 byte on 64bit arch.
class CFaceO    : public vcg::Face<  CUsedTypesO,
  vcg::face::InfoOcf,              /* 4b */
  vcg::face::VertexRef,            /*12b */
  vcg::face::BitFlags,             /* 4b */
  vcg::face::Normal3f,             /*12b */
  vcg::face::QualityfOcf,          /* 0b */
  vcg::face::MarkOcf,              /* 0b */
  vcg::face::Color4bOcf,           /* 0b */
  vcg::face::FFAdjOcf,             /* 0b */
  vcg::face::VFAdjOcf,             /* 0b */
  vcg::face::WedgeTexCoordfOcf     /* 0b */

> {};

class CMeshO : public vcg::tri::TriMesh< vcg::vertex::vector_ocf<CVertexO>, vcg::face::vector_ocf<CFaceO> > {
public :
  int sfn; //The number of selected faces.
  int svn; //The number of selected faces.
  vcg::Matrix44f Tr; // Usually it is the identity. It is applied in rendering and filters can or cannot use it. (most of the filter will ignore this)

  double grid_radius;//by wsh 11-2-27

  const vcg::Box3f &trBB()
  {
    static vcg::Box3f bb;
    bb.SetNull();
    bb.Add(Tr,bbox);
    return bb;
  }
};
//commonly used VCG type end


class CVertex;
class CFace;

class CEdge;
class CUsedTypes: public vcg::UsedTypes< vcg::Use<CVertex>::AsVertexType, vcg::Use<CEdge>::AsEdgeType, vcg::Use<CFace>::AsFaceType>{};


class CVertex : public vcg::Vertex<CUsedTypes,
  vcg::vertex::InfoOcf,           /*  4b */
  vcg::vertex::Coord3f,           /* 12b */
  vcg::vertex::BitFlags,          /*  4b */
  vcg::vertex::Normal3f,          /* 12b */
  vcg::vertex::Qualityf,          /*  4b */
  vcg::vertex::Color4b,           /*  4b */
  vcg::vertex::VFAdjOcf,          /*  0b */
  vcg::vertex::MarkOcf,           /*  0b */
  vcg::vertex::TexCoordfOcf,      /*  0b */
  vcg::vertex::CurvaturefOcf,     /*  0b */
  vcg::vertex::CurvatureDirfOcf,  /*  0b */
  vcg::vertex::RadiusfOcf         /*  0b */
  /*vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::Color4b, vcg::vertex::BitFlags*/>
{
public:
    vector<int> neighbors;
    vector<int> original_neighbors;
  bool is_ray_hit;
  bool is_ray_stop;
  bool is_view_grid;// should change name to is_view_grid
  bool is_field_grid;
  bool is_model;
  bool is_scanned;
  bool is_scanned_visible;
    bool is_original;
  bool is_iso;
  bool is_hole;
  bool is_poisson;
  bool is_visible;
  bool is_barely_visible;
  bool is_boundary;
    int m_index;

    bool is_fixed_sample; //feature points (blue color)
    bool is_ignore;

    /* for skeletonization */
    float eigen_confidence; //record ISO value for Poisson
    Point3f eigen_vector0; //Associate with the biggest eigen value
    Point3f eigen_vector1; // Also use for remember last better virtual point
    //Point3f eigen_vector2; //The smallest eigen value : should be PCA normal N()

  union
  {
    float skel_radius; // remember radius for branches
    float weight_sum; //
    int remember_iso_index;
    float ground_angle;
    int neighbor_num;
  };

public:
    operator Point3f &()
    {
        return P();
    }

    operator const Point3f &() const
    {
        return cP();
    }

    float & operator[](unsigned int i)
    {
        return P()[i];
    }

    CVertex():
        m_index(0),
    is_visible(false),
    is_barely_visible(false),
    is_view_grid(false),
    is_ray_stop(false),
    is_ray_hit(false),
    is_model(false),
    is_scanned(false),
    is_scanned_visible(false),
        is_original(false),
    is_iso(false),
    is_hole(false),
    is_poisson(false),
        is_fixed_sample(false),
    is_boundary(false),
        eigen_confidence(-1),
        is_ignore(false),
    is_field_grid(false),
        eigen_vector0(Point3f(1, 0, 0)),
        eigen_vector1(Point3f(0, 1, 0))
        {
            N() = Point3f(0,0,0);
      //C().SetRGB(0, 0, 255);
        }

    /* for skeletonization */
    void remove() //important, some time we don't want to earase points, just remove them
    {
        neighbors.clear();
        original_neighbors.clear();
        is_ignore = true;
        P() = Point3f(88888888888.8, 88888888888.8, 88888888888.8);
    }

    bool isSample_Moving()
    {
        return (!is_ignore && !is_fixed_sample);
    }

    bool isSample_JustMoving()
    {
        return (!is_ignore && !is_fixed_sample);
    }

    bool isSample_MovingAndVirtual()
    {
        return (!is_ignore && !is_fixed_sample);
    }

    bool isSample_JustFixed()
    {
        return (!is_ignore && is_fixed_sample);
    }

    bool isSample_FixedAndBranched()
    {
        return (!is_ignore && is_fixed_sample);
    }

    void setSample_JustMoving()
    {
        is_fixed_sample = false;
    }

    void setSample_MovingAndVirtual()
    {
        is_fixed_sample = false;
    }

    void setSample_JustFixed()
    {
        is_fixed_sample = true;
    }

    void setSample_FixedAndBranched()
    {
        is_fixed_sample = true;
    }

    void recompute_m_render()
    {
        srand(time(NULL));
        int x = rand()%1000;
        int y = rand()%1000;
        int z = rand()%1000;

        Point3f normal = N();
        normal.Normalize();

        Point3f helper(x/1000.0, y/1000.0, z/1000.0);
        Point3f new_m3_to_m5 = normal ^ helper;
        new_m3_to_m5.Normalize();
        Point3f new_m6_to_m8 = normal ^ new_m3_to_m5;

        eigen_vector0 = new_m3_to_m5;
        eigen_vector1 = new_m6_to_m8;

        //eigen_vector0 = N() ^ Point3d(0.12345, 0.12346, 0.12347);
        //eigen_vector0.Normalize();
        //eigen_vector1 = eigen_vector0 ^ N();
        //eigen_vector1.Normalize();
    }
};

class CEdge : public vcg::Edge<CUsedTypes, vcg::edge::EVAdj>
{
public:
  inline CEdge(){};
  inline CEdge( CVertex * v0, CVertex * v1){ V(0)= v0 ; V(1)= v1;};
  static inline CEdge OrderedEdge(CVertex* v0,CVertex* v1){
    if(v0<v1) return CEdge(v0,v1);
    else return CEdge(v1,v0);
  }
};

class CFace : public vcg::Face<CUsedTypes,
  vcg::face::InfoOcf,              /* 4b */
  vcg::face::VertexRef,            /*12b */
  vcg::face::BitFlags,             /* 4b */
  vcg::face::Normal3f,             /*12b */
  vcg::face::QualityfOcf,          /* 0b */
  vcg::face::MarkOcf,              /* 0b */
  vcg::face::Color4bOcf,           /* 0b */
  vcg::face::FFAdjOcf,             /* 0b */
  vcg::face::VFAdjOcf,             /* 0b */
  vcg::face::WedgeTexCoordfOcf,     /* 0b */
  vcg::face::EdgePlane //new 2018
  /*vcg::face::FFAdj, vcg::face::VFAdj, vcg::face::VertexRef, vcg::face::BitFlags*/>
{};

class CMesh : public vcg::tri::TriMesh< vcg::vertex::vector_ocf<CVertex>, vcg::face::vector_ocf<CFace>/*, std::vector<CEdge>*/ >
{
public:
    vcg::Matrix44f Tr;
};


typedef vcg::AABBBinaryTreeIndex<CFace, float, vcg::EmptyClass> CIndex;
