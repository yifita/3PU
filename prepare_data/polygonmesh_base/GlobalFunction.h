#pragma once
#ifndef GLOBALFUNCTION_H
#define GLOBALFUNCTION_H
#include <vector>
#include "CMesh.h"
#include "grid.h"

//#include "TriMesh.h"
//#include "TriMesh_algo.h"
//#include "vcg/complex/trimesh/create/ball_pivoting.h" //#include "LAP_Others/eigen.h"
//#include "vcg/complex/trimesh/autoalign_4pcs.h"
//#include "Algorithm/pointcloud_normal.h"
#include <fstream>
#include <float.h>
#include <QString>
#include <iostream>
#include <time.h>
#include <string>
#include <ctime>
#include <algorithm>
#include <math.h>
//#include "ANN/ANN.h"
#include <eigenlib/Eigen/Dense>

#include "vcg/complex/algorithms/intersection.h"

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_EXCEPTIONS
//#define LINKED_WITH_TBB

using namespace std;
using namespace vcg;

#define MyMax(a,b) (((a) > (b)) ? (a) : (b))  
#define MyMin(a,b) (((a) < (b)) ? (a) : (b))  

const double PI = 3.1415926;
const double EPS_SUN = 1e-8;    //lion: 1e-8, dc: 1e-10, sphere: 1e-7, anno:1e-7
const double EPS_VISIBILITY = 1e-4;
const double BIG = 100000;

namespace GlobalFun
{
//  struct DesityAndIndex{
//    int index;
//    double density;
//  };

//  bool cmp(DesityAndIndex &a, DesityAndIndex &b);

  double getAbsMax(double x, double y, double z);
  float tinyrand();

	void computeKnnNeigbhors(vector<CVertex> &datapts, vector<CVertex> &querypts, int numKnn, bool need_self_included, QString purpose);
	void computeEigen(CMesh* _samples);
	void computeEigenIgnoreBranchedPoints(CMesh* _samples);
	void computeEigenWithTheta(CMesh* _samples, double radius);
	void computeAnnNeigbhors(vector<CVertex> &datapts, vector<CVertex> &querypts, int numKnn, bool need_self_included, QString purpose);
  void computeBallNeighbors(CMesh* goal_set, CMesh* search_set, double radius, vcg::Box3f& box);
  double estimateKnnSize(CMesh* mesh0, CMesh* mesh1, double radius, vcg::Box3f& box);

	void static  __cdecl self_neighbors(CGrid::iterator start, CGrid::iterator end, double radius);
	void static  __cdecl other_neighbors(CGrid::iterator starta, CGrid::iterator enda, 
	CGrid::iterator startb, CGrid::iterator endb, double radius);
	void static __cdecl find_original_neighbors(CGrid::iterator starta, CGrid::iterator enda, 
	CGrid::iterator startb, CGrid::iterator endb, double radius); 

	double computeEulerDist(const Point3f& p1, const Point3f& p2);
	double computeEulerDistSquare(Point3f& p1, Point3f& p2);
	double computeProjDist(Point3f& p1, Point3f& p2, Point3f& normal_of_p1);
	double computeProjDistSquare(Point3f& p1, Point3f& p2, Point3f& normal_of_p1);
	double computePerpendicularDistSquare(Point3f& p1, Point3f& p2, Point3f& normal_of_p1);
	double computePerpendicularDist(Point3f& p1, Point3f& p2, Point3f& normal_of_p1);
	double computeProjPlusPerpenDist(Point3f& p1, Point3f& p2, Point3f& normal_of_p1);
	double getDoubleMAXIMUM();
	vector<int> GetRandomCards(int Max);

  bool isPointInBoundingBox(Point3f &v0, CMesh *mesh, double delta = 0.0f);
	double computeRealAngleOfTwoVertor(Point3f v0, Point3f v1);
	bool isTwoPoint3fTheSame(Point3f& v0, Point3f& v1);
	bool isTwoPoint3fOpposite(Point3f& v0, Point3f& v1);
  double computeTriangleArea_3(Point3f& v0, Point3f& v1, Point3f& v2);
  bool isPointInTriangle_3(Point3f& v0, Point3f& v1, Point3f& v2, Point3f& p);
  double computeMeshLineIntersectPoint(CMesh *target, Point3f& p, const Point3f& line_dir, Point3f& result, Point3f& result_normal, bool& is_barely_visible);
  Point3f scalar2color(double scalar);
  void normalizeConfidence(vector<CVertex>& vertexes, float delta);

  void ballPivotingReconstruction(CMesh& mesh, double radius = 0.0, double clustering = 20 / 100, double creaseThr = 90.0f);
  void computePCANormal(CMesh *mesh, int knn);

  void removeOutliers(CMesh *mesh, double radius, double remove_percent);
  void removeOutliers(CMesh *mesh, double radius, int remove_num);
  void addOutliers(CMesh *mesh, double outlier_percent, double max_move_dist);
  void addOutliers(CMesh *mesh, int add_num, double max_move_dist);
  void addNoise(CMesh *mesh, float noise_size);
  double sampleNormalDistribution(double _sigma, double _magnitude);
  double gaussian_beam(double _x, double _sigma);
  void addBenchmarkNoise(CMesh *mesh, Point3f &camera_pos, Point3f &view_ray, double mag = 0.0f);
  void addBenchmarkNoiseAfterwards(CMesh *mesh, double mag);
  void mergeMesh(CMesh *src, CMesh *target);
  void downSample(CMesh *dst, CMesh *src, double sample_ratio, bool use_random_downsample = true);
  void clearCMesh(CMesh &mesh);

  void deleteIgnore(CMesh* mesh);
  void recoverIgnore(CMesh* mesh);

  void cutPointSelfSlice(CMesh* mesh, Point3f anchor, Point3f direction, double width);

  void printMatrix33(ostream& out, vcg::Matrix33f mat33);
  void printMatrix44(ostream& out, vcg::Matrix44f mat44);
  void printPoint3(ostream& out, vcg::Point3f p);
  void printQuaternionf(ostream& out, vcg::Quaternionf qua);

  vcg::Matrix33f myQuaternionToMatrix33(Quaternionf qua_in);
  vcg::Matrix33f directionToMatrix33(Point3f direction);
  vcg::Matrix33f axisToMatrix33(CVertex v);

  vcg::Matrix33f getMat33FromMat44(vcg::Matrix44f mat44);
  Point3f getVectorFromMat44(vcg::Matrix44f mat44);
  vcg::Matrix44f getMat44FromMat33AndVector(vcg::Matrix33f mat33, Point3f vec);
  void convertCMeshO2CMesh(CMeshO &src, CMesh &dst);
  void convertCMesh2CMeshO(CMesh &src, CMeshO &dst);
}

class Timer
{
public:

	inline void start(const string& str)
	{
		cout << endl;
		starttime = clock();
		mid_start = clock();
		cout << "@@@@@ Time Count Start For: " << str << endl;

		_str = str;
	}

	inline void insert(const string& str)
	{
		mid_end = clock();
		timeused = mid_end - mid_start;
		cout << "##" << str << "  time used:  " << timeused / double(CLOCKS_PER_SEC) << " seconds." << endl;
		mid_start = clock();
	}

	inline void end()
	{
		stoptime = clock();
		timeused = stoptime - starttime;
		cout <<  "@@@@ finish	" << _str << "  time used:  " << timeused / double(CLOCKS_PER_SEC) << " seconds." << endl;
		cout << endl;
	}

private:
	int starttime, mid_start, mid_end, stoptime, timeused;
	string _str;
};


class Slice
{
public:
  Slice()
  {
    res = 0;
  }

  //void build_slice(Point3f min, Point3f max, float cell_size);
  //void build_slice(Point3f a, Point3f b, Point3f c, float cell_length);
  vector<CVertex> getSliceNodes(){ return slice_nodes; }

public:

  vector<CVertex> slice_nodes;
  int res;
};

typedef vector<Slice> Slices;

/* Useful code template

(1)
for(int i = 0; i < samples->vert.size(); i++)
{
CVertex& v = samples->vert[i];

for (int j = 0; j < v.neighbors.size(); j++)
{
CVertex& t = samples->vert[v.neighbors[j]];
}
}

(2)
int count = 0;
time.start("Test 2");
CMesh::VertexIterator vi;
Point3f p0 = Point3f(0,0,0);
for(vi = original->vert.begin(); vi != original->vert.end(); ++vi)
{
count += GlobalFun::computeEulerDistSquare(p0, vi->P());
}
cout << count << endl;
time.end();


time.start("Test 1");
for(int i = 0; i < original->vert.size(); i++)
{
CVertex& v = original->vert[i];
count += (p0 - v.P()).SquaredNorm();
}
cout << count << endl;
time.end();
*/
#endif
