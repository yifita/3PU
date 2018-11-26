#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>
#include <assert.h>
//#include "tbb/parallel_for.h"

#include "grid.h"
#include "GlobalFunction.h"

using namespace vcg;
using namespace std;
using namespace tri;

void GlobalFun::find_original_neighbors(CGrid::iterator starta, CGrid::iterator enda, 
	CGrid::iterator startb, CGrid::iterator endb, double radius) 
{	
	double radius2 = radius*radius;
	double iradius16 = -4/radius2;
	//const double PI = 3.1415926;

	for(CGrid::iterator dest = starta; dest != enda; dest++) 
	{
		CVertex &v = *(*dest);

		Point3f &p = v.P();
		for(CGrid::iterator origin = startb; origin != endb; origin++)
		{
			CVertex &t = *(*origin);

			Point3f &q = t.P();
			Point3f diff = p-q;

			double dist2 = diff.SquaredNorm();

			if(dist2 < radius2) 
			{                          
				v.original_neighbors.push_back((*origin)->m_index);
			}
		}
	}
}

// get neighbors
void GlobalFun::self_neighbors(CGrid::iterator start, CGrid::iterator end, double radius)
{
	double radius2 = radius*radius;
	for(CGrid::iterator dest = start; dest != end; dest++)
	{
		CVertex &v = *(*dest);
		Point3f &p = v.P();


		for(CGrid::iterator origin = dest+1; origin != end; origin++)
		{
			CVertex &t = *(*origin);
			Point3f &q = t.P();
			Point3f diff = p-q;
			double dist2 = diff.SquaredNorm();
			if(dist2 < radius2) 
			{   
				v.neighbors.push_back((*origin)->m_index);
				t.neighbors.push_back((*dest)->m_index);
			}
		}
	}
}

void GlobalFun::other_neighbors(CGrid::iterator starta, CGrid::iterator enda, 
	CGrid::iterator startb, CGrid::iterator endb, double radius)
{
	double radius2 = radius*radius;
	for(CGrid::iterator dest = starta; dest != enda; dest++)
	{
		CVertex &v = *(*dest);
		Point3f &p = v.P();

		for(CGrid::iterator origin = startb; origin != endb; origin++)
		{
			CVertex &t = *(*origin);
			Point3f &q = t.P();
			Point3f diff = p-q;
			double dist2 = diff.SquaredNorm();
			if(dist2 < radius2) 
			{   
				v.neighbors.push_back((*origin)->m_index);
				t.neighbors.push_back((*dest)->m_index);
			}
		}
	}
}

//mesh0: goal_set; mesh1: search_set
void GlobalFun::computeBallNeighbors(CMesh* mesh0, CMesh* mesh1, double radius, vcg::Box3f& box)
{
  if (radius < 0.0001)
  {
    cout << "too small grid!!" << endl; 
    return;
  }
  //mesh1 should be original

  //cout << "compute_Bll_Neighbors" << endl;
  //cout << "radius: " << radius << endl;

  CGrid samples_grid;
  samples_grid.init(mesh0->vert, box, radius);
  //cout << "finished init" << endl;

  if (mesh1 != NULL)
  {
    for (int i = 0; i < mesh0->vn; i++)
    {
      mesh0->vert[i].original_neighbors.clear();
    }

    CGrid original_grid;
    original_grid.init(mesh1->vert, box, radius); // This can be speed up
    samples_grid.sample(original_grid, find_original_neighbors);
  }
  else
  {
    for (int i = 0; i < mesh0->vn; i++)
    {
      mesh0->vert[i].neighbors.clear();
    }

    samples_grid.iterate(self_neighbors, other_neighbors);
  }

}

double GlobalFun::estimateKnnSize(CMesh* samples, CMesh* original, double radius, vcg::Box3f& box)
{
  computeBallNeighbors(samples, original, radius, box);

  double sum_neighbor = 0;
  for(int i = 0; i < samples->vert.size(); i++)
  {
    CVertex& v = samples->vert[i];

    sum_neighbor += v.original_neighbors.size();
  }

  cout << "estimated original KNN: " << sum_neighbor / samples->vert.size() << endl;
  return sum_neighbor / samples->vert.size();
}

void GlobalFun::computeAnnNeigbhors(vector<CVertex> &datapts, vector<CVertex> &querypts, int knn, bool need_self_included = false, QString purpose = "?_?")
{
//	cout << endl <<"Compute ANN for: " << purpose.toStdString() << endl;
//	int numKnn = knn + 1;

//  vector<CVertex>::iterator vi_temp;
//  for(vi_temp = datapts.begin(); vi_temp != datapts.end(); ++vi_temp)
//      vi_temp->neighbors.clear();

//  //if (querypts.size() <= numKnn+2)
//  //{
//  //  vector<CVertex>::iterator vi;
//  //  for(vi = datapts.begin(); vi != datapts.end(); ++vi)
//  //      vi->neighbors.clear();

//  //  return;
//  //}

//	int					    nPts;			 // actual number of data points
//  ANNpointArray		dataPts;	 // data points
//  ANNpoint			  queryPt;	 // query point
//  ANNidxArray			nnIdx;		 // near neighbor indices
//  ANNdistArray		dists;		 // near neighbor distances
//  ANNkd_tree*			kdTree;		 // search structure
//  int			k				= numKnn;			      // number of nearest neighbors
//  int			dim			= 3;			          // dimension
//  double	eps			= 0;			          // error bound
//	int			maxPts	= numKnn + 3000000;	// maximum number of data points

//	if (datapts.size() >= maxPts)
//	{
//		cout << "Too many data" << endl;
//		return;
//	}
//	queryPt = annAllocPt(dim);					// allocate query point
//	dataPts = annAllocPts(maxPts, dim);	// allocate data points
//	nnIdx   = new ANNidx[k];						// allocate near neigh indices
//	dists   = new ANNdist[k];						// allocate near neighbor dists

//	vector<CVertex>::iterator vi;
//	int index = 0;
//	for(vi = datapts.begin(); vi != datapts.end(); ++vi)
//	{
//		for(int j = 0; j < 3; j++)
//			dataPts[index][j] = double(vi->P()[j]);

//		index++;
//	}
//  nPts = datapts.size();	 // read data points
//	kdTree = new ANNkd_tree( // build search structure
//		dataPts,					     // the data points
//		nPts,						       // number of points
//		dim);						       // dimension of space
//	for (vi = querypts.begin(); vi != querypts.end(); ++vi)
//	{
//		vi->neighbors.clear();
//		for (int j = 0; j < 3; j++)
//			queryPt[j] = vi->P()[j];

//		kdTree->annkSearch( // search
//			queryPt,					// query point
//			k,								// number of near neighbors
//			nnIdx,						// nearest neighbors (returned)
//			dists,						// distance (returned)
//			eps);							// error bound

//		for (int k = 1; k < numKnn; k++)
//			vi->neighbors.push_back(nnIdx[k]);
//	}
//	delete [] nnIdx; // clean things up
//	delete [] dists;
//	delete kdTree;
//	annClose();			 // done with ANN
}

void GlobalFun::computeKnnNeigbhors(vector<CVertex> &datapts, vector<CVertex> &querypts, int numKnn, bool need_self_included = false, QString purpose = "?_?")
{
	if (querypts.size() <= numKnn+1)
	{
		vector<CVertex>::iterator vi;
		for(vi = datapts.begin(); vi != datapts.end(); ++vi)
		{
			vi->neighbors.clear();	
		}
		return;
	}

	bool isComputingOriginalNeighbor = false;
	//if (!datapts.empty() && datapts[0].is_original)
	//{
	//	isComputingOriginalNeighbor = true;
	//}

	int starttime, stoptime, timeused;
	starttime = clock();

	cout << endl;
	cout << "compute KNN Neighbors for: " << purpose.toStdString() << endl;


	ofstream outfile1;
	ofstream outfile2;
	float val;

	outfile1.open("point_cloud.txt", ofstream::binary);
	outfile2.open("query.txt", ofstream::binary);

	val = datapts.size();
	outfile1.write((char *)(&val), sizeof(float));
	val = querypts.size();
	outfile2.write((char *)(&val), sizeof(float));
	val = 3;
	outfile1.write((char *)(&val), sizeof(float));
	val = 4;
	outfile2.write((char *)(&val), sizeof(float));


	vector<CVertex>::iterator vi;
	for(vi = datapts.begin(); vi != datapts.end(); ++vi)
	{
		for(int j = 0; j < 3; j++)
		{
			val = vi->P()[j];
			outfile1.write((char *)(&val), sizeof(float));
		}
	}


	for (vi = querypts.begin(); vi != querypts.end(); ++vi) 
	{
		for (int j = 0; j < 3; j++) 
		{
			val = vi->P()[j];
			outfile2.write((char *)(&val), sizeof(float));
		}
		val = 0;
		outfile2.write((char *)(&val), sizeof(float));
	}

	outfile1.close();
	outfile2.close();

	char mycmd[100];
	sprintf(mycmd, "RG_NearestNeighbors.exe point_cloud.txt query.txt result.txt %d", numKnn+1);
	//sprintf(mycmd, "RG_NearestNeighbors.exe point_cloud.txt query.txt result.txt", numKnn+1);

	//cout << mycmd;

	system(mycmd); 

	//cout << "knn_neighbor file saved\n";

	//clean querypts 
	for (vi = querypts.begin(); vi != querypts.end(); ++vi)
	{
		if (isComputingOriginalNeighbor)
		{
			vi->original_neighbors.clear();
		}
		else
		{
			vi->neighbors.clear();
		}
	}

	ifstream infile;
	float size[2];
	int row,col;
	float *data;

	infile.open ("result.txt", ifstream::binary);
	infile.read((char*)size, 2*sizeof(float));
	row = (int)size[0];
	col = (int)size[1];
	data = new float [row*col];
	infile.read((char*)data,row*col*sizeof(float));
	infile.close();

	for (int idx = 0; idx < row; idx++)
	{

		CVertex &v = querypts[(int)data[idx*col+1]-1];
		if (isComputingOriginalNeighbor)
		{
			v.original_neighbors.push_back((int)data[idx*col]-1);
		}
		else
		{
			v.neighbors.push_back((int)data[idx*col]-1);
		}
	}

	if (!need_self_included)// slow solution...
	{
		for(int i = 0; i < querypts.size(); i++)
		{
			CVertex& v = querypts[i];
			v.neighbors.erase(v.neighbors.begin());
		}
	}


	delete[] data;
	//cout << "compute_knn_neighbor end." << endl << endl;

	stoptime = clock();
	timeused = stoptime - starttime;
	cout << "KNN time used:  " << timeused/double(CLOCKS_PER_SEC) << " seconds." << endl;
	cout << endl;
}		


vector<int> GlobalFun::GetRandomCards(int Max)
{
	vector<int> nCard(Max, 0);
	srand(time(NULL));
	for(int i=0; i < Max; i++)
	{
		nCard[i] = i;
	}
	random_shuffle(nCard.begin(), nCard.begin() + Max);
	return nCard;
}


void GlobalFun::computeEigenIgnoreBranchedPoints(CMesh* _samples)
{
//	vector<vector<int> > neighborMap;

//	typedef vector<CVertex>::iterator VertexIterator;

//	VertexIterator begin = _samples->vert.begin();
//	VertexIterator end = _samples->vert.end();

//	neighborMap.assign(end - begin, vector<int>());

//	int curr_index = 0;
//	for (VertexIterator iter=begin; iter!=end; ++iter, curr_index++)
//	{
//    if(iter->neighbors.size() <= 3)
//    {
//      iter->eigen_confidence = 0.5;
//      continue;
//    }

//		//neighborMap[curr_index].push_back(curr_index);
//		for(int j = 0; j < iter->neighbors.size(); j++)
//		{
//			CVertex& t = _samples->vert[iter->neighbors[j]];
//			if (t.is_ignore)
//			{
//				continue;
//			}
//			neighborMap[curr_index].push_back(iter->neighbors[j]);
//		}
//	}

//	int currIndex = 0;
//	for (VertexIterator iter=begin; iter!=end; iter++, currIndex++)
//	{
//		int neighbor_size = neighborMap[currIndex].size();

//		if (neighbor_size < 3)
//		{
//			iter->eigen_confidence = 0.95;
//			iter->eigen_vector0 = Point3f(0, 0, 0);

//			continue;
//		}

//		Matrix33d covariance_matrix;
//		Point3f diff;
//		covariance_matrix.SetZero();
//		int neighborIndex = -1;

//		for (unsigned int n=0; n<neighbor_size; n++)
//		{
//			neighborIndex = neighborMap[currIndex][n];
//			if(neighborIndex < 0)
//				break;
//			VertexIterator neighborIter = begin + neighborIndex;

//			diff = iter->P() - neighborIter->P();

//			for (int i=0; i<3; i++)
//				for (int j=0; j<3; j++)
//					covariance_matrix[i][j] += diff[i]*diff[j];
//		}

//		Point3f   eigenvalues;
//		Matrix33d	eigenvectors;
//		int required_rotations;
//		vcg::Jacobi< Matrix33d, Point3f >(covariance_matrix, eigenvalues, eigenvectors, required_rotations);
//		vcg::SortEigenvaluesAndEigenvectors< Matrix33d, Point3f >(eigenvalues, eigenvectors);

//		double sum_eigen_value = (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
//		iter->eigen_confidence = eigenvalues[0] / sum_eigen_value;

//		for (int d=0; d<3; d++)
//			iter->eigen_vector0[d] = eigenvectors[d][0];
//		for (int d=0; d<3; d++)
//			iter->eigen_vector1[d] = eigenvectors[d][1];
//		for (int d=0; d<3; d++)
//			iter->N()[d] = eigenvectors[d][2];

//		iter->eigen_vector0.Normalize();
//		iter->eigen_vector1.Normalize();
//		iter->N().Normalize();
//	}
}

void GlobalFun::computeEigen(CMesh* _samples)
{
//	vector<vector<int> > neighborMap;

//	typedef vector<CVertex>::iterator VertexIterator;

//	VertexIterator begin = _samples->vert.begin();
//	VertexIterator end = _samples->vert.end();

//	int curr_index = 0;

//	int currIndex = 0;
//	for (VertexIterator iter=begin; iter!=end; iter++, currIndex++)
//	{
//		Matrix33d covariance_matrix;
//		Point3f diff;
//		covariance_matrix.SetZero();
//		int neighbor_size = iter->neighbors.size();
//		for (unsigned int n=0; n<neighbor_size; n++)
//		{
//			Point3f& tP =_samples->vert[iter->neighbors[n]].P();
//			diff = iter->P() - tP;

//			for (int i=0; i<3; i++)
//				for (int j=0; j<3; j++)
//					covariance_matrix[i][j] += diff[i]*diff[j];
//		}

//		Point3f   eigenvalues;
//		Matrix33d	eigenvectors;
//		int required_rotations;
//		vcg::Jacobi< Matrix33d, Point3f >(covariance_matrix, eigenvalues, eigenvectors, required_rotations);
//		vcg::SortEigenvaluesAndEigenvectors< Matrix33d, Point3f >(eigenvalues, eigenvectors);

//		double sum_eigen_value = (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

//		iter->eigen_confidence = eigenvalues[0] / sum_eigen_value;

//		for (int d=0; d<3; d++)
//			iter->eigen_vector0[d] = eigenvectors[d][0];
//		for (int d=0; d<3; d++)
//			iter->eigen_vector1[d] = eigenvectors[d][1];
//		for (int d=0; d<3; d++)
//			iter->N()[d] = eigenvectors[d][2];

//		iter->eigen_vector0.Normalize();
//		iter->eigen_vector1.Normalize();
//		iter->N().Normalize();
//	}
}


void GlobalFun::computeEigenWithTheta(CMesh* _samples, double radius)
{
//	vector<vector<int> > neighborMap;

//	typedef vector<CVertex>::iterator VertexIterator;

//	VertexIterator begin = _samples->vert.begin();
//	VertexIterator end = _samples->vert.end();

//	neighborMap.assign(end - begin, vector<int>());

//	int curr_index = 0;

//	for (VertexIterator iter=begin; iter!=end; iter++, curr_index++)
//	{
//		if(iter->neighbors.size() <= 3)
//		{
//			iter->eigen_confidence = 0.5;
//			continue;
//		}

//		for(int j = 0; j < iter->neighbors.size(); j++)
//		{
//			neighborMap[curr_index].push_back(iter->neighbors[j]);
//		}
//	}

//	double radius2 = radius*radius;
//	double iradius16 = -1/radius2;

//	int currIndex = 0;
//	for (VertexIterator iter=begin; iter!=end; iter++, currIndex++)
//	{
//    if(iter->neighbors.size() <= 3)
//    {
//      iter->eigen_confidence = 0.5;
//      continue;
//    }

//		Matrix33d covariance_matrix;
//		Point3f diff;
//		covariance_matrix.SetZero();
//		int neighborIndex = -1;
//		int neighbor_size = iter->neighbors.size();
//		for (unsigned int n=0; n<neighbor_size; n++)
//		{
//			neighborIndex = neighborMap[currIndex][n];
//			if(neighborIndex < 0)
//				break;
//			VertexIterator neighborIter = begin + neighborIndex;

//			diff = iter->P() - neighborIter->P();

//			Point3f vm = iter->N();
//			Point3f tm = neighborIter->N();
//			double dist2 = diff.SquaredNorm();
//			double theta = exp(dist2*iradius16);

//			for (int i=0; i<3; i++)
//				for (int j=0; j<3; j++)
//					covariance_matrix[i][j] += diff[i]*diff[j] * theta;
//		}

//		Point3f   eigenvalues;
//		Matrix33d	eigenvectors;
//		int required_rotations;
//		vcg::Jacobi< Matrix33d, Point3f >(covariance_matrix, eigenvalues, eigenvectors, required_rotations);
//		vcg::SortEigenvaluesAndEigenvectors< Matrix33d, Point3f >(eigenvalues, eigenvectors);


//		double sum_eigen_value = (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);

//		iter->eigen_confidence = eigenvalues[0] / sum_eigen_value;

//		for (int d=0; d<3; d++)
//			iter->eigen_vector0[d] = eigenvectors[d][0];
//		for (int d=0; d<3; d++)
//			iter->eigen_vector1[d] = eigenvectors[d][1];
//		for (int d=0; d<3; d++)
//			iter->N()[d] = eigenvectors[d][2];

//		iter->eigen_vector0.Normalize();
//		iter->eigen_vector1.Normalize();
//		iter->N().Normalize();
//	}
}


double GlobalFun::computeEulerDist(const Point3f& p1, const Point3f& p2)
{
	double dist2 = (p1-p2).SquaredNorm();
	if (dist2 < 1e-8 || dist2 > 1e8)
	{
		return 0;
	}
	return sqrt(dist2);
}

double GlobalFun::computeEulerDistSquare(Point3f& p1, Point3f& p2)
{
	return (p1-p2).SquaredNorm();
}

double GlobalFun::computeProjDist(Point3f& p1, Point3f& p2, Point3f& normal_of_p1)
{
	return (p2-p1) * normal_of_p1.Normalize();
}

double GlobalFun::computeProjDistSquare(Point3f& p1, Point3f& p2, Point3f& normal_of_p1)
{
	double proj_dist = computeProjDist(p1, p2, normal_of_p1);
	return proj_dist * proj_dist;
}

double GlobalFun::computePerpendicularDistSquare(Point3f& p1, Point3f& p2, Point3f& normal_of_p1)
{
	//Point3f v_p2_p1 = p1-p2;
	//double proj_dist = computeProjDist(p1, p2, normal_of_p1);
	//Point3f v_proj = /*p1 + */normal_of_p1 * proj_dist;
	//   return (v_p2_p1 + v_proj).SquaredNorm();
	double proj_dist = computeProjDist(p1, p2, normal_of_p1);
	Point3f proj_p = p1 + normal_of_p1 * proj_dist;
	return (proj_p - p2).SquaredNorm();
}

double GlobalFun::computePerpendicularDist(Point3f& p1, Point3f& p2, Point3f& normal_of_p1)
{
	return sqrt(computePerpendicularDistSquare(p1, p2, normal_of_p1));
}

double GlobalFun::computeProjPlusPerpenDist(Point3f& p1, Point3f& p2, Point3f& normal_of_p1)
{
	normal_of_p1.Normalize();
	double proj_dist = GlobalFun::computeProjDist(p1, p2, normal_of_p1);

	if (proj_dist <= 0)
		return -1.;
  
	Point3f proj_p = p1 + normal_of_p1 * proj_dist;
	double perpend_dist = sqrt((proj_p - p2).SquaredNorm());
	double eular_dist = GlobalFun::computeEulerDist(p1, p2);
	return eular_dist + perpend_dist;
	/*return proj_dist  * 0.5 + perpend_dist;*/
}

double GlobalFun::getDoubleMAXIMUM()
{  
	return (numeric_limits<double>::max)();
}


bool GlobalFun::isTwoPoint3fTheSame(Point3f& v0, Point3f& v1)
{
	if (abs(v0[0] - v1[0]) < 1e-7 &&  
		abs(v0[1] - v1[1]) < 1e-7 && 
		abs(v0[2] - v1[2]) < 1e-7)
	{
		return true;
	}

	return false;
}


bool GlobalFun::isTwoPoint3fOpposite(Point3f& v0, Point3f& v1)
{
  if (abs(v0[0] + v1[0]) < 1e-7
    && abs(v0[1] + v1[1]) < 1e-7
    && abs(v0[2] + v1[2]) < 1e-7)
  {
    return true;
  }
  else
    return false;

	/*if (abs(-v0[0] - v1[0]) < 1e-7 &&  
		abs(-v0[1] - v1[1]) < 1e-7 && 
		abs(-v0[2] - v1[2]) < 1e-7)
	{
		return true;
	}

	return false;*/
}


double GlobalFun::computeRealAngleOfTwoVertor(Point3f v0, Point3f v1)
{
	v0.Normalize();
	v1.Normalize();

	if (isTwoPoint3fTheSame(v0, v1))
	{
		return 0;
	}

	if (isTwoPoint3fOpposite(v0, v1))
	{
		return 180;
	}

	double angle_cos = v0 * v1;
	if (angle_cos > 1)
	{
		angle_cos = 0.999999999;
	}
	if (angle_cos < -1)
	{
		angle_cos = -0.99999999;
	}
	if (angle_cos > 0 && angle_cos < 1e-8)
	{
		return 90;
	}

	double angle = acos(angle_cos) * 180. / 3.1415926 ;

	if (angle < 0 || angle > 181)
	{
		cout << "compute angle wrong!!" << endl;
    std::cout<< "angle: " <<angle <<std::endl;
		//system("Pause");
		return -1;
	}

	return angle;
}

double inline GlobalFun::computeTriangleArea_3(Point3f& v0, Point3f& v1, Point3f& v2)
{
  Point3f AB = v1 - v0;  //vector v0v1
  Point3f AC = v2 - v0;  //vector v0v2
  Point3f AP = AB ^ AC;  
  return AP.Norm() / 2.0f;
}

bool GlobalFun::isPointInBoundingBox(Point3f &v0, CMesh *mesh, double delta)
{
  if (NULL == mesh)
  {
    cout << "Empty Mesh When isPointInBoundingBox" << endl;
    return false; 
  }

  Point3f bmin = mesh->bbox.min - Point3f(delta, delta, delta);
  Point3f bmax = mesh->bbox.max + Point3f(delta, delta, delta);

  if ( v0[0] >= bmin[0] && v0[1] >= bmin[1] && v0[2] >= bmin[2]
         && v0[0] <= bmax[0] && v0[1] <= bmax[1] && v0[2] <= bmin[2])
    return true;
  else
    return false;
}

bool inline GlobalFun::isPointInTriangle_3(Point3f& v0, Point3f& v1, Point3f& v2, Point3f& p)
{
  double area1 = GlobalFun::computeTriangleArea_3(v0, v1, p);
  double area2 = GlobalFun::computeTriangleArea_3(v0, v2, p);
  double area3 = GlobalFun::computeTriangleArea_3(v1, v2, p);
  double area  = GlobalFun::computeTriangleArea_3(v0, v1, v2);
  if (fabs(area - (area1 + area2 + area3)) < EPS_SUN) return true;
  else return false;
}


double GlobalFun::computeMeshLineIntersectPoint(CMesh *target, Point3f& pos, const Point3f& line_dir, Point3f& result, Point3f& result_normal, bool& is_barely_visible)
{
  //cout << "compute intersection 1" << endl;

  //compute the intersecting point between the ray and the mesh
  int n_face = target->face.size();
  double min_dist = BIG;

#ifdef LINKED_WITH_TBB
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n_face),
    [&](const tbb::blocked_range<size_t>& r)
  {
    for (size_t f = r.begin(); f < r.end(); ++f)
    {
      Point3f& v0 = target->face[f].V(0)->P();
      Point3f& v1 = target->face[f].V(1)->P();
      Point3f& v2 = target->face[f].V(2)->P();

      Point3f face_norm = target->face[f].cN();
      //if the face can't be seen, then continue
      if(face_norm * line_dir > 0) continue;

      //the line cross the point: pos, and line vector is viewray_iter
      double tmp = face_norm * line_dir;

      if (abs(tmp) < EPS_SUN)
      continue;

      double tmp2 = 1.0f / tmp;
      double t = (v0 - p) * face_norm * tmp2;
      Point3f intersect_point = p + line_dir * t;

      if(GlobalFun::isPointInTriangle_3(v0, v1, v2, intersect_point))
      {
        Point3f d = intersect_point - p;
        double dist_temp = d.SquaredNorm();
        //get the visible point
        if (dist_temp < min_dist)
        {
          min_dist = dist_temp;
          result = intersect_point;
          result_normal = face_norm;

          //for visibility based NBV. classify the scanned points
          //TODO: open for visibility
          /*if (computeRealAngleOfTwoVertor(face_norm, -line_dir) > 60.0f)
            is_barely_visible = true;*/
        }
      }
    }
  });
#else

   //CMesh::FaceIterator fi = target->face.begin();

  //cout << "compute intersection 2" << endl;

  vcg::Line3<float> ray;
  ray.SetOrigin(pos);
  ray.SetDirection(line_dir);

  float bar1, bar2, bar3, dist;

  CMesh::FaceIterator fi;
  for(fi = target->face.begin(); fi!=target->face.end(); ++fi )
  {
      Point3f& p0 = fi->P(0);
      Point3f& p1 = fi->P(1);
      Point3f& p2 = fi->P(2);

      Point3f e1 = p0 - p1;
      Point3f e2 = p1 - p2;
      //Point3f face_norm = (e1 ^ e2).Normalize();
      Point3f face_norm = (e1 ^ e2);

      //if the face can't be seen, then continue
      if(face_norm * line_dir > 0) continue;

      if(IntersectionLineTriangle<float>(ray, p0, p1, p2, dist, bar1, bar2) && dist < min_dist)
      {
          min_dist = dist;

          bar3 = (1-bar1-bar2);
          result = p0*bar3 + p1*bar1 + p2*bar2;
          result_normal = face_norm;
      }

//      Point3f e1 = v0 - v1;
//      Point3f e2 = v1 - v2;
//      //Point3f face_norm = (e1 ^ e2).Normalize();
//      //Point3f face_norm = fi->N();
//      Point3f face_norm = (e1 ^ e2);

//      //if the face can't be seen, then continue
//      if(face_norm * line_dir > 0) continue;


//      //the line cross the point: pos, and line vector is viewray_iter
//      double t = ( (v0.X() - p.X()) * face_norm.X()
//        + (v0.Y() - p.Y()) * face_norm.Y()
//        + (v0.Z() - p.Z()) * face_norm.Z() )
//        / ( face_norm.X() * line_dir.X() + face_norm.Y() * line_dir.Y() + face_norm.Z() * line_dir.Z() ) ;

//      Point3f intersect_point = p + line_dir * t;

//      if(GlobalFun::isPointInTriangle_3(v0, v1, v2, intersect_point))
//      {
//        Point3f d = intersect_point - p;
//        double dist_temp = d.SquaredNorm();
//        //get the visible point
//        if (dist_temp < min_dist)
//        {
//          min_dist = dist_temp;
//          result = intersect_point;
//          result_normal = face_norm;

//          //cout << result.X() << " " << result.Y() << " " << result.Z() << endl;
//        }
//      }else continue;
  }

#endif

  //return 0.0;
  //cout << "compute intersection 3 " << min_dist << endl;
  return sqrt(min_dist);
}



//double GlobalFun::computeMeshLineIntersectPoint(CMesh *target, Point3f& p, const Point3f& line_dir, Point3f& result, Point3f& result_normal, bool& is_barely_visible)
//{
//  //cout << "compute intersection 1" << endl;

//  //compute the intersecting point between the ray and the mesh
//  int n_face = target->face.size();
//  double min_dist = BIG;

//#ifdef LINKED_WITH_TBB
//  tbb::parallel_for(tbb::blocked_range<size_t>(0, n_face),
//    [&](const tbb::blocked_range<size_t>& r)
//  {
//    for (size_t f = r.begin(); f < r.end(); ++f)
//    {
//      Point3f& v0 = target->face[f].V(0)->P();
//      Point3f& v1 = target->face[f].V(1)->P();
//      Point3f& v2 = target->face[f].V(2)->P();

//      Point3f face_norm = target->face[f].cN();
//      //if the face can't be seen, then continue
//      if(face_norm * line_dir > 0) continue;

//      //the line cross the point: pos, and line vector is viewray_iter
//      double tmp = face_norm * line_dir;

//      if (abs(tmp) < EPS_SUN)
//      continue;

//      double tmp2 = 1.0f / tmp;
//      double t = (v0 - p) * face_norm * tmp2;
//      Point3f intersect_point = p + line_dir * t;

//      if(GlobalFun::isPointInTriangle_3(v0, v1, v2, intersect_point))
//      {
//        Point3f d = intersect_point - p;
//        double dist_temp = d.SquaredNorm();
//        //get the visible point
//        if (dist_temp < min_dist)
//        {
//          min_dist = dist_temp;
//          result = intersect_point;
//          result_normal = face_norm;

//          //for visibility based NBV. classify the scanned points
//          //TODO: open for visibility
//          /*if (computeRealAngleOfTwoVertor(face_norm, -line_dir) > 60.0f)
//            is_barely_visible = true;*/
//        }
//      }
//    }
//  });
//#else

//   //CMesh::FaceIterator fi = target->face.begin();

//  //cout << "compute intersection 2" << endl;

//  CMesh::FaceIterator fi;
//  for(fi = target->face.begin(); fi!=target->face.end(); ++fi )
//  {
//      Point3f& v0 = fi->P(0);
//      Point3f& v1 = fi->P(1);
//      Point3f& v2 = fi->P(2);

//      Point3f e1 = v0 - v1;
//      Point3f e2 = v1 - v2;
//      //Point3f face_norm = (e1 ^ e2).Normalize();
//      //Point3f face_norm = fi->N();
//      Point3f face_norm = (e1 ^ e2);

//      //if the face can't be seen, then continue
//      if(face_norm * line_dir > 0) continue;


//      //the line cross the point: pos, and line vector is viewray_iter
//      double t = ( (v0.X() - p.X()) * face_norm.X()
//        + (v0.Y() - p.Y()) * face_norm.Y()
//        + (v0.Z() - p.Z()) * face_norm.Z() )
//        / ( face_norm.X() * line_dir.X() + face_norm.Y() * line_dir.Y() + face_norm.Z() * line_dir.Z() ) ;

//      Point3f intersect_point = p + line_dir * t;

//      if(GlobalFun::isPointInTriangle_3(v0, v1, v2, intersect_point))
//      {
//        Point3f d = intersect_point - p;
//        double dist_temp = d.SquaredNorm();
//        //get the visible point
//        if (dist_temp < min_dist)
//        {
//          min_dist = dist_temp;
//          result = intersect_point;
//          result_normal = face_norm;

//          //cout << result.X() << " " << result.Y() << " " << result.Z() << endl;
//        }
//      }else continue;
//  }

////  for (int f = 0; f < n_face; ++f)
////  {
////    Point3f& v0 = target->face[f].V(0)->P();
////    Point3f& v1 = target->face[f].V(1)->P();
////    Point3f& v2 = target->face[f].V(2)->P();
////    //Point3f v0 = target->face[f].V(0)->P();
////    //Point3f v1 = target->face[f].V(1)->P();
////    //Point3f v2 = target->face[f].V(2)->P();
////    Point3f e1 = v0 - v1;
////    Point3f e2 = v1 - v2;
////    Point3f face_norm = (e1 ^ e2).Normalize();
////    //if the face can't be seen, then continue
////    if(face_norm * line_dir > 0) continue;

////    //just choose one point and calculate the distance,if the triangle is too far from the line, then continue
////    /*double A = line_dir.Y();
////    double B = line_dir.Z() - line_dir.X();
////    double C = -line_dir.Y();
////    double D = line_dir.Y() * p.Z() + line_dir.X() * p.Y() - line_dir.Y() * p.X() - line_dir.Z() * p.Y();
////    double proj_dist = abs(A * v0.X() + B * v0.Y() + C * v0.Z() + D) / sqrt(A * A + B * B + C * C);
////    if (proj_dist < )
////    {
////    }*/

////    //the line cross the point: pos, and line vector is viewray_iter
////    double t = ( (v0.X() - p.X()) * face_norm.X()
////      + (v0.Y() - p.Y()) * face_norm.Y()
////      + (v0.Z() - p.Z()) * face_norm.Z() )
////      / ( face_norm.X() * line_dir.X() + face_norm.Y() * line_dir.Y() + face_norm.Z() * line_dir.Z() ) ;

////    Point3f intersect_point = p + line_dir * t;

////    if(GlobalFun::isPointInTriangle_3(v0, v1, v2, intersect_point))
////    {
////      Point3f d = intersect_point - p;
////      double dist_temp = d.SquaredNorm();
////      //get the visible point
////      if (dist_temp < min_dist)
////      {
////        min_dist = dist_temp;
////        result = intersect_point;
////        result_normal = face_norm;

////        //for visibility based NBV. classify the scanned point
////        if (computeRealAngleOfTwoVertor(face_norm, -line_dir) > 60)
////          is_barely_visible = true;
////      }
////    }else continue;
////  }
//#endif
  
//  //return 0.0;
//  //cout << "compute intersection 3 " << min_dist << endl;
//  return sqrt(min_dist);
//}

//bool GlobalFun::cmp(DesityAndIndex &a, DesityAndIndex &b)
//{
//  if (a.density == b.density)
//    return false;

//  return a.density < b.density;
//}

double GlobalFun::getAbsMax(double x, double y, double z)
{
  return std::max(abs(x), std::max(abs(y), abs(z)));
}

//code from http://www.cs.utah.edu/~bergerm/recon_bench/  registration\trimesh2\include\noise3d.h
float GlobalFun::tinyrand()
{
  static unsigned trand = 0;
  trand = 1664525u * trand + 1013904223u;
  return (float) trand / 4294967296.0f;
}

void GlobalFun::removeOutliers(CMesh *mesh, double radius, double remove_percent)
{
//  if (NULL == mesh)
//  {
//    cout<<"Empty Mesh, When Remove Outliers!"<<endl;
//    return;
//  }
//  if(radius <= 0.0f) radius = 0.1;
//  mesh->face.clear();
//  mesh->fn = 0;

//  double radius2 = radius * radius;
//  double iradius16 = .0f - 4.0f / radius2;
//  computeBallNeighbors(mesh, NULL, radius, mesh->bbox);

//  vector<DesityAndIndex> mesh_density;
//  for (int i = 0; i < mesh->vert.size(); ++i)
//  {
//    CVertex &v = mesh->vert[i];
//    DesityAndIndex dai;
//    dai.index = i;
//    dai.density = 1.0f;

//    vector<int>* neighbors = & v.neighbors;
//    for (int j = 0; j < neighbors->size(); ++j)
//    {
//      CVertex &nei = mesh->vert[(*neighbors)[j]];
//      double dist2 = (v.P() - nei.P()).SquaredNorm();
//      double den = exp(dist2 * iradius16);

//      dai.density += den;
//    }
//    mesh_density.push_back(dai);
//  }

//  //sort the density and remove low ones
//  sort(mesh_density.begin(), mesh_density.end(), cmp);
//  int remove_num = static_cast<int> (mesh_density.size() * remove_percent);
//  //set those who are removed, ignored = false
//  for (int i = 0; i < remove_num; ++i)
//  {
//    mesh->vert[mesh_density[i].index].is_ignore = true;
//  }

//  //wsh truly remove points
//  vector<CVertex> temp_vert;
//  for (int i = 0; i < mesh->vert.size(); i++)
//  {
//    CVertex& v = mesh->vert[i];
//    if (!v.is_ignore)
//    {
//      temp_vert.push_back(v);
//    }
//  }

//  mesh->vert.clear();
//  for (int i = 0; i < temp_vert.size(); i++)
//  {
//    CVertex& v = temp_vert[i];
//    v.m_index = i;
//    mesh->vert.push_back(v);
//  }
//  mesh->vn = mesh->vert.size();
}

void GlobalFun::removeOutliers(CMesh *mesh, double radius, int remove_num)
{
  if (NULL == mesh) 
  { 
    cout<<"Empty Mesh, When RemoveOutliers!"<<endl;
    return;
  }

  double remove_percent = 1.0f * remove_num / mesh->vert.size();

  GlobalFun::removeOutliers(mesh, radius, remove_percent);
}

void GlobalFun::addOutliers(CMesh *mesh, int add_num, double max_move_dist)
{
  assert(mesh != NULL);
  for(int i = 0; i < add_num; ++i){
    int idx = rand() % mesh->vert.size();
    CVertex &v = mesh->vert[idx];

    Point3f move_dir = Point3f(rand() * 1.0f / RAND_MAX, rand() * 1.0f / RAND_MAX, rand() * 1.0f / RAND_MAX)
                    - Point3f(0.5f, 0.5f, 0.5f);
    v.P() += move_dir * max_move_dist * (rand() / RAND_MAX + 0.5);
  }
}

void GlobalFun::addOutliers(CMesh *mesh, double outlier_percent, double max_move_dist)
{
  assert(mesh != NULL);
  int outlier_num =  outlier_percent * mesh->vert.size();
  GlobalFun::addOutliers(mesh, outlier_num, max_move_dist);
}

//trmesh2 noise
void GlobalFun::addNoise(CMesh *mesh, float noise_size)
{
  assert(mesh != NULL);
  if (mesh == NULL)  {
    return ;
  }

  double radius = 0.1;
  Point3f *disp = new Point3f[mesh->vert.size()];

  GlobalFun::computeBallNeighbors(mesh, NULL, radius, mesh->bbox);
  cout<<"end ball neighbor" <<endl;
  for(int i = 0; i < mesh->vert.size(); ++i){
    CVertex &v = mesh->vert[i];
    disp[i] = Point3f(0, 0, 0); //initialize
    for(int j = 0; j < v.neighbors.size(); ++j){
      const CVertex &n = mesh->vert[v.neighbors[j]];
      double dist = GlobalFun::computeEulerDist(v.P(), n.P());
      double scale = noise_size / (noise_size + dist);
      disp[i] += (n.P() - v.P()) * (float)tinyrand() * scale;
    }
    if(v.neighbors.size() != 0){
      disp[i] /= float(v.neighbors.size());
    }
    //normal
    disp[i] += v.N() * (2.0f * (float) tinyrand() - 1.0f) * noise_size;
  }
  //update
  cout<<"before update" <<endl;
  for (int i = 0; i < mesh->vert.size(); ++i){
    mesh->vert[i].P() += disp[i];
  }
  delete disp;
}

double GlobalFun::gaussian_beam(double _x, double _sigma)  {
  return exp((-2.0 * (_x * _x)) / (_sigma * _sigma));
}

double GlobalFun::sampleNormalDistribution(double _sigma, double _magnitude)  {
	
	double rand_val = _sigma*((double)rand() / (double)RAND_MAX);
	double gaussian_val = _magnitude*exp(-(2.0*rand_val*rand_val)/(_sigma*_sigma));
	double neg_val = (double)rand() / (double)RAND_MAX;
	//return neg_val < 0.5 ? -gaussian_val : gaussian_val;
	return abs(gaussian_val);


  /*double rand_val = 0;
	for(int i = 0; i < 12; i++)
		rand_val += ((double)rand() / (double)RAND_MAX);
	rand_val -= 6;

	double normal_x = rand_val*_sigma;
	double gaussian_val = _magnitude*exp(-(2.0*normal_x*normal_x)/(_sigma*_sigma));
	double neg_val = (double)rand() / (double)RAND_MAX;
	return neg_val < 0.5 ? -gaussian_val : gaussian_val;
	//return -gaussian_val;*/
}

void GlobalFun::addBenchmarkNoise(CMesh *mesh, Point3f &camera_pos, Point3f &view_ray, double mag) //in angle
{
  assert(mesh != NULL);
  if (mesh == NULL){
    return;
  }
  ofstream out;
  out.open("radiance.txt");
  for (int i = 0;i < mesh->vert.size(); ++i){
    CVertex &v = mesh->vert[i];
    Point3f laser_dir = v.P() - camera_pos;
    double angle = GlobalFun::computeRealAngleOfTwoVertor(laser_dir, view_ray);
    double light_sigma = GlobalFun::computeEulerDist(v.P(), camera_pos)
                          * tan(angle / 180.0f * PI);
    //TODO: we should get the dist from the point to the central laser beam
    double pulse_dist = GlobalFun::computeEulerDist(v.P(), camera_pos);
    double luminance = GlobalFun::gaussian_beam(pulse_dist, 2.0f * light_sigma);
    luminance = luminance < 0 ? 0 : luminance;

    double noise_scale = (1.0 - luminance);
    double noise_magnitude = 0.02f;//reasonable
    double gaussian_noise = GlobalFun::sampleNormalDistribution(noise_scale, noise_magnitude);

    Point3f laser_ray = camera_pos - v.P();
    laser_ray.Normalize();
    Point3f pt_normal = v.N();
    double cos_falloff = abs(laser_ray * pt_normal);

    double radiance = luminance * cos_falloff + mag * gaussian_noise;
    //radiance = radiance > 1 ? 1 : radiance;
    //radiance = radiance < 0 ? 0 : radiance;
    int quantized_radiance = 255 * radiance;
    radiance = quantized_radiance / 255.0;
    out<<"radiance: " <<radiance <<endl;
    v.P() = v.P() + v.N() * radiance;
  }
  out.close();
}

void GlobalFun::addBenchmarkNoiseAfterwards(CMesh *mesh, double mag)
{
  assert(mesh != NULL);
  if (mesh == NULL){
    return;
  }
  ofstream out;
  out.open("addAfterwards.txt");
  cout.rdbuf(out.rdbuf());
  for (int i = 0;i < mesh->vert.size(); ++i){
    CVertex &v = mesh->vert[i];

    double noise_scale = 1.0; //(1.0 - luminance);
    double noise_magnitude = 0.02f;//reasonable
    double gaussian_noise = GlobalFun::sampleNormalDistribution(noise_scale, noise_magnitude);

    double radiance = mag * gaussian_noise;
    //radiance = radiance > 1 ? 1 : radiance;
    //radiance = radiance < 0 ? 0 : radiance;
    int quantized_radiance = 255 * radiance;
    radiance = quantized_radiance / 255.0;
    cout<<"radiance: " <<radiance <<endl;
    v.P() = v.P() + v.N() * radiance;
  }
  out.close();
}

//no face will be exist after the merge
void GlobalFun::mergeMesh(CMesh *src, CMesh *target)
{
  target->face.clear();
  target->fn = 0;
  src->face.clear();
  src->fn = 0;

  int idx = target->vert.back().m_index + 1;

  for (int i = 0; i < src->vert.size(); i++)
  {
    CVertex t = src->vert[i];
    t.is_original = true;
    t.is_fixed_sample = false;
    t.m_index = idx++;

    target->vert.push_back(t);
    target->bbox.Add(t.P());
  }
  target->vn = target->vert.size();
}

void GlobalFun::downSample(CMesh *dst, CMesh *src, double sample_ratio, bool use_random_downsample)
{
  if (NULL == src || src->vert.size() <= 0)
  {
    cout<<"Down sample Error: Empty src points!" <<endl;
    return;
  }

  int want_sample_num = static_cast<int>(src->vert.size() * sample_ratio);
  dst->vn = want_sample_num;

  vector<int> nCard = GlobalFun::GetRandomCards(src->vert.size());
  for ( int i = 0; i < dst->vn; ++i)
  {
    int index = nCard[i];
    if (!use_random_downsample)
    {
      index = i;
    }

    CVertex &v = src->vert[index];
    dst->vert.push_back(v);
    dst->bbox.Add(v.P());
  }

  CMesh::VertexIterator vi;
  for(vi = dst->vert.begin(); vi != dst->vert.end(); ++vi)
    vi->is_original = false;
}

void GlobalFun::clearCMesh(CMesh &mesh)
{
  mesh.face.clear();
  mesh.fn = 0;
  mesh.vert.clear();
  mesh.vn = 0;
  mesh.bbox = Box3f();
}

void GlobalFun::deleteIgnore(CMesh* mesh)
{
  vector<CVertex> temp_vert;
  for (int i = 0; i < mesh->vert.size(); i++)
  {
    CVertex& v = mesh->vert[i];
    if (!v.is_ignore)
    {
      temp_vert.push_back(v);
    }
  }

  mesh->vert.clear();
  for (int i = 0; i < temp_vert.size(); i++)
  {
    CVertex& v = temp_vert[i];
    v.m_index = i;
    mesh->vert.push_back(v);
    mesh->bbox.Add(v.P());
  }
  mesh->vn = mesh->vert.size();
}

void GlobalFun::recoverIgnore(CMesh* mesh)
{
  for (int i = 0; i < mesh->vert.size(); i++)
  {
    CVertex& v = mesh->vert[i];
    v.is_ignore = false;
  }
}

void GlobalFun::cutPointSelfSlice(CMesh* mesh, Point3f anchor, Point3f direction, double width)
{
  //mesh->vert.clear();
  cout << "cut anchor point: " << anchor[0] << ", " << anchor[1] << ", " << anchor[2] << endl;
  cout << "direction: " << direction[0] << ", " << direction[1] << ", " << direction[2] << endl;

  double width2 = width * width;
  for (int i = 0; i < mesh->vert.size(); i++)
  {
    CVertex& v = mesh->vert[i];
    //double perpend_dist2 = GlobalFun::computePerpendicularDistSquare(anchor, v.P(), direction);
    double proj_dist = GlobalFun::computeProjDist(anchor, v.P(), direction);
    double proj_dist2 = proj_dist * proj_dist;

    //if (i < 100)
    //{
    //   cout << perpend_dist2 << " " << width2 << endl;
    //}

    if (proj_dist2 > width2)
    {
      v.is_ignore = true;
      continue;
    }
    
    //cout << "Not Ignore!!!!" << endl;

    Point3f new_p = v.P() - direction * proj_dist;
    v.P() = new_p;
  }
}

void GlobalFun::printMatrix33(ostream& out, vcg::Matrix33f mat33)
{
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      out << mat33[i][j] << "   ";  
    }
    out << endl;
  }
  out << endl;
}

void GlobalFun::printMatrix44(ostream& out, vcg::Matrix44f mat44)
{
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      out << mat44[i][j] << "   ";
    }
    out << endl;
  }
  out << endl;
}

void GlobalFun::printPoint3(ostream& out, vcg::Point3f p)
{
  out  << p.X() << "  " << p.Y() << "  " << p.Z() <<endl;
}

void GlobalFun::printQuaternionf(ostream& out, vcg::Quaternionf qua)
{
  out  << qua.X() << "  " << qua.Y() << "  " << qua.Z() << "  " << qua.W() <<endl;
}

vcg::Matrix33f GlobalFun::myQuaternionToMatrix33(Quaternionf qua_in)
{
  Matrix33f mat;
  mat[0][0] = (2. * (qua_in.X()*qua_in.X() + qua_in.W()*qua_in.W()) -1.);
  mat[0][1] = (2. * (qua_in.X()*qua_in.Y() - qua_in.Z()*qua_in.W()));
  mat[0][2] = (2. * (qua_in.X()*qua_in.Z() + qua_in.Y()*qua_in.W()));
  mat[1][0] = (2. * (qua_in.X()*qua_in.Y() + qua_in.Z()*qua_in.W()));
  mat[1][1] = (2. * (qua_in.Y()*qua_in.Y() + qua_in.W()*qua_in.W())-1.);
  mat[1][2] = (2. * (qua_in.Y()*qua_in.Z() - qua_in.X()*qua_in.W()));
  mat[2][0] = (2. * (qua_in.X()*qua_in.Z() - qua_in.Y()*qua_in.W()));
  mat[2][1] = (2. * (qua_in.Y()*qua_in.Z() + qua_in.X()*qua_in.W()));
  mat[2][2] = (2. * (qua_in.Z()*qua_in.Z() + qua_in.W()*qua_in.W())-1.);

  return mat;
}

vcg::Matrix33f GlobalFun::directionToMatrix33(Point3f direction)
{
  Matrix33f mat;


  return mat;
}

vcg::Matrix33f GlobalFun::axisToMatrix33(CVertex v)
{
  Matrix33f mat;

  mat[0][0] = v.eigen_vector0[0];
  mat[0][1] = v.eigen_vector0[1];
  mat[0][2] = v.eigen_vector0[2];
  mat[1][0] = v.eigen_vector1[0];
  mat[1][1] = v.eigen_vector1[1];
  mat[1][2] = v.eigen_vector1[2];
  mat[2][0] = v.N()[0];
  mat[2][1] = v.N()[1];
  mat[2][2] = v.N()[2];

  return mat;
}


vcg::Matrix33f GlobalFun::getMat33FromMat44(vcg::Matrix44f mat44)
{
  vcg::Matrix33f mat33;

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      mat33[i][j] = mat44[i][j];
    }
  }
  return mat33;
}

Point3f GlobalFun::getVectorFromMat44(vcg::Matrix44f mat44)
{
  Point3f vec;
  vec.X() = mat44[0][3];
  vec.Y() = mat44[1][3];
  vec.Z() = mat44[2][3];

  return vec;
}

vcg::Matrix44f GlobalFun::getMat44FromMat33AndVector(vcg::Matrix33f mat33, Point3f vec)
{
  vcg::Matrix44f mat44;
  mat44.SetIdentity();
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      mat44[i][j] = mat33[i][j];
    }
  }

  mat44[0][3] = vec.X();
  mat44[1][3] = vec.Y();
  mat44[2][3] = vec.Z();

  return mat44;
}

void GlobalFun::convertCMesh2CMeshO(CMesh &src, CMeshO &dst)
{
  const int src_len = src.vert.size();
  dst.vert.resize(src_len);
  for(int i = 0; i < src_len; ++i)
  {
    CVertex &v = src.vert[i];
    dst.vert[i].P() = v.P();
    dst.vert[i].N() = v.N();
    dst.vert[i].N().Normalize();
  }
  dst.vn = src.vn;
  dst.bbox = src.bbox;

  //CMesh::ConstVertexIterator vi;
  //CMeshO::VertexIterator viO;
  //for(vi = src.vert.begin(), viO = dst.vert.begin(); vi != src.vert.end(); ++vi, ++viO)
  //{
  //  //viO->P() = vi->P();
  //  //viO->N() = vi->cN();
  //  //viO->N().Normalize();

  //  viO->P()[0] = vi->P()[0];
  //  viO->P()[1] = vi->P()[1];
  //  viO->P()[2] = vi->P()[2];
  //  viO->N()[0] = vi->cN()[0];
  //  viO->N()[1] = vi->cN()[1];
  //  viO->N()[2] = vi->cN()[2];
  //  viO->N().Normalize();
  //}
  //dst.vn = src.vn;
  //dst.bbox = src.bbox;
}

void GlobalFun::convertCMeshO2CMesh(CMeshO &src, CMesh &dst)
{
  const int src_len = src.vert.size();
  dst.vert.resize(src_len);
  for(int i = 0; i < src_len; ++i){
    CVertexO &v = src.vert[i];
    dst.vert[i].P() = v.P();
    dst.vert[i].N() = v.N();
    dst.vert[i].N().Normalize();
  }
  dst.vn = src.vn;
  dst.bbox = src.bbox;

  /*CMesh::VertexIterator vi;
  CMeshO::VertexIterator viO;
  for(viO = src.vert.begin(), vi = dst.vert.begin(); viO != src.vert.end(); ++viO, ++vi){
  vi->P() = viO->P();
  vi->N() = viO->cN();
  vi->N().Normalize();
  }
  dst.vn = src.vn;
  dst.bbox = src.bbox;*/
}

//void Slice::build_slice(Point3f a, Point3f b, Point3f c, float c_length)
//{
//  cell_length = c_length;
//
//  Point3f origin = a;
//  Point3f row_axis = b-a;
//  Point3f col_axis = c-a;
//
//  float row_length = sqrt(row_axis.SquaredNorm());
//  float col_length = sqrt(col_axis.SquaredNorm());
//
//  int row_num = int(row_length / c_length) + 2;
//  int col_num = int(col_length / c_length) + 2;
//
//  row_axis.Normalize();
//  col_axis.Normalize();
//
//  slice_nodes.clear();
//
//  for (int i = 0; i < row_num; i++)
//  {
//    for (int j = 0; j < col_num; j++)
//    {
//      CVertex new_v;
//      new_v.P() = origin + row_axis * (c_length * i) + col_axis * (c_length * j);
//      slice_nodes.push_back(new_v);
//    }
//  }
//}

Point3f GlobalFun::scalar2color( double scalar ) 
{
  double dis = scalar ;
  dis = (std::min)( (std::max)(dis, 0.0 ), 1.0 ) ;
  dis = dis ;
  Point3f baseColor[9];
  baseColor[0] = Point3f(1.0, 0.0, 0.0) ;
  baseColor[1] = Point3f(1.0, 0.7, 0.0) ;
  baseColor[2] = Point3f(1.0, 1.0, 0.0) ;
  baseColor[3] = Point3f(0.7, 1.0, 0.0) ;
  baseColor[4] = Point3f(0.0, 1.0, 0.0) ;
  baseColor[5] = Point3f(0.0, 1.0, 0.7) ;
  baseColor[6] = Point3f(0.0, 1.0, 1.0) ;
  baseColor[7] = Point3f(0.0, 0.7, 1.0) ;
  baseColor[8] = Point3f(0.0, 0.0, 1.0) ;


  double step = 1.0 / 8.0 ;

  int baseID = dis/step;
  if( baseID == 8 )
    baseID = 7 ;

  Point3f mixedColor =  baseColor[baseID] * (baseID*step+step- dis)  + baseColor[baseID+1] * (dis-baseID*step) ;
  mixedColor = (mixedColor/step);
  return mixedColor ;
}

void GlobalFun::normalizeConfidence(vector<CVertex>& vertexes, float delta)
{
  float min_confidence = GlobalFun::getDoubleMAXIMUM();
  float max_confidence = 0;
  for (int i = 0; i < vertexes.size(); i++)
  {
    CVertex& v = vertexes[i];
    min_confidence = (std::min)(min_confidence, v.eigen_confidence);
    max_confidence = (std::max)(max_confidence, v.eigen_confidence);
  }
  float space = max_confidence - min_confidence;

  for (int i = 0; i < vertexes.size(); i++)
  {
    CVertex& v = vertexes[i];
    v.eigen_confidence = (v.eigen_confidence - min_confidence) / space;
    v.eigen_confidence += delta;
  }
}


void GlobalFun::ballPivotingReconstruction(CMesh &mesh, double radius, double clustering, double creaseThr)
{
//  tri::BallPivoting<CMesh> pivot(mesh,radius, clustering, creaseThr);
//  pivot.BuildMesh();
}

void GlobalFun::computePCANormal(CMesh *mesh, int knn)
{
//  if (mesh->vert.empty()) {std::cout <<"compute PCA empty input! "<<std::endl; return;}

//  //int knn = global_paraMgr.norSmooth.getInt("PCA KNN");
//  CMesh* samples = mesh;
 
//  vector<Point3f> before_normal;
//  for (int i = 0; i < samples->vert.size(); ++i)
//    before_normal.push_back(samples->vert[i].N());

//  //vcg::NormalExtrapolation<vector<CVertex> >::ExtrapolateNormals(samples->vert.begin(), samples->vert.end(), knn, -1);
//  vcg::tri::PointCloudNormal<CMesh>::Param pca_para;
//  pca_para.fittingAdjNum = knn;
//  vcg::tri::PointCloudNormal<CMesh>::Compute(*samples, pca_para, NULL);

//  for (int i = 0; i < samples->vert.size(); ++i)
//  {
//    if (before_normal[i] * samples->vert[i].N() < 0.0f)
//      samples->vert[i].N() *= -1;
//  }
}
