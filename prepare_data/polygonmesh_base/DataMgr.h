#pragma once

#include "CMesh.h".h"
#include "ParameterMgr.h"
#include "GlobalFunction.h"
//#include "vcg\complex\trimesh\update\selection.h"

#include <qfile.h>
#include <qtextstream.h>
#include <qtextcodec.h>

#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export.h>

#include <sstream>
#include <fstream>
#include <set>
#include <utility>


using namespace vcg;
using namespace std;
using namespace tri;

typedef pair<Point3f, Point3f> ScanCandidate;
typedef vcg::tri::UpdateFlags<CMesh>::EdgeSorter MyBoarderEdge;

class DataMgr
{
public:
	DataMgr(RichParameterSet* _para);
	~DataMgr(void);

  void      loadPlyToModel(QString fileName);
  void      loadPlyToOriginal(QString fileName);
  void      loadPlyToSample(QString fileName);
  void      loadPlyToISO(QString fileName);
  void      loadPlyToPoisson(QString fileName);
  void      loadPlyToNBV(QString fileName);
  void      saveParameters(QString fileName);
  void      loadParameters(QString fileName);
	void      savePly(QString fileName, CMesh& mesh);
	void      loadImage(QString fileName);
  void      loadXYZN(QString fileName);
  void      loadCameraModel(QString fileName);

  bool      isModelEmpty();
  bool      isSamplesEmpty();
  bool      isOriginalEmpty();
  bool      isIsoPointsEmpty();
  bool      isFieldPointsEmpty();
  bool      isViewCandidatesEmpty();
  bool      isScannedMeshEmpty();
  bool      isScannedResultsEmpty();
  bool      isPoissonSurfaceEmpty();
  bool      isViewGridsEmpty();
  bool      isNBVCandidatesEmpty();

  void                    setCurrentTemperalSample(CMesh *mesh);
  CMesh*                  getCurrentSamples();
  CMesh*                  getCurrentTemperalSamples();
  CMesh*                  getCurrentModel();
  CMesh*                  getCurrentPoissonSurface();
  CMesh*                  getCurrentOriginal();
  CMesh*                  getCurrentTemperalOriginal();
  CMesh*                  getCurrentIsoPoints();
  CMesh*                  getCurrentFieldPoints();
  Slices*                 getCurrentSlices();
  
  CMesh*                  getCameraModel();
  Point3f&                getCameraPos();
  Point3f&                getCameraDirection();
  double                  getCameraResolution();
  double                  getCameraHorizonDist();
  double                  getCameraVerticalDist();
  double                  getCameraMaxDistance();
  double                  getCameraMaxAngle();
  CMesh*                  getViewGridPoints();
  CMesh*                  getNbvCandidates();
  vector<ScanCandidate>*  getInitCameraScanCandidates();
  vector<ScanCandidate>*  getScanCandidates();
  vector<ScanCandidate>*  getScanHistory();
  vector<ScanCandidate>*  getSelectedScanCandidates();
  vector<ScanCandidate>*  getVisibilityFirstScanCandidates();
  vector<ScanCandidate>*  getPVSFirstScanCandidates();
  CMesh*                  getCurrentScannedMesh();
  vector<CMesh* >*        getScannedResults();
  int*                    getScanCount();

	void      recomputeBox();
	double    getInitRadiuse();
  
	void      downSamplesByNum(bool use_random_downsample = true);
	void      subSamples();

	void      normalizeROSA_Mesh(CMesh& mesh, bool is_original = false);
	Box3f     normalizeAllMesh();

  void     eraseRemovedSamples();
  void     clearData();
  void     recomputeQuad();

  void     saveFieldPoints(QString fileName);
  void     saveViewGrids(QString fileName);
  void     saveMergedMesh(QString fileName);
  
  void switchSampleToOriginal();
  void switchSampleToISO();
  void switchSampleToNBV();

  void replaceMesh(CMesh& src_mesh, CMesh& target_mesh, bool isOriginal);
  void replaceMeshISO(CMesh& src_mesh, CMesh& target_mesh, bool isIso);
  void replaceMeshView(CMesh& src_mesh, CMesh& target_mesh, bool isViewGrid);

//  void cleanScannedMesh();
    
private:
	void clearCMesh(CMesh& mesh);
  void initDefaultScanCamera();

public:
  CMesh                  model;
  CMesh                  original;
  CMesh                  poisson_surface;
  CMesh                 *temperal_original;
  Point3f                original_center_point;
  CMesh                  samples;
  CMesh                 *temperal_sample;
  CMesh                  iso_points;
  CMesh                  field_points;
  CMesh                  camera_model;
  CMesh                  view_grid_points;
  CMesh                  nbv_candidates;
  Point3f                camera_pos;
  Point3f                camera_direction;
  double                 camera_horizon_dist;
  double                 camera_vertical_dist;
  double                 camera_resolution;
  double                 camera_max_distance;
  double                 camera_max_angle;
  vector<ScanCandidate>  init_scan_candidates;
  vector<ScanCandidate>  scan_candidates;
  vector<ScanCandidate>  selected_scan_candidates;
  vector<ScanCandidate>  scan_history;
  CMesh                  current_scanned_mesh; 
  vector<CMesh *>        scanned_results;  
  Slices                 slices;

	RichParameterSet*      para;
	double                 init_radius;
	QString                curr_file_name;

  Box3f                  whole_space_box;
  Point3f                scanner_position;  
  int                    scan_count;
};
