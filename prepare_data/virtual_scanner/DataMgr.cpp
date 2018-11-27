#include "DataMgr.h"
//#include "GLDrawer.h"

//ParameterMgr global_paraMgr;

DataMgr::DataMgr(RichParameterSet* _para)
{
  para = _para;
  camera_pos = Point3f(0.0f, 0.0f, 1.0f);
  camera_direction = Point3f(0.0f, 0.0f, -1.0f);
  scan_count = 0;
  initDefaultScanCamera();

  whole_space_box.Add(Point3f(2.0, 2.0, 2.0));
  whole_space_box.Add(Point3f(-2.0, -2.0, -2.0));
  scanner_position = Point3f(134, 13, -293); //sheep 5-19
  slices.assign(3, Slice());
}

DataMgr::~DataMgr(void)
{
}

void DataMgr::clearCMesh(CMesh& mesh)
{
  mesh.face.clear();
  mesh.fn = 0;
  mesh.vert.clear();
  mesh.vn = 0;
  mesh.bbox = Box3f();
}

void DataMgr::initDefaultScanCamera()
{
  double predict_size = global_paraMgr.camera.getDouble("Predicted Model Size");
  double far_dist = global_paraMgr.camera.getDouble("Camera Far Distance") / predict_size;
  double camera_dist_to_model = global_paraMgr.camera.getDouble("Camera Dist To Model") / predict_size;
  //default cameras for initial scanning, pair<pos, direction>
  //x axis
  init_scan_candidates.push_back(make_pair(Point3f(1.0f * camera_dist_to_model, 0.0f, 0.0f), Point3f(-1.0f, 0.0f, 0.0f)));
  init_scan_candidates.push_back(make_pair(Point3f(-1.0f * camera_dist_to_model, 0.0f, 0.0f), Point3f(1.0f, 0.0f, 0.0f)));
  ////z axis
  //init_scan_candidates.push_back(make_pair(Point3f(0.0f, 0.0f, 1.0f * camera_dist_to_model), Point3f(0.0f, 0.0f, -1.0f)));
  //init_scan_candidates.push_back(make_pair(Point3f(0.0f, 0.0f, -1.0f * camera_dist_to_model), Point3f(0.0f, 0.0f, 1.0f)));
  //y axis
  //init_scan_candidates.push_back(make_pair(Point3f(0.0f, 1.0f * camera_dist_to_model, 0.0f), Point3f(0.0f, -1.0f * camera_dist_to_model, 0.0f)));
  //init_scan_candidates.push_back(make_pair(Point3f(0.0f, -1.0f * camera_dist_to_model, 0.0f), Point3f(0.0f, 1.0f * camera_dist_to_model, 0.0f)));
  
  //another four angles
  /*init_scan_candidates.push_back(make_pair(Point3f(-1.0f * far_dist / sqrt(2.0f), 0.0f, 1.0f * far_dist / sqrt(2.0f)), Point3f(1.0f , 0.0f, -1.0f )));
  init_scan_candidates.push_back(make_pair(Point3f(1.0f * far_dist / sqrt(2.0f), 0.0f, 1.0f * far_dist / sqrt(2.0f)), Point3f(-1.0f , 0.0f, -1.0f)));
  init_scan_candidates.push_back(make_pair(Point3f(1.0f * far_dist / sqrt(2.0f), 0.0f, -1.0f * far_dist / sqrt(2.0f)), Point3f(-1.0f, 0.0f, 1.0f)));
  init_scan_candidates.push_back(make_pair(Point3f(-1.0f * far_dist / sqrt(2.0f), 0.0f, -1.0f * far_dist / sqrt(2.0f)), Point3f(1.0f, 0.0f, 1.0f)));*/

  //this should be deleted, for UI debug
  //for test
  scan_candidates.push_back(make_pair(Point3f(0.0f, 0.0f, 1.0f * far_dist), Point3f(0.0f, 0.0f, -1.0f)));
  //x axis
  scan_candidates.push_back(make_pair(Point3f(1.0f * far_dist, 0.0f, 0.0f), Point3f(-1.0f, 0.0f, 0.0f)));
  //scan_candidates.push_back(make_pair(Point3f(-1.0f, 0.0f, 0.0f), Point3f(1.0f, 0.0f, 0.0f)));
  ////y axis
  //scan_candidates.push_back(make_pair(Point3f(0.0f, 1.0f, 0.0f), Point3f(0.0f, -1.0f, 0.0f)));
  //scan_candidates.push_back(make_pair(Point3f(0.0f, -1.0f, 0.0f), Point3f(0.0f, 1.0f, 0.0f)));
  ////z axis
  //scan_candidates.push_back(make_pair(Point3f(0.0f, 0.0f, 1.0f), Point3f(0.0f, 0.0f, -1.0f)));
  //scan_candidates.push_back(make_pair(Point3f(0.0f, 0.0f, -1.0f), Point3f(0.0f, 0.0f, 1.0f)));
}

bool DataMgr::isSamplesEmpty()
{
  return samples.vert.empty();
}

bool DataMgr::isModelEmpty()
{
  return model.vert.empty();
}

bool DataMgr::isOriginalEmpty()
{
  return original.vert.empty();
}

bool DataMgr::isIsoPointsEmpty()
{
  return iso_points.vert.empty();
}

bool DataMgr::isFieldPointsEmpty()
{
  return field_points.vert.empty();
}

bool DataMgr::isScannedMeshEmpty()
{
  return current_scanned_mesh.vert.empty();
}

bool DataMgr::isScannedResultsEmpty()
{
  return scanned_results.empty();
}

bool DataMgr::isPoissonSurfaceEmpty()
{
  return poisson_surface.vert.empty();
}

bool DataMgr::isViewGridsEmpty()
{
  return view_grid_points.vert.empty();
}

bool DataMgr::isNBVCandidatesEmpty()
{
  return nbv_candidates.vert.empty();
}

void DataMgr::loadPlyToModel(QString fileName)
{
  clearCMesh(model);
  curr_file_name = fileName;

  int mask = tri::io::Mask::IOM_ALL;
  int err = tri::io::Importer<CMesh>::Open(model, curr_file_name.toStdString().c_str(), mask);
  if (err)
  {
    cout<<"Failed to read model: "<< err <<"\n";
    return;
  }
  cout<<"object model loaded \n";

  CMesh::VertexIterator vi;
  int idx = 0;
  for (vi = model.vert.begin(); vi != model.vert.end(); ++vi)
  {
    vi->is_model = true;
    vi->m_index = idx++;
    model.bbox.Add(vi->P());
  }
  model.vn = model.vert.size();
}

void DataMgr::loadPlyToOriginal(QString fileName)
{
  clearCMesh(original);
  curr_file_name = fileName;

  int mask = tri::io::Mask::IOM_VERTCOORD + tri::io::Mask::IOM_VERTNORMAL
    + tri::io::Mask::IOM_VERTCOLOR; 

   mask += tri::io::Mask::IOM_ALL + tri::io::Mask::IOM_FACEINDEX;

  int err = tri::io::Importer<CMesh>::Open(original, curr_file_name.toStdString().c_str(), mask);
  if(err) 
  {
    cout << "Failed reading mesh: " << err << "\n";
    return;
  }  
  cout << "points loaded\n";

  //vcg::tri::UpdateNormals<CMesh>::PerVertex(original);

  CMesh::VertexIterator vi;
  int idx = 0;
  for(vi = original.vert.begin(); vi != original.vert.end(); ++vi)
  {
    vi->is_original = true;
    vi->m_index = idx++;
    vi->N().Normalize();
    //vi->N() = Point3f(0, 0, 0);
    original.bbox.Add(vi->P());
  }
  original.vn = original.vert.size();
  cout<<"original vert size: " <<original.vert.size() <<endl;
}

void DataMgr::loadPlyToSample(QString fileName)
{
  clearCMesh(samples);
  curr_file_name = fileName;

  int mask= tri::io::Mask::IOM_VERTCOORD + tri::io::Mask::IOM_VERTNORMAL ;
  mask += tri::io::Mask::IOM_VERTCOLOR;
  mask += tri::io::Mask::IOM_BITPOLYGONAL;
  mask += tri::io::Mask::IOM_ALL;

  int err = tri::io::Importer<CMesh>::Open(samples, curr_file_name.toStdString().c_str(), mask);
  if(err) 
  {
    cout << "Failed reading mesh: " << err << "\n";
    return;
  }  

  CMesh::VertexIterator vi;
  int idx = 0;
  for(vi = samples.vert.begin(); vi != samples.vert.end(); ++vi)
  {
    vi->is_original = false;
    vi->m_index = idx++;
    samples.bbox.Add(vi->P());
  }
  samples.vn = samples.vert.size();
}

void DataMgr::loadPlyToISO(QString fileName)
{
  clearCMesh(iso_points);
  curr_file_name = fileName;

  int mask= tri::io::Mask::IOM_VERTCOORD + tri::io::Mask::IOM_VERTNORMAL ;
  mask += tri::io::Mask::IOM_VERTCOLOR;
  mask += tri::io::Mask::IOM_BITPOLYGONAL;

  int err = tri::io::Importer<CMesh>::Open(iso_points, curr_file_name.toStdString().c_str(), mask);
  if(err) 
  {
    cout << "Failed reading mesh: " << err << "\n";
    return;
  }  

  CMesh::VertexIterator vi;
  int idx = 0;
  for(vi = iso_points.vert.begin(); vi != iso_points.vert.end(); ++vi)
  {
    vi->is_iso = true;
    vi->m_index = idx++;
    iso_points.bbox.Add(vi->P());
  }
  iso_points.vn = iso_points.vert.size();
}

void DataMgr::loadPlyToPoisson(QString fileName)
{
  clearCMesh(poisson_surface);
  curr_file_name = fileName;

  int mask= tri::io::Mask::IOM_VERTCOORD + tri::io::Mask::IOM_VERTNORMAL ;

  int err = tri::io::Importer<CMesh>::Open(poisson_surface, curr_file_name.toStdString().c_str(), mask);
  if(err) 
  {
    cout << "Failed reading mesh: " << err << "\n";
    return;
  }  
  cout << "points loaded\n";

  CMesh::VertexIterator vi;
  int idx = 0;
  for(vi = poisson_surface.vert.begin(); vi != poisson_surface.vert.end(); ++vi)
  {
    vi->is_poisson = true;
    vi->m_index = idx++;
    poisson_surface.bbox.Add(vi->P());
  }
  poisson_surface.vn = poisson_surface.vert.size();
}

void DataMgr::loadPlyToNBV(QString fileName)
{
  clearCMesh(nbv_candidates);
  curr_file_name = fileName;

  int mask= tri::io::Mask::IOM_VERTCOORD + tri::io::Mask::IOM_VERTNORMAL ;
  mask += tri::io::Mask::IOM_VERTCOLOR;
  mask += tri::io::Mask::IOM_BITPOLYGONAL;
  mask += tri::io::Mask::IOM_ALL;

  int err = tri::io::Importer<CMesh>::Open(nbv_candidates, curr_file_name.toStdString().c_str(), mask);
  if(err) 
  {
    cout << "Failed reading mesh: " << err << "\n";
    return;
  }  

  CMesh::VertexIterator vi;
  int idx = 0;
  for(vi = nbv_candidates.vert.begin(); vi != nbv_candidates.vert.end(); ++vi)
  {
    vi->is_original = false;
    vi->m_index = idx++;
    nbv_candidates.bbox.Add(vi->P());
  }
  nbv_candidates.vn = nbv_candidates.vert.size();

  //put nbv_candidates into scan_candidates
  vector<ScanCandidate> *scan_candidate = getScanCandidates();
  for (int i = 0; i < nbv_candidates.vert.size(); ++i){
    CVertex &v = nbv_candidates.vert[i];
    scan_candidate->push_back(make_pair(Point3f(v.P()[0], v.P()[1], v.P()[2]),  
                                        Point3f(v.N()[0], v.N()[2], v.N()[2])));
  }
}

void DataMgr::loadXYZN(QString fileName)
{
//  clearCMesh(samples);
//  ifstream infile;
//  infile.open(fileName.toStdString().c_str());

//  int i = 0;
//  while(!infile.eof())
//  {
//    CVertex v;
//    float temp = 0.;
//    for (int j=0; j<3; j++)
//    {

//      infile >> temp;
//      v.P()[j] = temp;
//    }


//    for (int j=0; j<3; j++) {
//      infile >> v.N()[j];
//    }

//    v.m_index = i++;

//    samples.vert.push_back(v);
//    samples.bbox.Add(v.P());
//  }

//  // mesh.vert.erase(mesh.vert.end()-1);
//  samples.vert.pop_back();
//  samples.vn = samples.vert.size();

//  infile.close();



}

void DataMgr::loadImage(QString fileName)
{

  //image = cv::imread(fileName.toAscii().data());

  ////cv::namedWindow("image", CV_WINDOW_AUTOSIZE);
  ////cv::imshow("image", image);

  //clearCMesh(samples);
  //clearCMesh(original);
  //int cnt = 0;
  //for (int i = 0; i < image.rows; i++)
  //{
  //	for (int j = 0; j < image.cols; j++)
  //	{
  //		cv::Vec3b intensity = image.at<cv::Vec3b>(i, j);
  //		Point3f p;
  //		Color4b c;
  //		c.Z() = 1;
  //		p.X() = c.X() = intensity.val[0];
  //		p.Y() = c.Y() = intensity.val[1];
  //		p.Z() = c.Z() = intensity.val[2];
  //		CVertex new_v;
  //		new_v.P() = p;
  //		new_v.C() = c;
  //		new_v.m_index = cnt++;

  //		samples.vert.push_back(new_v);
  //		samples.bbox.Add(p);

  //		new_v.is_original = true;
  //		original.vert.push_back(new_v);
  //		original.bbox.Add(p);
  //	}
  //}
  //samples.vn = samples.vert.size();
  //original.vn = samples.vn;

  //cv::waitKey();

}

void DataMgr::loadCameraModel(QString fileName)
{
  clearCMesh(camera_model);
  curr_file_name = fileName;
  int mask = tri::io::Mask::IOM_VERTCOORD + tri::io::Mask::IOM_VERTNORMAL;
  mask += tri::io::Mask::IOM_FACEFLAGS;

  int err = tri::io::Importer<CMesh>::Open(camera_model, curr_file_name.toStdString().c_str(), mask);
  if (err)
  {
    cout<<"Failed to read camera model: "<< err << "\n";
    return;
  }
  cout<<"camera model loaded \n";
}

void DataMgr::setCurrentTemperalSample(CMesh *mesh)
{
  this->temperal_sample = mesh;
}

CMesh* DataMgr::getCurrentIsoPoints()
{
  if(&iso_points == NULL) return NULL;

  return & iso_points;
}

CMesh* DataMgr::getCurrentFieldPoints()
{
  if(&field_points == NULL) return NULL;

  return & field_points;
}


CMesh* DataMgr::getCurrentSamples()
{
  if(&samples == NULL) return NULL;

  return & samples;
}

CMesh* DataMgr::getCurrentTemperalSamples()
{
  return temperal_sample;
}

CMesh* DataMgr::getCurrentModel()
{
  return &model;
}

CMesh* DataMgr::getCurrentPoissonSurface()
{
  return &poisson_surface;
}

CMesh* DataMgr::getCurrentOriginal()
{
  if(&original == NULL) return NULL;

  return & original;
}

CMesh* DataMgr::getCurrentTemperalOriginal()
{
  return temperal_original;
}

CMesh* DataMgr::getCameraModel()
{
  return &camera_model;
}

Point3f& DataMgr::getCameraPos()
{
  return camera_pos;
}

Point3f& DataMgr::getCameraDirection()
{
  return camera_direction;
}

double DataMgr::getCameraResolution()
{
  return camera_resolution;
}

double DataMgr::getCameraHorizonDist()
{
  return camera_horizon_dist;
}

double DataMgr::getCameraVerticalDist()
{
  return camera_vertical_dist;
}

double DataMgr::getCameraMaxDistance()
{
  return camera_max_distance;
}

double DataMgr::getCameraMaxAngle()
{
  return camera_max_angle;
}

CMesh*
  DataMgr::getViewGridPoints()
{
  return &view_grid_points;
}

CMesh* DataMgr::getNbvCandidates()
{
  return &nbv_candidates;
}

vector<ScanCandidate>* DataMgr::getInitCameraScanCandidates()
{
  return &init_scan_candidates;
}

vector<ScanCandidate>* DataMgr::getScanCandidates()
{
  return &scan_candidates;
}

vector<ScanCandidate>* DataMgr::getScanHistory()
{
  return &scan_history;
}

vector<ScanCandidate>* DataMgr::getSelectedScanCandidates()
{
  return &selected_scan_candidates;
}

CMesh* DataMgr::getCurrentScannedMesh()
{
  return &current_scanned_mesh;
}

vector<CMesh* >* DataMgr::getScannedResults()
{
  return &scanned_results;
}

int* DataMgr::getScanCount()
{
  return &scan_count;
}

Slices* DataMgr::getCurrentSlices()
{
  return &slices;
}

void DataMgr::recomputeBox()
{
  model.bbox.SetNull();
  samples.bbox.SetNull();
  original.bbox.SetNull();

  CMesh::VertexIterator vi;
  for (vi = model.vert.begin(); vi != model.vert.end(); ++vi)
  {
    model.bbox.Add(vi->P());
  }

  for(vi = samples.vert.begin(); vi != samples.vert.end(); ++vi) 
  {
    if (vi->is_ignore)
    {
      continue;
    }
    samples.bbox.Add(vi->P());
  }

  for(vi = original.vert.begin(); vi != original.vert.end(); ++vi) 
  {
    original.bbox.Add(vi->P());
  }

  double camera_max_dist = global_paraMgr.camera.getDouble("Camera Far Distance") /
    global_paraMgr.camera.getDouble("Predicted Model Size"); 
  float scan_box_size = camera_max_dist + 0.5;

  Point3f whole_space_box_min = Point3f(-scan_box_size, -scan_box_size, -scan_box_size);
  Point3f whole_space_box_max = Point3f(scan_box_size, scan_box_size, scan_box_size);
  whole_space_box.SetNull();
  whole_space_box.Add(whole_space_box_min);
  whole_space_box.Add(whole_space_box_max);

}

double DataMgr::getInitRadiuse()
{
  double init_para = para->getDouble("Init Radius Para");
//  if (isOriginalEmpty() && isModelEmpty())
//  {
//    global_paraMgr.setGlobalParameter("CGrid Radius", DoubleValue(init_radius));
//    global_paraMgr.setGlobalParameter("Initial Radius", DoubleValue(init_radius));
//    return init_radius;
//  }

  Box3f box;
  if (!isOriginalEmpty())   box = original.bbox;
  else if (!isModelEmpty()) box = model.bbox;

  if ( abs(box.min.X() - box.max.X()) < 1e-5 ||   
    abs(box.min.Y() - box.max.Y()) < 1e-5 ||   
    abs(box.min.Z() - box.max.Z()) < 1e-5 )
  {
    double diagonal_length = sqrt((box.min - box.max).SquaredNorm());
    double original_size = sqrt(double(original.vn));
    init_radius = 2 * init_para * diagonal_length / original_size;
  }
  else
  {
    double diagonal_length = sqrt((box.min - box.max).SquaredNorm());
    double original_size = pow(double(original.vn), 0.333);
    init_radius = init_para * diagonal_length / original_size;
  }

//  global_paraMgr.setGlobalParameter("CGrid Radius", DoubleValue(init_radius));
//  global_paraMgr.setGlobalParameter("Initial Radius", DoubleValue(init_radius));

  return init_radius;
}

void DataMgr::downSamplesByNum(bool use_random_downsample)
{
  if (isOriginalEmpty() && !isSamplesEmpty())
  {
    subSamples();
    return;
  }

  if (isOriginalEmpty())  return;

  int want_sample_num = para->getDouble("Down Sample Num");

  if (want_sample_num > original.vn)
    want_sample_num = original.vn;

  clearCMesh(samples);
  samples.vn = want_sample_num;

  vector<int> nCard = GlobalFun::GetRandomCards(original.vert.size());
  for(int i = 0; i < samples.vn; i++) 
  {
    int index = nCard[i]; //could be not random!

    if (!use_random_downsample)
    {
      index = i;
    }

    CVertex& v = original.vert[index];
    samples.vert.push_back(v);
    samples.bbox.Add(v.P());
  }

  CMesh::VertexIterator vi;
  for(vi = samples.vert.begin(); vi != samples.vert.end(); ++vi)
  {
    vi->is_original = false;
  }

  getInitRadiuse();
}

void DataMgr::subSamples()
{
  clearCMesh(original);

  CMesh::VertexIterator vi;
  original.vn = samples.vert.size();
  original.bbox.SetNull();
  for(vi = samples.vert.begin(); vi != samples.vert.end(); ++vi)
  {
    CVertex v = (*vi);
    v.is_original = true;
    original.vert.push_back(v);
    original.bbox.Add(v.P());
  }

  downSamplesByNum();
  getInitRadiuse();
}


void DataMgr::savePly(QString fileName, CMesh& mesh)
{
  //int mask= tri::io::Mask::IOM_VERTNORMAL ;
  //mask += tri::io::Mask::IOM_VERTCOLOR;
  int mask = tri::io::Mask::IOM_ALL;
  //mask += tri::io::Mask::IOM_BITPOLYGONAL;
  //mask += tri::io::Mask::IOM_FACEINDEX;

  //GLDrawer drawer(global_paraMgr.getDrawerParameterSet());
  //for (int i = 0; i < mesh.vert.size(); i++)
  //{
  //  CVertex& v = mesh.vert[i];
  //  GLColor c = drawer.getColorByType(v);
  //  ////QColor qc(c.r * 255.0, c.g * 255.0, c.b * 255.0);
  //  //vcg::Color4f color;
  //  //color.X() = c.r * 255.0;
  //  //color.Y() = c.g * 255.0;
  //  //color.Z() = c.b * 255.0;
  //  v.C().SetRGB(255, 0, 0);

  //}
  if (fileName.endsWith("ply"))
    tri::io::ExporterPLY<CMesh>::Save(mesh, fileName.toStdString().c_str(), mask, false);
}

void DataMgr::normalizeROSA_Mesh(CMesh& mesh, bool is_original)
{
  if (mesh.vert.empty()) return;

  mesh.bbox.SetNull();
  Box3f box = mesh.bbox;

  float max_length = global_paraMgr.data.getDouble("Max Normalize Length");

  Box3f box_temp;
  for(int i = 0; i < mesh.vert.size(); i++)
  {
    Point3f& p = mesh.vert[i].P();

    p /= max_length;

    mesh.vert[i].N().Normalize(); 
    box_temp.Add(p);
  }

  Point3f mid_point = (box_temp.min + box_temp.max) / 2.0;

  for(int i = 0; i < mesh.vert.size(); i++)
  {
    Point3f& p = mesh.vert[i].P();
    p -= mid_point;
    mesh.bbox.Add(p);
  }

  if (is_original)
  {
    this->original_center_point = mid_point;
  }
}


Box3f DataMgr::normalizeAllMesh()
{
  Box3f box;
  if (!isModelEmpty())
  {
    for (int i = 0; i < model.vert.size(); ++i)
      box.Add(model.vert[i].P());
  }

  if (!isSamplesEmpty())
  {
    for (int i = 0; i < samples.vert.size(); ++i)
      box.Add(samples.vert[i].P());
  }

  if (!isOriginalEmpty())
  {
    for (int i = 0; i < original.vert.size(); ++i)
      box.Add(original.vert[i].P());
  }

  model.bbox = box;
  original.bbox =box;
  samples.bbox = box;

  float max_x = abs((box.min - box.max).X());
  float max_y = abs((box.min - box.max).Y());
  float max_z = abs((box.min - box.max).Z());
  float max_length = std::max(max_x, std::max(max_y, max_z));
  global_paraMgr.data.setValue("Max Normalize Length", DoubleValue(max_length));

  normalizeROSA_Mesh(model);
  normalizeROSA_Mesh(original, true);
  normalizeROSA_Mesh(samples);
  normalizeROSA_Mesh(iso_points);

  recomputeBox();
  getInitRadiuse();

  return samples.bbox;
}


void DataMgr::eraseRemovedSamples()
{
  int cnt = 0;
  vector<CVertex> temp_mesh;
  for (int i = 0; i < samples.vert.size(); i++)
  {
    CVertex& v = samples.vert[i];
    if (!v.is_ignore)
    {
      temp_mesh.push_back(v);
    }
  }

  samples.vert.clear();
  samples.vn = temp_mesh.size();
  for (int i = 0; i < temp_mesh.size(); i++)
  {
    temp_mesh[i].m_index = i;
    samples.vert.push_back(temp_mesh[i]);
  }
}

void DataMgr::clearData()
{
  clearCMesh(original);
  clearCMesh(samples);
  clearCMesh(iso_points);
  clearCMesh(field_points);

  clearCMesh(model);  
  clearCMesh(current_scanned_mesh);

  clearCMesh(view_grid_points);
  clearCMesh(nbv_candidates);
  clearCMesh(current_scanned_mesh);

  slices.clear();
}

void DataMgr::recomputeQuad()
{
  for (int i = 0; i < samples.vert.size(); i++)
  {
    samples.vert[i].recompute_m_render();
  }
  for (int i = 0; i < iso_points.vert.size(); i++)
  {
    iso_points.vert[i].recompute_m_render();
  }
  for (int i = 0; i < original.vert.size(); i++)
  {
    original.vert[i].recompute_m_render();
  }
}

bool cmp_angle(const CVertex &v1, const CVertex &v2)
{
  if (v1.ground_angle == v2.ground_angle) 
    return false;

  //in ascending order
  return v1.ground_angle > v2.ground_angle;
}

void DataMgr::saveFieldPoints(QString fileName)
{
//  if (field_points.vert.empty())
//  {
//    cout<<"save Field Points Error: Empty field_points" <<endl;
//    return;
//  }

//  //ofstream outfile;
//  //outfile.open(fileName.toStdString().c_str());

//  //ostringstream strStream;

//  //strStream << "ON " << original.vert.size() << endl;

//  //FILE *fp = fopen(fileName.toStdString().c_str(),"rb");
//  //if(!fp)
//  //  cerr<<"open "<< fileName.toStdString().c_str() <<" failed"<<endl;

//  ofstream fout;
//  fout.open(fileName.toStdString(), std::ios::out | std::ios::binary);
//  if( fout == NULL)
//  {
//    cout<<" error ----- "<<endl;
//    return;
//  }

//  //for (int i = 0; i < field_points.vert.size(); i++)
//  cout << field_points.vert.size() << " grids" << endl;
//  for (int i = 0; i < field_points.vert.size(); i++)
//  {
//    CVertex& v = field_points.vert[i];
//    float eigen_value = v.eigen_confidence * 255;

//    unsigned char pTest = static_cast<unsigned char>(eigen_value);

//    fout << pTest;
//  }
//  fout.close();

//  ofstream fout_dat;
//  QString fileName_dat = fileName;
//  QString temp = fileName;
//  QStringList str_list = temp.split(QRegExp("[/]"));
//  QString last_name = str_list.at(str_list.size()-1);
//  cout << "file name: " << last_name.toStdString() << endl;

//  int resolution = global_paraMgr.poisson.getInt("Field Points Resolution");
//  fileName_dat.replace(".raw", ".dat");
//  fout_dat.open(fileName_dat.toStdString());
//  fout_dat << "ObjectFileName:  " << last_name.toStdString() << endl;
//  fout_dat << "Resolution:  " << resolution << " " << resolution << " " << resolution << endl;
//  fout_dat << "SliceThickness:	0.0127651 0.0127389 0.0128079" << endl;
//  fout_dat << "Format:		    UCHAR" << endl;
//  fout_dat << "ObjectModel:	I" << endl;
//  fout_dat << "Modality:	    CT" << endl;
//  fout_dat << "Checksum:	    7b197a4391516321308b81101d6f09e8" << endl;
//  fout_dat.close();
}

void
  DataMgr::saveViewGrids(QString fileName)
{
//  if (view_grid_points.vert.empty()) return;

//  ofstream out;
//  out.open(fileName.toStdString(), std::ios::out | std::ios::binary);
//  if (NULL == out)
//  {
//    cout<<"open file Error!" <<endl;
//    return;
//  }

//  for (int i = 0; i < view_grid_points.vert.size(); ++i)
//  {
//    CVertex &v = view_grid_points.vert[i];
//    float eigen_value = v.eigen_confidence * 255;
//    unsigned char p = static_cast<unsigned char>(eigen_value);
//    out << p;
//  }
//  out.close();

//  QString tmp = fileName;
//  QStringList str_lst = tmp.split(QRegExp("[/]"));
//  QString last_name = str_lst.at(str_lst.size() - 1);

//  double resolution = global_paraMgr.nbv.getDouble("View Grid Resolution");
//  ofstream out_dat;
//  QString fileName_dat = fileName;
//  fileName_dat.replace(".raw", ".dat");
//  out_dat.open(fileName_dat.toStdString());
//  out_dat << "ObjectFileName:  " << last_name.toStdString() << endl;
//  out_dat << "Resolution:  " << resolution << " " << resolution << " " << resolution << endl;
//  out_dat << "SliceThickness:	0.0127651 0.0127389 0.0128079" << endl;
//  out_dat << "Format:		    UCHAR" << endl;
//  out_dat << "ObjectModel:	I" << endl;
//  out_dat << "Modality:	    CT" << endl;
//  out_dat << "Checksum:	    7b197a4391516321308b81101d6f09e8" << endl;
//  out_dat.close();
}

void
  DataMgr::saveMergedMesh(QString fileName)
{
  for(int i = 0; i < scanned_results.size(); ++i)
  {
    QString s_i;
    s_i.sprintf("_%d.ply", i);
    QString r = fileName + s_i;
    savePly(r, *scanned_results[i]);
  }
}

void 
  DataMgr::saveParameters(QString fileName)
{
  ofstream out_para;
  out_para.open(fileName.toStdString().data(), std::ios::out);
  if (!out_para)
    return ;

  out_para << "#1. KNN for compute PCA normal" << endl
    << global_paraMgr.norSmooth.getInt("PCA KNN") << endl << endl; 

  out_para << "#2. Camera Resolution, something like(1.0 / 50.0f)" << endl
    << global_paraMgr.camera.getDouble("Camera Resolution") << endl << endl;

  out_para << "#3. Sharp Sigma" << endl
    << global_paraMgr.norSmooth.getDouble("Sharpe Feature Bandwidth Sigma") << endl <<endl;

  out_para << "#4. View Grid Resolution" <<endl
    << global_paraMgr.nbv.getDouble("View Grid Resolution") << endl <<endl;

  out_para << "#5. Poisson Max Depth" <<endl
    << global_paraMgr.poisson.getDouble("Max Depth") << endl <<endl;

  out_para << "#6. Original KNN" <<endl
    << global_paraMgr.poisson.getDouble("Original KNN") << endl << endl;

  out_para << "#7. merge probability X . pow(1-confidence, x)" <<endl
    << global_paraMgr.nbv.getDouble("Merge Probability Pow") <<endl <<endl;

  out_para << "#8. Optimal Plane Width" <<endl
    << global_paraMgr.camera.getDouble("Optimal Plane Width") <<endl <<endl;

  out_para << "#9. Merge Confidence Threshold" << endl
    << global_paraMgr.camera.getDouble("Merge Confidence Threshold") <<endl << endl;

  out_para << "#10. View Bin Number On Each Axis" << endl
    << global_paraMgr.nbv.getInt("View Bin Each Axis") <<endl << endl;

  out_para.close();

  std::cout<<"save parameters to ./"<<fileName.toStdString() <<std::endl;
}

void DataMgr::loadParameters(QString fileName)
{
  ifstream in_para;
  in_para.open(fileName.toStdString().data());
  if (!in_para)
    return;

  string value;

  in_para.ignore(1000, '\n');
  int knn;
  getline(in_para, value);
  knn = atoi(value.c_str());
  global_paraMgr.norSmooth.setValue("PCA KNN", IntValue(knn));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  double camera_resolution;
  getline(in_para, value);
  camera_resolution = atof(value.c_str());
  global_paraMgr.camera.setValue("Camera Resolution", DoubleValue(camera_resolution));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  double sharp_sigma;
  getline(in_para, value);
  sharp_sigma = atof(value.c_str());
  global_paraMgr.norSmooth.setValue("Sharpe Feature Bandwidth Sigma", DoubleValue(sharp_sigma));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  int grid_resolution;
  getline(in_para, value);
  grid_resolution = atoi(value.c_str());
  global_paraMgr.nbv.setValue("View Grid Resolution", DoubleValue(grid_resolution));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  int poisson_depth;
  getline(in_para, value);
  poisson_depth = atoi(value.c_str());
  global_paraMgr.poisson.setValue("Max Depth", DoubleValue(poisson_depth));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  double original_knn;
  getline(in_para, value);
  original_knn = atof(value.c_str());
  global_paraMgr.poisson.setValue("Original KNN", DoubleValue(original_knn));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  double merge_pow;
  getline(in_para, value);
  merge_pow = atof(value.c_str());
  global_paraMgr.nbv.setValue("Merge Probability Pow", DoubleValue(merge_pow));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  double optimal_plane_width;
  getline(in_para, value);
  optimal_plane_width = atof(value.c_str());
  global_paraMgr.camera.setValue("Optimal Plane Width", DoubleValue(optimal_plane_width));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  double merge_confidence_threshold;
  getline(in_para, value);
  merge_confidence_threshold = atof(value.c_str());
  global_paraMgr.camera.setValue("Merge Confidence Threshold", DoubleValue(merge_confidence_threshold));

  in_para.ignore(1000, '\n');
  in_para.ignore(1000, '\n');
  int nbv_bin_num;
  getline(in_para, value);
  nbv_bin_num = atoi(value.c_str());
  global_paraMgr.nbv.setValue("View Bin Each Axis", IntValue(nbv_bin_num));

  in_para.close();
}

void DataMgr::switchSampleToOriginal()
{
  CMesh temp_mesh;
  replaceMesh(original, temp_mesh, false);
  replaceMesh(samples, original, true);
  replaceMesh(temp_mesh, samples, false);
}

void DataMgr::switchSampleToISO()
{
  CMesh temp_mesh;
  replaceMeshISO(iso_points, temp_mesh, false);
  replaceMeshISO(samples, iso_points, true);
  replaceMeshISO(temp_mesh, samples, false);
}

void DataMgr::switchSampleToNBV()
{
  CMesh temp_mesh;
  replaceMeshView(nbv_candidates, temp_mesh, false);
  replaceMeshView(samples, nbv_candidates, true);
  replaceMeshView(temp_mesh, samples, false);
}

void DataMgr::replaceMesh(CMesh& src_mesh, CMesh& target_mesh, bool isOriginal)
{
  clearCMesh(target_mesh);
  for(int i = 0; i < src_mesh.vert.size(); i++)
  {
    CVertex v = src_mesh.vert[i];
    v.is_original = isOriginal;
    v.m_index = i;
    target_mesh.vert.push_back(v);
  }
  target_mesh.vn = src_mesh.vn;
  target_mesh.bbox = src_mesh.bbox;
}

void DataMgr::replaceMeshISO(CMesh& src_mesh, CMesh& target_mesh, bool isIso)
{
  clearCMesh(target_mesh);
  for(int i = 0; i < src_mesh.vert.size(); i++)
  {
    CVertex v = src_mesh.vert[i];
    v.is_iso = isIso;
    v.m_index = i;
    target_mesh.vert.push_back(v);
  }
  target_mesh.vn = src_mesh.vn;
  target_mesh.bbox = src_mesh.bbox;
}

void DataMgr::replaceMeshView(CMesh& src_mesh, CMesh& target_mesh, bool isViewGrid)
{
  clearCMesh(target_mesh);
  for(int i = 0; i < src_mesh.vert.size(); i++)
  {
    CVertex v = src_mesh.vert[i];
    v.is_view_grid = isViewGrid;
    v.m_index = i;
    target_mesh.vert.push_back(v);
  }
  target_mesh.vn = src_mesh.vn;
  target_mesh.bbox = src_mesh.bbox;
}
