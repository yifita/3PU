#include "Camera.h"

vcc::Camera::Camera(RichParameterSet* _para)
{
  para = _para;
}

void vcc::Camera::setInput(DataMgr* pData)
{
  if (!pData->isOriginalEmpty())
  {
    target = pData->getCurrentModel();
    original = pData->getCurrentOriginal();

    occluder = pData->getCurrentModel();
    occludee = pData->getCurrentOriginal();


    //scan candidates for initialing
    scan_count = pData->getScanCount();
    init_scan_candidates = pData->getInitCameraScanCandidates();
    //candidates for nbv computing
    scan_candidates = pData->getScanCandidates();
    scan_history = pData->getScanHistory();
    current_scanned_mesh = pData->getCurrentScannedMesh();
    scanned_results = pData->getScannedResults();
    nbv_candidates = pData->getNbvCandidates();

    far_horizon_dist = global_paraMgr.camera.getDouble("Camera Horizon Dist")
      / global_paraMgr.camera.getDouble("Predicted Model Size");
    far_vertical_dist = global_paraMgr.camera.getDouble("Camera Vertical Dist")
      / global_paraMgr.camera.getDouble("Predicted Model Size");

    far_distance = global_paraMgr.camera.getDouble("Camera Far Distance")
      / global_paraMgr.camera.getDouble("Predicted Model Size");
    near_distance = global_paraMgr.camera.getDouble("Camera Near Distance")
      / global_paraMgr.camera.getDouble("Predicted Model Size");

    dist_to_model = global_paraMgr.camera.getDouble("Camera Dist To Model")
      / global_paraMgr.camera.getDouble("Predicted Model Size");

    resolution = global_paraMgr.camera.getDouble("Camera Resolution");
    cout << "get camera resolution " << resolution << endl;
  }else
  {
    cout<<"ERROR: Camera::setInput empty!!" << endl;
    return;
  }
}

void vcc::Camera::run()
{
  if (para->getBool("Run Virtual Scan"))
  {
    cout << "Run Virtual Scan 1" << endl;
    runVirtualScan();
    return ;
  }
  if (para->getBool("Run Initial Scan"))
  {
    runInitialScan();
    return ;
  }
  if (para->getBool("Run NBV Scan"))
  {
    runNBVScan();
    return;
  }
  if (para->getBool("Run One Key NewScans"))
  {
    runOneKeyNewScan();
    return;
  }
}


void vcc::Camera::runVirtualScan()
{
  //point current_scanned_mesh to a new address
  // current_scanned_mesh = new CMesh;
  current_scanned_mesh->face.clear();
  current_scanned_mesh->fn = 0;
  current_scanned_mesh->vert.clear();
  current_scanned_mesh->vn = 0;
  current_scanned_mesh->bbox = Box3f();
  resolution = global_paraMgr.camera.getDouble("Camera Resolution");
  cout << "get camera resolution " << resolution << endl;
  double max_displacement = global_paraMgr.camera.getDouble("Max Displacement"); //8.0f;//global_paraMgr.nbv.getDouble("Max Displacement"); //resolution * 2; //for adding noise
  computeUpAndRight();
  Point3f viewray = direction.Normalize();
  //compute the end point of viewray
  Point3f viewray_end = pos + viewray * far_distance;

  cout << "camera position: " << pos.X() << " " << pos.Y() << " " << pos.Z() << endl;

  cout << "ray " << viewray.X() << " " << viewray.Y() << " " << viewray.Z() << endl;

  cout << "Run Virtual Scan 2" << endl;

  //sweep and scan
  int n_point_hr_half  = static_cast<int>(0.5 * far_horizon_dist / resolution);
  int n_point_ver_half = static_cast<int>(0.5 * far_vertical_dist / resolution);

  cout << "resolution " << resolution << " n_point_hr_half " << n_point_hr_half << " n_point_ver_half " << n_point_ver_half << endl;
  int index = 0;
  for (int i = - n_point_hr_half; i < n_point_hr_half; ++i)
  {
    double i_res = i * resolution;
    for (int j = - n_point_ver_half; j < n_point_ver_half; ++j)
    {
      Point3f viewray_end_iter = viewray_end + right * i_res + up * (j * resolution);
      Point3f viewray_iter = viewray_end_iter - pos;
      //line direction vector
      Point3f line_dir = viewray_iter.Normalize();
      Point3f intersect_point, intersect_er, intersect_ee;
      Point3f intersect_point_normal, normal_er, normal_ee;

      bool is_barely_visible = false;

      double dist_ee = GlobalFun::computeMeshLineIntersectPoint(occludee, pos, line_dir, intersect_ee, normal_ee, is_barely_visible);

      double dist_er = GlobalFun::computeMeshLineIntersectPoint(occluder, pos, line_dir, intersect_er, normal_er, is_barely_visible);

      bool hit_occluder = false;

      double dist;
      if(dist_er < dist_ee)
      {
          intersect_point = intersect_er;
          intersect_point_normal = normal_er;
          dist = dist_er;
          hit_occluder  = true;
      }
      else
      {
          intersect_point = intersect_ee;
          intersect_point_normal = normal_ee;
          dist = dist_ee;
      }

      if ( dist <= far_distance && dist >= near_distance)
      {
        //add some random noise
        //srand(time(NULL));
        double rndax = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;
        double rnday = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;
        double rndaz = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;

        CVertex t;
        t.is_scanned = true;
        t.is_barely_visible= is_barely_visible;

        t.m_index = index++;
        t.P() = intersect_point + Point3f(rndax, rnday, rndaz);//noise 1
        t.N() = intersect_point_normal; //set out direction as approximate normal

        if(hit_occluder) {
          t.C() = vcg::Color4b(255, 0, 0, 255);
        }
        else{
          t.C() = vcg::Color4b(0, 255, 0, 255);
        }
        current_scanned_mesh->vert.push_back(t);
        current_scanned_mesh->bbox.Add(t.P());
        //cout << "add one point" << endl;
      }
    }
  }

  current_scanned_mesh->vn = current_scanned_mesh->vert.size();
}




//void vcc::Camera::runVirtualScan()
//{
//  current_scanned_mesh->face.clear();
//  current_scanned_mesh->fn = 0;
//  current_scanned_mesh->vert.clear();
//  current_scanned_mesh->vn = 0;
//  current_scanned_mesh->bbox = Box3f();

//  //***AABB
////  vcg::tri::Dodecahedron<CMesh>(*target);

////  vcg::tri::UpdateFlags<CMesh>::Clear(*target);
////  vcg::tri::UpdateNormal<CMesh>::PerVertexNormalized(*target);
////  vcg::tri::UpdateComponentEP<CMesh>::Set(*target);

////  gIndex.Set(target->face.begin(), target->face.end());

////  const bool TEST_BACK_FACES = true;
//  //***AABB

//  //*** IntersectionRayMesh

//  //*** IntersectionRayMesh

//  double max_displacement = global_paraMgr.nbv.getDouble("Max Displacement"); //8.0f;//global_paraMgr.nbv.getDouble("Max Displacement"); //resolution * 2; //for adding noise
//  computeUpAndRight();
//  Point3f viewray = direction.Normalize();
//  //compute the end point of viewray
//  Point3f viewray_end = pos + viewray * far_distance;

//  cout << "camera position: " << pos.X() << " " << pos.Y() << " " << pos.Z() << endl;
//  cout << "ray " << viewray.X() << " " << viewray.Y() << " " << viewray.Z() << endl;
//  cout << "Run Virtual Scan 2" << endl;

//  //sweep and scan
//  int n_point_hr_half  = static_cast<int>(0.5 * far_horizon_dist / resolution);
//  int n_point_ver_half = static_cast<int>(0.5 * far_vertical_dist / resolution);

//  cout << "resolution " << resolution << " n_point_hr_half " << n_point_hr_half << " n_point_ver_half " << n_point_ver_half << endl;
//  int index = 0;
//  for (int i = - n_point_hr_half; i < n_point_hr_half; ++i)
//  {
//    double i_res = i * resolution;
//    for (int j = - n_point_ver_half; j < n_point_ver_half; ++j)
//    {
//      Point3f viewray_end_iter = viewray_end + right * i_res + up * (j * resolution);
//      Point3f viewray_iter = viewray_end_iter - pos;
//      //line direction vector
//      Point3f line_dir = viewray_iter.Normalize();
//      Point3f intersect_point;
//      Point3f intersect_point_normal;
//      bool is_barely_visible = false;

//      //***AABB
////      double dist = 10000;

////      vcg::RayTriangleIntersectionFunctor<TEST_BACK_FACES> rayIntersector;
////      const CIndex::ScalarType maxDist = std::numeric_limits<CIndex::ScalarType>::max();
////      const CIndex::CoordType rayOrigin((CIndex::ScalarType)pos.X(), (CIndex::ScalarType)pos.Y(), (CIndex::ScalarType)pos.Z());
////      const CIndex::CoordType rayDirection((CIndex::ScalarType)line_dir.X(), (CIndex::ScalarType)line_dir.Y(), (CIndex::ScalarType)line_dir.Z());
////      const vcg::Ray3<CIndex::ScalarType, false> ray(rayOrigin, rayDirection);

////      CIndex::ObjPtr isectFace;
////      CIndex::ScalarType rayT;
////      CIndex::CoordType isectPt;

////      vcg::EmptyClass a;
////      isectFace = gIndex.DoRay(rayIntersector, a , ray, maxDist, rayT);

////      if(i < - n_point_hr_half + 5 && j < - n_point_ver_half + 5)
////      {
////          printf("DoRay Test:\n");
////          if (isectFace != 0) {
////              printf("\tface  : 0x%p\n", isectFace);
////              printf("\tray t : %f\n", rayT);
////          }
////          else {
////              printf("\tno object found (index is probably empty).\n");

////          }
////      }

////      dist = rayT;
////      //intersect_point = pos + line_dir * dist;
////      intersect_point = (isectFace->V(0))->P();
////      intersect_point_normal = -line_dir;
//      //***AABB


//      //*** IntersectionRayMesh
//      vcg::Line3<float> ray;
//      ray.SetOrigin(pos);
//      ray.SetDirection(line_dir);


//      float b1, b2, b3;
//      CMesh::FacePointer fi;

//      bool inter = IntersectionRayMesh<CMesh>(target, ray, intersect_point, b1, b2, b3, fi);

////      bool inter = IntersectionRayMesh<CMesh>(target, ray, intersect_point);
//      //*** IntersectionRayMesh

//      //double dist = GlobalFun::computeMeshLineIntersectPoint(target, pos, line_dir, intersect_point, intersect_point_normal, is_barely_visible);

////      if(i < - n_point_hr_half + 5 && j < - n_point_ver_half + 5)
////            cout << "dist " << dist << " far_distance " << far_distance << " near_distance " << near_distance << endl;

//      //cout << intersect_point.X() << " " << intersect_point.Y() << " " << intersect_point.Z() << endl;

////      if ( dist <= far_distance && dist >= near_distance)
//      if (inter)
//      {
//        //add some random noise
//        //srand(time(NULL));
//        double rndax = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;
//        double rnday = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;
//        double rndaz = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;

//        CVertex t;
//        t.is_scanned = true;
//        t.is_barely_visible= is_barely_visible;

//        t.m_index = index++;
//        t.P() = intersect_point + Point3f(rndax, rnday, rndaz);//noise 1
//        t.N() = intersect_point_normal; //set out direction as approximate normal
//        current_scanned_mesh->vert.push_back(t);
//        current_scanned_mesh->bbox.Add(t.P());

//        //cout << "add one point" << endl;
//      }
//    }
//  }

//  current_scanned_mesh->vn = current_scanned_mesh->vert.size();

//}





//void vcc::Camera::runVirtualScan()
//{
//  //point current_scanned_mesh to a new address
//  //current_scanned_mesh = new CMesh;

//  current_scanned_mesh->face.clear();
//  current_scanned_mesh->fn = 0;
//  current_scanned_mesh->vert.clear();
//  current_scanned_mesh->vn = 0;
//  current_scanned_mesh->bbox = Box3f();

//  double max_displacement = global_paraMgr.nbv.getDouble("Max Displacement"); //8.0f;//global_paraMgr.nbv.getDouble("Max Displacement"); //resolution * 2; //for adding noise
//  computeUpAndRight();
//  Point3f viewray = direction.Normalize();
//  //compute the end point of viewray
//  Point3f viewray_end = pos + viewray * far_distance;

//  cout << "camera position: " << pos.X() << " " << pos.Y() << " " << pos.Z() << endl;

//  cout << "ray " << viewray.X() << " " << viewray.Y() << " " << viewray.Z() << endl;

//  cout << "Run Virtual Scan 2" << endl;

//  //sweep and scan
//  int n_point_hr_half  = static_cast<int>(0.5 * far_horizon_dist / resolution);
//  int n_point_ver_half = static_cast<int>(0.5 * far_vertical_dist / resolution);

//  cout << "resolution " << resolution << " n_point_hr_half " << n_point_hr_half << " n_point_ver_half " << n_point_ver_half << endl;
//  int index = 0;
//  for (int i = - n_point_hr_half; i < n_point_hr_half; ++i)
//  {
//    double i_res = i * resolution;
//    for (int j = - n_point_ver_half; j < n_point_ver_half; ++j)
//    {
//      Point3f viewray_end_iter = viewray_end + right * i_res + up * (j * resolution);
//      Point3f viewray_iter = viewray_end_iter - pos;
//      //line direction vector
//      Point3f line_dir = viewray_iter.Normalize();
//      Point3f intersect_point;
//      Point3f intersect_point_normal;
//      bool is_barely_visible = false;
//      double dist = GlobalFun::computeMeshLineIntersectPoint(target, pos, line_dir, intersect_point, intersect_point_normal, is_barely_visible);

////      if(i < - n_point_hr_half + 5 && j < - n_point_ver_half + 5)
////            cout << "dist " << dist << " far_distance " << far_distance << " near_distance " << near_distance << endl;
//      //cout << intersect_point.X() << " " << intersect_point.Y() << " " << intersect_point.Z() << endl;

//      if ( dist <= far_distance && dist >= near_distance)
//      {
//        //add some random noise
//        //srand(time(NULL));
//        double rndax = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;
//        double rnday = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;
//        double rndaz = (double(2.0f * rand()) / RAND_MAX - 1.0f ) * max_displacement;

//        CVertex t;
//        t.is_scanned = true;
//        t.is_barely_visible= is_barely_visible;



//        t.m_index = index++;
//        t.P() = intersect_point + Point3f(rndax, rnday, rndaz);//noise 1
//        t.N() = intersect_point_normal; //set out direction as approximate normal
//        current_scanned_mesh->vert.push_back(t);
//        current_scanned_mesh->bbox.Add(t.P());

//        //cout << "add one point" << endl;
//      }
//    }
//  }

//  //cout << "Run Virtual Scan 3" << endl;

//  current_scanned_mesh->vn = current_scanned_mesh->vert.size();

//   //cout << "current_scanned_mesh->vn " << current_scanned_mesh->vn << endl;

//  //cout << "Run Virtual Scan 3.1" << endl;

//  //increase the scan count;
//  //(*scan_count)++;

//  //cout << "Run Virtual Scan 3.2" << endl;

//  //std::cout<<"scan count right after virtual scan: "<<*scan_count <<std::endl;

//  //cout << "Run Virtual Scan 3.3" << endl;
//}



void vcc::Camera::runInitialScan()
{
  //clear original points
  GlobalFun::clearCMesh(*original);

  //release scanned_result
  vector<CMesh* >::iterator it_scanned_result = scanned_results->begin();
  for (; it_scanned_result != scanned_results->end(); ++it_scanned_result)
  {
    if ( (*it_scanned_result) != NULL)
    {
      delete (*it_scanned_result);
      (*it_scanned_result) = NULL;
    }
  }
  scanned_results->clear();

  //release scan history
  scan_history->clear();

  //run initial scan
  vector<ScanCandidate>::iterator it = init_scan_candidates->begin();
  int i = 1;
  for (; it != init_scan_candidates->end(); ++it)
  {
    pos = it->first;
    direction = it->second;
    /******* call runVirtualScan() *******/
    cout<<i << "th initial scan begin" <<endl;
    runVirtualScan();
    cout<<i++ <<"th initial scan done!" <<endl;

    scan_history->push_back(*it);

    //merge scanned mesh with original
    int index = 0;
    if (!original->vert.empty()) index = original->vert.back().m_index + 1;

    for (int i = 0; i < current_scanned_mesh->vert.size(); ++i)
    {
      CVertex& v = current_scanned_mesh->vert[i];
      CVertex t = v;
      t.m_index = index++;
      t.is_original = true;
      original->vert.push_back(t);
      original->bbox.Add(t.P());
    }
    original->vn = original->vert.size();
  }

}

void vcc::Camera::runNBVScan()
{
  //release scanned_result
  vector< CMesh* >::iterator it_scanned_result = scanned_results->begin();
  for (; it_scanned_result != scanned_results->end(); ++it_scanned_result)
  {
    if ( (*it_scanned_result) != NULL)
    {
      delete (*it_scanned_result);
      (*it_scanned_result) = NULL;
    }
  }
  scanned_results->clear();

  //traverse the scan_candidates and do virtual scan
  vector<ScanCandidate>::iterator it = scan_candidates->begin();
  cout<<"scan candidates size: " <<scan_candidates->size() <<endl;
  int i = 1;
  for (; it != scan_candidates->end(); ++it)
  {
    pos = it->first;
    direction = it->second;
    /********* call runVirtualScan() *******/
    cout<< i << "th candidate Begins!" <<endl;
    runVirtualScan();
    cout<< i++ << "th candidate Ends!" <<endl;

    scan_history->push_back(*it);
    scanned_results->push_back(current_scanned_mesh);
    cout << "scanned points:  " << current_scanned_mesh->vert.size() << endl;
  }
}

void vcc::Camera::runOneKeyNewScan()
{
  runNBVScan();
}

void vcc::Camera::computeUpAndRight()
{
  Point3f x_axis(1.0f, 0.0f, 0.0f);
  Point3f z_axis(0.0f, 0.0f, 1.0f);

  Point3f viewray = direction.Normalize();
  if (viewray.Z() > 0)
  {
    up = viewray ^ x_axis;
  }else if (fabs(viewray.Z()) < EPS_SUN)
  {
    up = viewray ^ z_axis;
  }else
  {
    up = x_axis ^ viewray;
  }
  //compute the right vector
  right = viewray ^ up;

  up = up.Normalize();
  right = right.Normalize();
}
