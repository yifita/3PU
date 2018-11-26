#include "ParameterMgr.h"
#include <iostream>

int ParameterMgr::init_time = 0;
ParameterMgr global_paraMgr;

ParameterMgr::ParameterMgr(void)
{
//    cout << "all parameter set begin" << endl;

  init_time++;
  if(init_time > 1)
  {
    std::cout << "can not init ParameterMgr twice!" << endl;
    return;
  }

  grid_r = 0.22;

  initDataMgrParameter();
  initDrawerParameter();
  initGlareaParameter();
  initNormalSmootherParameter();
  initPoissonParameter();
  initCameraParameter();
  initNBVParameter();

//    cout << "all parameter set end" << endl;
}

ParameterMgr::~ParameterMgr(void)
{

}

void ParameterMgr::setGlobalParameter(QString paraName,Value& val)
{
  if(glarea.hasParameter(paraName))
    glarea.setValue(paraName, val);
  if(data.hasParameter(paraName))
    data.setValue(paraName, val);
  if(drawer.hasParameter(paraName))
    drawer.setValue(paraName, val);
  if(norSmooth.hasParameter(paraName))
    norSmooth.setValue(paraName, val);
  if (poisson.hasParameter(paraName))
    poisson.setValue(paraName, val);
}

void ParameterMgr::initDataMgrParameter()
{
  data.addParam(new RichDouble("Init Radius Para", 2.0));
  data.addParam(new RichDouble("Down Sample Num", 10000));
  data.addParam(new RichDouble("CGrid Radius", grid_r));
  data.addParam(new RichDouble("Outlier Percentage", 0.01));
  data.addParam(new RichDouble("H Gaussian Para", 4));
  data.addParam(new RichDouble("Max Normalize Length", -1.0f));
}

void ParameterMgr::initGlareaParameter()
{
  glarea.addParam(new RichString("Running Algorithm Name", "") );
  glarea.addParam(new RichBool("Light On or Off", true) );
  glarea.addParam(new RichBool("Show Normal", true) );
  glarea.addParam(new RichBool("Show Samples", true) );
  glarea.addParam(new RichBool("Show Samples Quad", false) );
  glarea.addParam(new RichBool("Show Samples Dot", true) );
  glarea.addParam(new RichBool("Show Samples Circle", false) );
  glarea.addParam(new RichBool("Show Samples Sphere", false) );
  glarea.addParam(new RichBool("Show ISO Points", false) );
  glarea.addParam(new RichBool("Use ISO Interval", false) );

  glarea.addParam(new RichBool("Show View Grids", false));
  glarea.addParam(new RichBool("Show NBV Candidates", false));
  glarea.addParam(new RichBool("Show Scan Candidates", false));
  glarea.addParam(new RichBool("Show Scan History", false));
  glarea.addParam(new RichBool("Show Scanned Mesh", true));

  glarea.addParam(new RichBool("Show Model", false));
  glarea.addParam(new RichBool("Show Original", false) );
  glarea.addParam(new RichBool("Show Original Quad", false) );
  glarea.addParam(new RichBool("Show Original Dot", true) );
  glarea.addParam(new RichBool("Show Original Circle", false) );
  glarea.addParam(new RichBool("Show Original Sphere", false) );

  glarea.addParam(new RichBool("Show Skeleton", false));

  glarea.addParam(new RichBool("Show Radius", false));
  glarea.addParam(new RichBool("Show All Radius", false));
  glarea.addParam(new RichBool("Show Radius Use Pick", true));
  glarea.addParam(new RichBool("Show Red Radius Line", true));
  glarea.addParam(new RichBool("Multiply Pick Point", true) );

  glarea.addParam(new RichBool("Show Bounding Box", true));
  glarea.addParam(new RichBool("Show NBV Label", false));
  glarea.addParam(new RichBool("Show NBV Ball", false));


  glarea.addParam(new RichBool("GLarea Busying", false) );
  glarea.addParam(new RichBool("Algorithm Stop", false) );


  glarea.addParam(new RichPoint3f("Light Position", vcg::Point3f(-4.0, -4.0, -4.0)));
  glarea.addParam(new RichColor("Light Ambient Color", QColor(55, 55, 55)));
    //glarea.addParam(new RichColor("Light Diffuse Color", QColor(164, 241, 101)));
  glarea.addParam(new RichColor("Light Diffuse Color", QColor(160, 160, 164)));
  glarea.addParam(new RichColor("Light Specular Color", QColor(255, 255, 255)));

    //glarea.addParam(new RichPoint3f("Light Position", vcg::Point3f(4.0, 4.0, 4.0)));
    //glarea.addParam(new RichColor("Light Ambient Color", QColor(0.0, 0.0, 0.0)));
    //glarea.addParam(new RichColor("Light Diffuse Color", QColor(204, 204, 204)));
    //glarea.addParam(new RichColor("Light Specular Color", QColor(255, 255, 255)));

  glarea.addParam(new RichDouble("Snapshot Resolution", 2));
  glarea.addParam(new RichDouble("Snapshot Index", 1));
  glarea.addParam(new RichDouble("Radius Ball Transparency", 0.3));

  glarea.addParam(new RichDouble("ISO Interval Size", 50));
  glarea.addParam(new RichDouble("Sample Confidence Color Scale", 0.5));

  glarea.addParam(new RichDouble("Grid ISO Color Scale", 0.5));
  glarea.addParam(new RichDouble("Grid ISO Value Shift", -0.5));//new

  glarea.addParam(new RichDouble("Point ISO Color Scale", 0.5)); //new
  glarea.addParam(new RichDouble("Point ISO Value Shift", -0.5));

  glarea.addParam(new RichBool("Show View Grid Slice", false));

  glarea.addParam(new RichBool("SnapShot Each Iteration", false));
  glarea.addParam(new RichBool("No Snap Radius", false));
  glarea.addParam(new RichBool("All Octree Nodes", false));
  glarea.addParam(new RichBool("Show Poisson Surface", false));
}

void ParameterMgr::initDrawerParameter()
{
  drawer.addParam(new RichBool("Doing Pick", false));
  drawer.addParam(new RichBool("Need Cull Points", false) );
  drawer.addParam(new RichBool("Use Pick Original", false));
  drawer.addParam(new RichBool("Use Pick Mode2", true) );
  drawer.addParam(new RichBool("Skeleton Light", true));
  drawer.addParam(new RichBool("Show Individual Color", true));
  drawer.addParam(new RichBool("Use Color From Normal", false));
  drawer.addParam(new RichBool("Use Differ Branch Color", false));
  drawer.addParam(new RichBool("Show Confidence Color", true));

  drawer.addParam(new RichDouble("Original Draw Width", 0.0015));
  drawer.addParam(new RichDouble("Sample Draw Width", 0.005));
  drawer.addParam(new RichDouble("Sample Dot Size", 6));
  drawer.addParam(new RichDouble("ISO Dot Size", 4));
  drawer.addParam(new RichDouble("Original Dot Size", 10));
  drawer.addParam(new RichDouble("Normal Line Width", 2));
  drawer.addParam(new RichDouble("Normal Line Length", 0.25f));

  drawer.addParam(new RichColor("Background Color", QColor(255, 255, 255) ));
  drawer.addParam(new RichColor("Normal Line Color", QColor(0, 0, 255) ));
  drawer.addParam(new RichColor("Sample Point Color", QColor(255, 0, 0) ));
  drawer.addParam(new RichColor("Original Point Color", QColor(48, 48, 48) ));
  drawer.addParam(new RichColor("Feature Color", QColor(0, 0, 255) ));
  drawer.addParam(new RichColor("Pick Point Color", QColor(128, 128, 0) ));
  drawer.addParam(new RichColor("Pick Point DNN Color", QColor(0, 0, 155) ));

  drawer.addParam(new RichColor("Skeleton Bone Color", QColor(200, 0, 0) ));
  drawer.addParam(new RichColor("Skeleton Node Color", QColor(50, 250, 50) ));
  drawer.addParam(new RichColor("Skeleton Branch Color", QColor(0, 0, 0)));
    drawer.addParam(new RichDouble("Skeleton Bone Width", 100)); // ./10000
    drawer.addParam(new RichDouble("Skeleton Node Size", 180)); // ./10000
    drawer.addParam(new RichDouble("Skeleton Branch Size", 30)); // abandoned
  }

  void ParameterMgr::initNormalSmootherParameter()
  {
    norSmooth.addParam(new RichString("Algorithm Name", "NormalSmooth") );

    norSmooth.addParam(new RichInt("PCA KNN", 50));
    norSmooth.addParam(new RichDouble("CGrid Radius", grid_r));
    norSmooth.addParam(new RichDouble("Sharpe Feature Bandwidth Sigma", 30));
    norSmooth.addParam(new RichBool("Run Anistropic PCA", false));
    norSmooth.addParam(new RichBool("Run Init Samples Using Normal", false));
    norSmooth.addParam(new RichInt("Number Of Iterate", 1));
    norSmooth.addParam(new RichInt("Number of KNN", 400));
    norSmooth.addParam(new RichDouble("PCA Threshold", 0.8));
  }

  void ParameterMgr::initPoissonParameter()
  {
    poisson.addParam(new RichString("Algorithm Name", "Poisson") );
    poisson.addParam(new RichDouble("CGrid Radius", 0.08) );
    poisson.addParam(new RichDouble("View Candidates Distance", 0.85));
    poisson.addParam(new RichBool("Run One Key PoissonConfidence", false));
    poisson.addParam(new RichBool("Run Extract All Octree Nodes", false));
    poisson.addParam(new RichBool("Run Extract MC Points", false));
    poisson.addParam(new RichBool("Run Poisson On Original", false));
    poisson.addParam(new RichBool("Run Generate Poisson Field", false));
    poisson.addParam(new RichBool("Run Cut Slice Points", false));

    poisson.addParam(new RichBool("Run Poisson On Samples", false));
    poisson.addParam(new RichBool("Run Label ISO Points", false));
    poisson.addParam(new RichBool("Run ISO Confidence Smooth", false));
    poisson.addParam(new RichBool("Run Label Boundary Points", false));
    poisson.addParam(new RichBool("Run Compute View Candidates", false));
    poisson.addParam(new RichBool("Run View Candidates Clustering", false));
    poisson.addParam(new RichBool("Run Normalize Field Confidence", false));

    poisson.addParam(new RichBool("Run Slice", false));
    poisson.addParam(new RichBool("Run Clear Slice", false));
    poisson.addParam(new RichDouble("Max Depth", 7));

    poisson.addParam(new RichBool("Show Slices Mode", false));
    poisson.addParam(new RichBool("Parallel Slices Mode", false));
    poisson.addParam(new RichBool("Run Estimate Original KNN", false));

    poisson.addParam(new RichBool("Show X Slices", false));
    poisson.addParam(new RichBool("Show Y Slices", false));
    poisson.addParam(new RichBool("Show Z Slices", false));
    poisson.addParam(new RichBool("Show Transparent Slices", false));
    poisson.addParam(new RichDouble("Current X Slice Position", 0.5));
    poisson.addParam(new RichDouble("Current Y Slice Position", 0.5));
    poisson.addParam(new RichDouble("Current Z Slice Position", 0.5));
    poisson.addParam(new RichDouble("Show Slice Percentage", 0.75));
    poisson.addParam(new RichDouble("Poisson Disk Sample Number", 3000));
    poisson.addParam(new RichDouble("Original KNN", 251));

    poisson.addParam(new RichBool("Use Confidence 1", false));
    poisson.addParam(new RichBool("Use Confidence 2", false));
    poisson.addParam(new RichBool("Use Confidence 3", false));
    poisson.addParam(new RichBool("Use Confidence 4", true));
    poisson.addParam(new RichBool("Use Confidence 5", false));
    poisson.addParam(new RichBool("Compute Original Confidence", false));
    poisson.addParam(new RichBool("Compute Sample Confidence", false));
    poisson.addParam(new RichBool("Compute ISO Confidence", false));
    poisson.addParam(new RichBool("Compute Hole Confidence", false));
    poisson.addParam(new RichBool("Use Sort Confidence Combination", true));
    poisson.addParam(new RichBool("Compute New ISO Confidence", false));
    poisson.addParam(new RichBool("Run Smooth Grid Confidence", false));
    poisson.addParam(new RichBool("Run Ball Pivoting Reconstruction", false));

    poisson.addParam(new RichInt("Field Points Resolution", -1));
  }

void ParameterMgr::initCameraParameter()
{
  camera.addParam(new RichString("Algorithm Name", "Camera") );
  camera.addParam(new RichBool("Run One Key NewScans", false));

  camera.addParam(new RichBool("Run Initial Scan", false));
  camera.addParam(new RichBool("Run NBV Scan", false));
  camera.addParam(new RichBool("Run Virtual Scan", false));
  camera.addParam(new RichBool("Is Init Camera Show", false));
  camera.addParam(new RichBool("Show Camera Border", true));


  camera.addParam(new RichDouble("Camera Far Distance", 40.0f));   //cm anno 25
  camera.addParam(new RichDouble("Camera Near Distance", 20.0f));  //cm anno: 10
  camera.addParam(new RichDouble("Predicted Model Size", 20.0f));  //cm anno: 37 lion:20
  camera.addParam(new RichDouble("Camera Horizon Dist", 55.0f));   //cm anno: 14
  camera.addParam(new RichDouble("Camera Vertical Dist", 55.0f));  //cm anno: 18
  camera.addParam(new RichDouble("Camera Dist To Model", 20.0f)); //cm  anno: 30 ((30.0f + 17.0f)/2 + 0.6 * 20.0f)

  camera.addParam(new RichDouble("Max Displacement", 0.0020));
  //camera.addParam(new RichDouble("Camera Resolution", 1.0f / 60.0f));
  camera.addParam(new RichDouble("Camera Resolution", 1.0f / 240.0f));

  //  camera.addParam(new RichDouble("Camera Far Distance", 40.0f));   //cm anno 25
  //  camera.addParam(new RichDouble("Camera Near Distance", 20.0f));  //cm anno: 10
  //  camera.addParam(new RichDouble("Predicted Model Size", 20.0f));  //cm anno: 37 lion:20
  //  camera.addParam(new RichDouble("Camera Horizon Dist", 20.0f));   //cm anno: 14
  //  camera.addParam(new RichDouble("Camera Vertical Dist", 18.0f));  //cm anno: 18
  //  camera.addParam(new RichDouble("Camera Dist To Model", 20.0f)); //cm  anno: 30 ((30.0f + 17.0f)/2 + 0.6 * 20.0f)

  //camera.addParam(new RichDouble("Camera Far Distance", 24.8f));   //cm
  //camera.addParam(new RichDouble("Camera Near Distance", 19.6f));  //cm
  //camera.addParam(new RichDouble("Camera Far Distance", 22.2f));   //cm
  //camera.addParam(new RichDouble("Camera Near Distance", 17.0f));  //cm
  //camera.addParam(new RichDouble("Camera Far Distance", 23.2f));   //cm
  //camera.addParam(new RichDouble("Camera Near Distance", 19.0f));  //cm

  camera.addParam(new RichDouble("Optimal Plane Width", 4.0f));    //cm
  camera.addParam(new RichDouble("Camera FOV Angle", 28.07)); // tan-1(Vert_dist/2 / far_dist)
  //20 for sphere test, 30 for dancing children


  camera.addParam(new RichDouble("Merge Confidence Threshold", 0.9f));
  camera.addParam(new RichDouble("Grid Step Size", -1));

//  cout << "set camera parameters " << endl;
}

void ParameterMgr::initNBVParameter()
{
  nbv.addParam(new RichString("Algorithm Name", "NBV"));
  nbv.addParam(new RichBool("Run One Key NBV", false));
  nbv.addParam(new RichInt("NBV Iteration Count", 3));
  nbv.addParam(new RichBool("Run Build Grid", false));
  nbv.addParam(new RichBool("Run Propagate", false));
  nbv.addParam(new RichBool("Run Propagate One Point", false));
  nbv.addParam(new RichBool("Run Grid Segment", false));
  nbv.addParam(new RichDouble("Merge Probability Pow", 1));
  nbv.addParam(new RichBool("Run Viewing Clustering", false));
  nbv.addParam(new RichBool("Run View Prune", false));
  nbv.addParam(new RichBool("Run Extract Views Into Bins", false));
  nbv.addParam(new RichBool("Run Viewing Extract", false));
  nbv.addParam(new RichDouble("Iso Bottom Delta", 0.05));
  nbv.addParam(new RichBool("Run Set Iso Bottom Confidence", false));
  nbv.addParam(new RichBool("Run Update View Directions", false));
  nbv.addParam(new RichBool("Run Compute View Candidate Index", false));
  nbv.addParam(new RichDouble("View Grid Resolution", 100.8f));
  nbv.addParam(new RichBool("Test Other Inside Segment", false));

  nbv.addParam(new RichBool("Use Confidence Separation", false));
  //nbv.addParam(new RichBool("Use Average Confidence", false));
  nbv.addParam(new RichBool("Use NBV Test1", false));
  nbv.addParam(new RichBool("Use Max Propagation", true));
  nbv.addParam(new RichDouble("Confidence Separation Value", 0.85));
  nbv.addParam(new RichDouble("Max Ray Steps Para", 1.5));
  nbv.addParam(new RichDouble("Ray Resolution Para", 0.511111111111111));
  nbv.addParam(new RichDouble("View Preserve Angle Threshold", 55));
  nbv.addParam(new RichDouble("Confidence Filter Threshold", 0.6f));
  nbv.addParam(new RichDouble("Propagate One Point Index", 0));
  nbv.addParam(new RichInt("View Bin Each Axis", 10));
  nbv.addParam(new RichDouble("View Prune Confidence Threshold", 0.9));
  nbv.addParam(new RichInt("NBV Top N", 100));
  nbv.addParam(new RichBool("Need Update Direction With More Overlaps", true));
  nbv.addParam(new RichDouble("Max Displacement", 0.0020));
  nbv.addParam(new RichBool("NBV Lock PaintGL", false));
}
